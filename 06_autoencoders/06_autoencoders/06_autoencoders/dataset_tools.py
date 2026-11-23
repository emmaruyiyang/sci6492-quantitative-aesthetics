"""Make a new dataset from an existing one: every image converted.

  grayscale    color removed
  canny        Canny edge detection: white lines where the brightness changes
               sharply. The image is blurred a little first (blur), so noise
               and texture don't become edges. A change stronger than `high`
               starts an edge, one stronger than `low` continues it.
  depth        Depth Anything V2 (a network trained to guess how far away
               every pixel is, from a single image): near = bright, far = dark.
               Relative depth, scaled to 0..255 per image.
  depth_edges  Canny edges of the depth image: the outlines of shapes standing
               in front of others, without the edges of their texture or color.

The folder structure is kept (subfolders = classes stay classes), file names
too; the extension can change with the output format.

The depth conversions need the transformers package (the model is downloaded
from Hugging Face the first time, into ~/.cache/huggingface).
"""

import base64
import io
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

CONVERSIONS = {
    "grayscale": {"label": "grayscale", "params": {}},
    "canny": {"label": "Canny edges",
              "params": {"blur": 2, "low": 100, "high": 200, "invert": False}},
    "depth": {"label": "depth (Depth Anything V2)",
              "params": {"depth_model": "small", "invert": False}},
    "depth_edges": {"label": "depth edges (Depth Anything V2 + Canny)",
                    "params": {"depth_model": "small", "blur": 2, "low": 20, "high": 60, "invert": False}},
}

DEPTH_MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}

FORMATS = {"png": ".png", "jpg": ".jpg", "same": None}
DEPTH_BATCH = 8
LOG_LINES = 200

# One depth model in memory, shared by the previews and the conversion runs.
# The lock stops two threads from using (or replacing) it at the same time.
depth_lock = threading.Lock()
depth_state = {"name": None, "model": None, "processor": None}


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def depth_maps(images: list[Image.Image], model_size: str, device: torch.device) -> list[np.ndarray]:
    """Depth Anything V2 -> one uint8 array per image (its size), near = 255"""
    name = DEPTH_MODELS[model_size]
    with depth_lock:
        # imported here, inside the lock: two threads importing transformers for
        # the first time at once can see a half-initialized module
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise RuntimeError("the depth conversions need the transformers package (uv add transformers)") from e
        if depth_state["name"] != name:
            depth_state.update(name=None, model=None, processor=None)     # free the previous one first
            if device.type == "cuda":
                torch.cuda.empty_cache()
            depth_state["processor"] = AutoImageProcessor.from_pretrained(name)
            depth_state["model"] = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()
            depth_state["name"] = name
        processor, model = depth_state["processor"], depth_state["model"]
        with torch.inference_mode():
            inputs = processor(images=images, return_tensors="pt").to(device)
            predicted = model(**inputs).predicted_depth            # [B, h, w], bigger = nearer
        out = []
        for image, depth in zip(images, predicted):
            # back to the image's own size, then scaled to 0..255
            depth = torch.nn.functional.interpolate(depth[None, None].float(), size=(image.height, image.width),
                                                    mode="bicubic", align_corners=False)[0, 0]
            lo, hi = depth.min(), depth.max()
            depth = (depth - lo) / (hi - lo).clamp(min=1e-6)
            out.append((depth.clamp(0, 1) * 255).round().byte().cpu().numpy())
        return out


def canny(gray: np.ndarray, blur: int, low: float, high: float) -> np.ndarray:
    if blur > 0:
        k = 2 * int(blur) + 1               # kernel size, odd
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return cv2.Canny(gray, float(low), float(high))


def convert(images: list[Image.Image], conversion: str, params: dict, device: torch.device) -> list[Image.Image]:
    """A batch of RGB images -> converted images (grayscale "L" images)"""
    p = {**CONVERSIONS[conversion]["params"], **(params or {})}
    if conversion == "grayscale":
        out = [np.asarray(image.convert("L")) for image in images]
    elif conversion == "canny":
        out = [canny(np.asarray(image.convert("L")), p["blur"], p["low"], p["high"]) for image in images]
    elif conversion == "depth":
        out = depth_maps(images, p["depth_model"], device)
    elif conversion == "depth_edges":
        out = [canny(d, p["blur"], p["low"], p["high"]) for d in depth_maps(images, p["depth_model"], device)]
    else:
        raise ValueError(f"unknown conversion {conversion}")
    if p.get("invert"):
        out = [255 - a for a in out]
    return [Image.fromarray(a, "L") for a in out]


def load(path: Path, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_side:
        image.thumbnail((max_side, max_side))
    return image


def data_url(image: Image.Image, size: int) -> str:
    image = image.copy()
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def target_path(path: Path, source: Path, target: Path, fmt: str) -> Path:
    out = target / path.relative_to(source)
    return out.with_suffix(FORMATS[fmt]) if FORMATS.get(fmt) else out


# ---------------------------------------------------------------------------
# A conversion run (background thread)
# ---------------------------------------------------------------------------

class ConversionRun:

    def __init__(self, paths: list[Path], source: Path, target: Path, conversion: str, params: dict,
                 fmt: str, max_side: int, device: torch.device, on_done=None):
        self.paths, self.source, self.target = paths, Path(source), Path(target)
        self.conversion, self.params, self.fmt, self.max_side = conversion, params, fmt, max_side
        self.device = device
        self.on_done = on_done
        self.state = "running"          # running, stopping, done, stopped, error
        self.done, self.failed = 0, 0
        self.message = ""
        self.log = []
        self.stop_requested = False
        self.started = time.time()
        self.last = None                # data urls of the last converted image (before, after)
        self.thread = threading.Thread(target=self.run, daemon=True)

    @property
    def active(self) -> bool:
        return self.state in ("running", "stopping")

    def start(self) -> None:
        self.write_log(f"{CONVERSIONS[self.conversion]['label']}: {len(self.paths)} images "
                       f"{self.source} -> {self.target}")
        self.thread.start()

    def stop(self) -> None:
        if self.active:
            self.stop_requested = True
            self.state = "stopping"

    def write_log(self, text: str) -> None:
        print(f"[dataset] {text}")
        self.log = (self.log + [f"{time.strftime('%H:%M:%S')}  {text}"])[-LOG_LINES:]

    def status(self) -> dict:
        elapsed = time.time() - self.started
        remaining = len(self.paths) - self.done - self.failed
        rate = (self.done + self.failed) / elapsed if elapsed > 0 else 0
        return {"state": self.state, "done": self.done, "failed": self.failed, "total": len(self.paths),
                "target": str(self.target), "message": self.message, "log": self.log[-40:],
                "elapsed": round(elapsed, 1), "eta": round(remaining / rate, 1) if rate and self.active else None,
                "conversion": CONVERSIONS[self.conversion]["label"], "last": self.last}

    def run(self) -> None:
        batch = DEPTH_BATCH if self.conversion.startswith("depth") else 1
        try:
            for start in range(0, len(self.paths), batch):
                if self.stop_requested:
                    self.state = "stopped"
                    self.write_log(f"stopped after {self.done} images (they stay in {self.target})")
                    return
                paths = self.paths[start:start + batch]
                images, kept = [], []
                for path in paths:
                    try:
                        images.append(load(path, self.max_side))
                        kept.append(path)
                    except Exception as e:
                        self.failed += 1
                        self.write_log(f"can't read {path.name}: {e}")
                if not images:
                    continue
                for path, image, converted in zip(kept, images, convert(images, self.conversion, self.params, self.device)):
                    out = target_path(path, self.source, self.target, self.fmt)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    converted.save(out, **({"quality": 92} if out.suffix.lower() in (".jpg", ".jpeg") else {}))
                    self.done += 1
                self.last = {"before": data_url(images[-1], 256), "after": data_url(converted, 256),
                             "name": kept[-1].relative_to(self.source).as_posix()}
            self.state = "done"
            self.write_log(f"finished: {self.done} images in {self.target}"
                           + (f", {self.failed} could not be read" if self.failed else ""))
        except Exception as e:
            self.state = "error"
            self.message = f"{type(e).__name__}: {e}"
            self.write_log(self.message)
            traceback.print_exc()
        finally:
            if self.on_done:
                self.on_done()
