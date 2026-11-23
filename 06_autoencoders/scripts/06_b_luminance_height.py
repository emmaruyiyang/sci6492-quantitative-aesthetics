"""Brightness as elevation: a height field that is not a depth estimate.

Depth Anything reads an astrophotograph as a lit surface and answers with a
smooth ramp, because "bright means near" is the wrong prior for deep space:
brightness there is emission, not distance. The filaments and dust lanes that
carry all the structure come out flat. This conversion drops the depth model
and maps luminance straight to height, which for a nebula is the honest
reading: what glows, stands up.

Two steps make it usable as a relief rather than a black plane with spikes:

  percentile clip   the faintest 0.5% and the brightest 0.1% of the pixels set
                    the ends of the range, so a handful of foreground stars
                    cannot eat the whole scale
  asinh stretch     the astrophotographer's curve: near-linear on the bright
                    core, steep on the faint wings, so nebulosity a linear map
                    would leave at height 2/255 becomes readable relief

Output matches dataset_tools' depth conversion -- one 8-bit grayscale PNG per
source image, same name, same size -- so the result drops into the app's
dataset folder beside <name>_depth and the two can be compared directly.

    uv run --project ../autoencoder_app_v2 --env-file ../autoencoder_app_v2/.env \
        python 06_b_luminance_height.py <src folder> <dst folder> [stretch=10] [low=0.5] [high=99.9]
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULTS = {
    "stretch": 10.0,   # asinh steepness; 0 = plain linear
    "low": 0.5,        # percentile mapped to height 0
    "high": 99.9,      # percentile mapped to height 255
}

# Rec. 709 luminance, applied to the sRGB values as they are stored. Linearising
# first would push the faint wings even further down, which the stretch then has
# to undo, so this stays in the encoded space on purpose.
WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def height(image: Image.Image, stretch: float, low: float, high: float) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    lum = rgb @ WEIGHTS

    lo, hi = np.percentile(lum, [low, high])
    if hi - lo < 1e-6:
        lo, hi = float(lum.min()), float(lum.max() + 1e-6)
    x = np.clip((lum - lo) / (hi - lo), 0.0, 1.0)

    if stretch > 0:
        x = np.arcsinh(stretch * x) / np.arcsinh(stretch)

    return Image.fromarray(np.round(x * 255).astype(np.uint8), mode="L")


def parse(args: list[str]) -> dict:
    params = dict(DEFAULTS)
    for arg in args:
        name, _, value = arg.partition("=")
        if name not in DEFAULTS:
            raise SystemExit(f"unknown parameter {name} (have {', '.join(DEFAULTS)})")
        params[name] = float(value)
    return params


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    params = parse(sys.argv[3:])

    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"no images in {src}")
    dst.mkdir(parents=True, exist_ok=True)

    print(f"{src} -> {dst}")
    print(f"{len(paths)} images, " + ", ".join(f"{k}={v:g}" for k, v in params.items()))
    started = time.time()
    for i, path in enumerate(paths, 1):
        with Image.open(path) as image:
            height(image, **params).save(dst / f"{path.stem}.png")
        if i % 100 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}", flush=True)
    print(f"{len(paths)} images in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
