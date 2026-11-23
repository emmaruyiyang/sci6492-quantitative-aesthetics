"""Live camera test: encode every camera frame and decode it again.

    [ what the model sees ]  [ latent vector ]  [ reconstruction ]

The model only sees the square in the middle of the frame, resized to its
image size (64 x 64 by default); that square is shown on the left, scaled up.
The latent vector is drawn standing up, one horizontal stripe per number
(the first one on top): black for 0, cyan for positive, red for negative
(or gray levels).

The dropdowns at the top switch between every autoencoder found under
MODELS_FOLDER, the colors and the normalization. OpenCV windows can't have
dropdowns, so the window is made with tkinter (part of Python).

Keys:
  c     colors: diverging (red / black / cyan) or gray
  n     normalization of the stripes
          global   value / largest |value| seen recently: shows the raw code
          dims     each number compared to its own recent average and spread:
                   shows which numbers change when the picture changes
  q/Esc quit
"""

import os
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

from autoencoder_model import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_FOLDER = r"D:\ML\QA2026\models"
MODEL_NAME = "baroque_ae_cnn"     # selected at start (if found)

CAMERA_INDEX = 0            # 0 = first camera, try 1, 2... if you have several

DISPLAY_SIZE = 384          # input and reconstruction are shown this big (pixels)
BARCODE_WIDTH = 96          # width of the latent vector panel between them
HEADER = 28                 # pixels above the panels for the labels

# Running statistics forget the past at this rate per frame:
# 0.9 = adapt fast, 0.99 = slow and steady.
STATS_MOMENTUM = 0.95

COLORS = ["diverging", "gray"]
NORMALIZATIONS = ["global", "dims"]


def to_bgr(image: Image.Image, size: int) -> np.ndarray:
    # NEAREST keeps the pixels of the small image visible instead of blurring them
    image = image.resize((size, size), Image.NEAREST)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def center_square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


def colorize(values: np.ndarray, colors: str) -> np.ndarray:
    """values in -1..1 -> BGR colors [N, 3] (uint8)"""
    v = np.clip(values, -1, 1)[:, None]
    if colors == "gray":
        return np.repeat((v + 1) * 127.5, 3, axis=1).astype(np.uint8)
    cyan = np.array([255.0, 255.0, 0.0])        # BGR
    red = np.array([0.0, 0.0, 255.0])
    # 0 -> black, +1 -> cyan, -1 -> red
    bgr = np.where(v >= 0, v * cyan, -v * red)
    return bgr.astype(np.uint8)


def draw_barcode(values: np.ndarray, colors: str, width: int, height: int) -> np.ndarray:
    """The vector standing up: one horizontal stripe per number, first on top."""
    stripes = colorize(values, colors)[:, None, :]                  # [N, 1, 3]
    return cv2.resize(stripes, (width, height), interpolation=cv2.INTER_NEAREST)


def find_autoencoders(root: str) -> list[str]:
    """Every folder under root with an autoencoder config.json, relative to root"""
    found = []
    for folder, subfolders, files in os.walk(root):
        subfolders[:] = sorted(s for s in subfolders if not s.startswith((".", "__")))
        if "config.json" in files and folder_model_type(folder) == "autoencoder":
            found.append(Path(folder).relative_to(root).as_posix())
            subfolders[:] = []      # a model's subfolders (snapshots...) are not models
    return found


class RunningStats:
    """Exponential moving average of the mean, the variance (per latent number)
    and the largest |value|, for the two normalizations."""

    def __init__(self, n: int):
        self.mean = np.zeros(n)
        self.variance = np.ones(n)
        self.scale = None

    def update(self, z: np.ndarray) -> None:
        m = STATS_MOMENTUM
        if self.scale is None:          # first frame: start from it
            self.mean = z.copy()
            self.scale = np.abs(z).max() or 1.0
        self.mean = m * self.mean + (1 - m) * z
        self.variance = m * self.variance + (1 - m) * (z - self.mean) ** 2
        self.scale = max(m * self.scale + (1 - m) * np.abs(z).max(), 1e-6)

    def normalize(self, z: np.ndarray, normalization: str) -> np.ndarray:
        if normalization == "global":
            return z / self.scale
        # z-score, 2.5 standard deviations -> full color
        return (z - self.mean) / np.sqrt(self.variance + 1e-8) / 2.5


def label(panel: np.ndarray, text: str) -> np.ndarray:
    header = np.full((HEADER, panel.shape[1], 3), 30, np.uint8)
    cv2.putText(header, text, (8, HEADER - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, panel])


class CameraApp:
    """A tkinter window: dropdowns on top, the camera panels below. Every few
    milliseconds tick() reads a frame, runs it through the autoencoder and
    shows the result."""

    def __init__(self, root: tk.Tk, model_names: list[str]):
        self.root = root
        self.autoencoder = None
        self.stats = None
        self.fps, self.last_time = 0.0, time.perf_counter()
        self.gap = np.full((DISPLAY_SIZE + HEADER, 4, 3), 30, np.uint8)

        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            raise RuntimeError(f"could not open camera {CAMERA_INDEX}")

        root.title("autoencoder camera   [c] colors  [n] normalization  [q] quit")
        root.configure(background="#1e1e1e")
        bar = ttk.Frame(root, padding=6)
        bar.pack(fill="x")

        self.model = tk.StringVar(value=MODEL_NAME if MODEL_NAME in model_names else model_names[0])
        self.colors = tk.StringVar(value=COLORS[0])
        self.normalization = tk.StringVar(value=NORMALIZATIONS[0])
        for text, variable, values, width in (("model", self.model, model_names, 40),
                                              ("colors", self.colors, COLORS, 10),
                                              ("normalization", self.normalization, NORMALIZATIONS, 8)):
            ttk.Label(bar, text=text).pack(side="left", padx=(8, 4))
            box = ttk.Combobox(bar, textvariable=variable, values=values, width=width, state="readonly")
            box.pack(side="left")
            # give the keyboard back to the window, so the keys keep working
            box.bind("<<ComboboxSelected>>", lambda event: self.root.focus_set())
        self.model.trace_add("write", lambda *args: self.load_model())

        self.view = tk.Label(root, background="#1e1e1e")
        self.view.pack()
        root.bind("<Key>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.load_model()
        self.tick()

    def load_model(self) -> None:
        name = self.model.get()
        self.autoencoder = None             # free the previous one first
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        self.autoencoder = Autoencoder.from_folder(Path(MODELS_FOLDER) / name).to(DEVICE).eval()
        self.stats = RunningStats(self.autoencoder.latent_dim)
        print(f"{name}: {self.autoencoder}")

    def on_key(self, event) -> None:
        if event.keysym in ("q", "Escape"):
            self.close()
        elif event.keysym == "c":
            self.colors.set(COLORS[(COLORS.index(self.colors.get()) + 1) % len(COLORS)])
        elif event.keysym == "n":
            n = NORMALIZATIONS
            self.normalization.set(n[(n.index(self.normalization.get()) + 1) % len(n)])

    def tick(self) -> None:
        ok, frame = self.camera.read()
        if not ok:
            self.close()
            return
        autoencoder = self.autoencoder

        # OpenCV frames are BGR numpy arrays, the autoencoder expects RGB PIL images
        image = Image.fromarray(cv2.cvtColor(center_square(frame), cv2.COLOR_BGR2RGB))
        model_input = image.resize((autoencoder.image_size, autoencoder.image_size), Image.BILINEAR)
        reconstructions, z = autoencoder.reconstruct([model_input])
        z = z[0].numpy()
        self.stats.update(z)

        now = time.perf_counter()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.last_time, 1e-6)
        self.last_time = now

        normalization = self.normalization.get()
        barcode = draw_barcode(self.stats.normalize(z, normalization), self.colors.get(),
                               BARCODE_WIDTH, DISPLAY_SIZE)
        s = autoencoder.image_size
        panels = [
            label(to_bgr(model_input, DISPLAY_SIZE), f"input {s}x{s}"),
            self.gap,
            label(barcode, f"{autoencoder.latent_dim} {normalization}"),
            self.gap,
            label(to_bgr(reconstructions[0], DISPLAY_SIZE), f"{autoencoder.TYPE}  {self.fps:4.1f} fps"),
        ]
        rgb = cv2.cvtColor(np.hstack(panels), cv2.COLOR_BGR2RGB)
        # keep a reference to the image, otherwise tkinter shows nothing
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.view.configure(image=self.photo)
        self.root.after(1, self.tick)

    def close(self) -> None:
        self.camera.release()
        self.root.destroy()


def main():
    model_names = find_autoencoders(MODELS_FOLDER)
    if not model_names:
        raise RuntimeError(f"no autoencoders in {MODELS_FOLDER}")
    print(f"{len(model_names)} autoencoders: {model_names}")
    root = tk.Tk()
    CameraApp(root, model_names)
    root.mainloop()


if __name__ == "__main__":
    main()
