"""Live camera test: classify every camera frame and show the score of each class
as a bar next to the image.

The model only sees the square in the middle of the frame (the preprocessing
crops the center), it is outlined in the image.

Keys:
  a     toggle activation between softmax and sigmoid
          softmax  the scores add up to 1, the classes compete. Point the
                   camera at nothing in particular and some class still wins.
          sigmoid  every class gets an independent score. The model was trained
                   with softmax, so these are less calibrated, but they show
                   how strongly each class responds on its own.
  q/Esc quit
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from classifier_model import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_FOLDER = str(Path(__file__).resolve().parents[2] / "models")
MODEL_NAME = "art_classifier_quick_CLIP_DATACOMP"

CAMERA_INDEX = 0            # 0 = first camera, try 1, 2... if you have several
ACTIVATION = "softmax"      # starting activation, toggle with "a"

# Bars move towards the new scores by (1 - SMOOTHING) each frame:
# 0 = no smoothing (flickery), 0.9 = very smooth but slow to react.
SMOOTHING = 0.7

PANEL_WIDTH = 360           # width in pixels of the panel with the bars
BAR_COLOR = (120, 200, 80)  # RGB
OTHER_BAR_COLOR = (110, 110, 110)
TEXT_COLOR = (255, 255, 255)

# OpenCV can only write text in its own built-in fonts, so the panel is drawn
# with PIL, which can use any TrueType font file.
FONT_FILE = "cour.ttf"      # Courier New (in C:\Windows\Fonts)
FONT_SIZE = 15


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_FILE, size)
    except OSError:
        print(f"font {FONT_FILE} not found, using PIL's default font")
        return ImageFont.load_default(size)


FONT = load_font(FONT_SIZE)


def draw_panel(height: int, class_names: list[str], scores: np.ndarray, activation: str, fps: float) -> np.ndarray:
    panel = Image.new("RGB", (PANEL_WIDTH, height), (30, 30, 30))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 10), f"{activation}  {fps:4.1f} fps", font=FONT, fill=TEXT_COLOR)

    top = 45
    row = min(50, (height - top) // max(len(class_names), 1))
    best = int(np.argmax(scores))
    for i, (name, score) in enumerate(zip(class_names, scores)):
        y = top + i * row
        bar_width = int(score * (PANEL_WIDTH - 20))
        color = BAR_COLOR if i == best else OTHER_BAR_COLOR
        draw.text((10, y), f"{name}  {score:.2f}", font=FONT, fill=TEXT_COLOR)
        draw.rectangle((10, y + row // 2, 10 + bar_width, y + row - 6), fill=color)

    # PIL is RGB, OpenCV frames are BGR: reverse the color channels
    return np.asarray(panel)[:, :, ::-1]


def main():
    classifier = Classifier.from_folder(Path(MODELS_FOLDER) / MODEL_NAME).to(DEVICE)
    classifier.eval()
    print(classifier)

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"could not open camera {CAMERA_INDEX}")

    activation = ACTIVATION
    smoothed = np.zeros(len(classifier.class_names))
    fps, last_time = 0.0, time.perf_counter()

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        # OpenCV frames are BGR numpy arrays, the classifier expects RGB PIL images
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scores = classifier.predict([image], activation)[0].numpy()
        smoothed = SMOOTHING * smoothed + (1 - SMOOTHING) * scores

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last_time, 1e-6)
        last_time = now

        # outline the center square the model actually sees
        h, w = frame.shape[:2]
        side = min(h, w)
        x0, y0 = (w - side) // 2, (h - side) // 2
        cv2.rectangle(frame, (x0, y0), (x0 + side - 1, y0 + side - 1), (255, 255, 255), 1)

        panel = draw_panel(h, classifier.class_names, smoothed, activation, fps)
        cv2.imshow(f"{MODEL_NAME}  [a] softmax/sigmoid  [q] quit", np.hstack([frame, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("a"):
            activation = "sigmoid" if activation == "softmax" else "softmax"

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
