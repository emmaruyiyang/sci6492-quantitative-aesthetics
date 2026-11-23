"""Place the images of a folder on a plane whose two axes are the scores of two
classes, chosen from the drop downs.

  x = score of the image for the class picked as "x axis"
  y = score of the image for the class picked as "y axis"

Top right: the classifier thinks the image is both. Bottom left: neither.

With "softmax" checked the scores of an image add up to 1 across all classes,
so x + y can never exceed 1: every image lies below the dashed diagonal, and
images close to it belong to one of the two classes rather than any other.
Unchecked, each class gets an independent sigmoid score and the whole square
can be used.

The window is a small tkinter app (comes with Python) with a matplotlib plot
inside it. The toolbar under the plot zooms and pans, the thumbnails keep their
size, so zooming spreads out crowded areas.
"""

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import torch
from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageOps
from tqdm import tqdm

from classifier_model import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_FOLDER = str(Path(__file__).resolve().parents[2] / "models")
MODEL_NAME = "art_classifier_quick_CLIP_DATACOMP"

TEST_DATA_FOLDER = str(Path(__file__).resolve().parents[2] / "datasets" / "eval" / "ostrich")

THUMBNAIL_SIZE = 56         # pixels, longest side of each thumbnail
THUMBNAIL_BORDER = 2        # pixels of white frame around each thumbnail
BATCH_SIZE = 32

FONT = "Courier New"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_thumbnail(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    image = ImageOps.expand(image, border=THUMBNAIL_BORDER, fill="white")
    return np.asarray(image)


class BiaxialViewer:

    def __init__(self, root: tk.Tk, class_names: list[str], logits: torch.Tensor, thumbnails: list[np.ndarray]):
        self.class_names = class_names
        self.thumbnails = thumbnails
        # Both versions of the scores, computed once: [num_images, num_classes]
        self.scores = {
            True: logits.softmax(dim=-1).numpy(),
            False: logits.sigmoid().numpy(),
        }

        root.title(f"{MODEL_NAME}: biaxial view of {Path(TEST_DATA_FOLDER).name}")

        # same font for the plot (matplotlib) and the controls (tkinter)
        rcParams["font.family"] = FONT
        ttk.Style(root).configure(".", font=(FONT, 10))
        root.option_add("*TCombobox*Listbox.font", (FONT, 10))     # the opened drop down list

        # --- controls -------------------------------------------------------
        controls = ttk.Frame(root, padding=6)
        controls.pack(side=tk.TOP, fill=tk.X)

        self.x_class = tk.StringVar(value=class_names[0])
        self.y_class = tk.StringVar(value=class_names[min(1, len(class_names) - 1)])
        self.use_softmax = tk.BooleanVar(value=True)

        ttk.Label(controls, text="x axis").pack(side=tk.LEFT)
        x_menu = ttk.Combobox(controls, textvariable=self.x_class, values=class_names, state="readonly")
        x_menu.pack(side=tk.LEFT, padx=(4, 16))
        ttk.Label(controls, text="y axis").pack(side=tk.LEFT)
        y_menu = ttk.Combobox(controls, textvariable=self.y_class, values=class_names, state="readonly")
        y_menu.pack(side=tk.LEFT, padx=(4, 16))
        ttk.Checkbutton(controls, text="softmax", variable=self.use_softmax, command=self.redraw).pack(side=tk.LEFT)

        for menu in (x_menu, y_menu):
            menu.bind("<<ComboboxSelected>>", lambda event: self.redraw())

        # --- plot -----------------------------------------------------------
        self.figure = Figure(figsize=(9, 9))
        self.ax = self.figure.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        NavigationToolbar2Tk(self.canvas, root).pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.redraw()

    def redraw(self) -> None:
        softmax = self.use_softmax.get()
        scores = self.scores[softmax]
        x_index = self.class_names.index(self.x_class.get())
        y_index = self.class_names.index(self.y_class.get())

        ax = self.ax
        ax.clear()
        for thumbnail, x, y in zip(self.thumbnails, scores[:, x_index], scores[:, y_index]):
            ax.add_artist(AnnotationBbox(OffsetImage(thumbnail), (x, y), frameon=False, pad=0))

        if softmax:
            ax.plot([0, 1], [1, 0], "--", color="gray", linewidth=1, label="x + y = 1")
            ax.legend(loc="upper right")

        margin = 0.05
        ax.set_xlim(-margin, 1 + margin)
        ax.set_ylim(-margin, 1 + margin)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel(f"{self.x_class.get()} score")
        ax.set_ylabel(f"{self.y_class.get()} score")
        ax.set_title("softmax" if softmax else "sigmoid")
        self.canvas.draw_idle()


def main():
    classifier = Classifier.from_folder(Path(MODELS_FOLDER) / MODEL_NAME).to(DEVICE)
    print(classifier)

    paths = sorted(p for p in Path(TEST_DATA_FOLDER).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"{len(paths)} images in {TEST_DATA_FOLDER}")

    batches = []
    for start in tqdm(range(0, len(paths), BATCH_SIZE), desc="classifying"):
        images = [Image.open(p) for p in paths[start:start + BATCH_SIZE]]
        batches.append(classifier.predict(images, "logits"))
    logits = torch.cat(batches)

    thumbnails = [load_thumbnail(p) for p in tqdm(paths, desc="thumbnails")]

    root = tk.Tk()
    BiaxialViewer(root, classifier.class_names, logits, thumbnails)
    root.mainloop()


if __name__ == "__main__":
    main()
