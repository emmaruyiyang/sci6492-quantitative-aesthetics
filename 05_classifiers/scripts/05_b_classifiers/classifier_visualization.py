"""Run a trained classifier on a folder of images and show, for every class, the
TOP_N images it scores highest for that class.

One row per class, best match on the left. The folder is a flat folder of
images, no labels needed: it's a way to see what the classifier "thinks" each
class looks like, including on images of things it was never trained on.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm

from classifier_model import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_FOLDER = str(Path(__file__).resolve().parents[2] / "models")
MODEL_NAME = "art_classifier_quick_CLIP_DATACOMP"

TEST_DATA_FOLDER = str(Path(__file__).resolve().parents[2] / "datasets" / "eval" / "ostrich")

TOP_N = 8                   # images per class (columns)
ACTIVATION = "softmax"      # "softmax" (classes compete) or "sigmoid" (independent scores)
BATCH_SIZE = 32

FONT = "Courier New"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    plt.rcParams["font.family"] = FONT
    classifier = Classifier.from_folder(Path(MODELS_FOLDER) / MODEL_NAME).to(DEVICE)
    print(classifier)

    paths = sorted(p for p in Path(TEST_DATA_FOLDER).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"{len(paths)} images in {TEST_DATA_FOLDER}")

    # scores[i, c] = score of image i for class c
    batches = []
    for start in tqdm(range(0, len(paths), BATCH_SIZE), desc="classifying"):
        images = [Image.open(p) for p in paths[start:start + BATCH_SIZE]]
        batches.append(classifier.predict(images, ACTIVATION))
    scores = torch.cat(batches)

    n_classes = len(classifier.class_names)
    n_cols = min(TOP_N, len(paths))
    fig, axes = plt.subplots(n_classes, n_cols, figsize=(1.8 * n_cols, 2.0 * n_classes), squeeze=False)

    for c, class_name in enumerate(classifier.class_names):
        top_scores, top_indices = scores[:, c].topk(n_cols)
        for col, (score, i) in enumerate(zip(top_scores.tolist(), top_indices.tolist())):
            ax = axes[c, col]
            ax.imshow(Image.open(paths[i]).convert("RGB"))
            ax.set_title(f"{score:.2f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[c, 0].set_ylabel(class_name, fontsize=10, rotation=0, ha="right", va="center")

    fig.suptitle(f"{MODEL_NAME}: top {n_cols} per class ({ACTIVATION})")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
