"""Build one contact sheet per category so the dataset can be eyeballed
(and dropped straight into the report, item c).

    python 05_c_contact_sheet.py [train|train_pinterest]
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1] / "data"
SET = sys.argv[1] if len(sys.argv) > 1 else "train"
TRAIN = ROOT / SET
OUT = ROOT / ("contact_sheets" if SET == "train" else f"contact_sheets_{SET.split('_', 1)[-1]}")

CELL = 240
COLS = 5
PAD = 6
LABEL = 28


def sheet(folder):
    files = sorted(folder.glob("*.jpg"))
    rows = (len(files) + COLS - 1) // COLS
    w = COLS * CELL + (COLS + 1) * PAD
    h = LABEL + rows * CELL + (rows + 1) * PAD
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 8), f"{folder.name}  ({len(files)} images)", fill="black")
    for i, path in enumerate(files):
        img = Image.open(path).convert("RGB")
        img.thumbnail((CELL, CELL))
        x = PAD + (i % COLS) * (CELL + PAD) + (CELL - img.width) // 2
        y = LABEL + PAD + (i // COLS) * (CELL + PAD) + (CELL - img.height) // 2
        canvas.paste(img, (x, y))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{folder.name}.jpg"
    canvas.save(out, quality=88)
    print(out)


if __name__ == "__main__":
    for folder in sorted(p for p in TRAIN.iterdir() if p.is_dir()):
        sheet(folder)
