"""Make a dataset from the images embedded in a product catalogue PDF.

Every embedded image is extracted once (catalogues are laid out in spreads:
both pages of a spread share their images, so an image is taken from the page
where it is actually visible) and sorted into two subfolders:

  products  cutouts on a white background: at least 3 of the 4 image borders
            are (nearly) white
  scenes    photographs: rooms, gardens, people; anything without a white border

Photos with a pale background (a white wall, snow) can land in products.

Images smaller than `min_side` pixels (icons, color swatches) are skipped, and
so are exact duplicates (the same picture embedded twice).

Files are named <prefix>_p<page>_<xref>.<ext>, page counted from 1; JPEGs are
written as embedded (no recompression), other images as PNG.

  python pdf_catalogue_images.py catalogue.pdf ikea_outdoor_2023
  python pdf_catalogue_images.py catalogue.pdf ikea_outdoor_2023 --dry-run

The dataset goes to ROOT_DATASETS_FOLDER/<name> (env_config.json).
"""

import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

import env_config


def white_sides(image: Image.Image, width: float = 0.01, white: int = 215) -> list[float]:
    """For each side (top, bottom, left, right): the fraction of the pixels in a
    thin strip along it that are near white."""
    a = np.asarray(image.convert("RGB"))
    h, w = a.shape[:2]
    bh, bw = max(1, round(h * width)), max(1, round(w * width))
    strips = (a[:bh], a[-bh:], a[:, :bw], a[:, -bw:])
    return [float((s.reshape(-1, 3).min(axis=1) >= white).mean()) for s in strips]


def is_product(image: Image.Image, threshold: float) -> bool:
    """A cutout on white: at least 3 of the 4 sides are mostly white (products
    cropped tight touch the frame on one side, a shelf standing on the bottom)."""
    return sum(f >= threshold for f in white_sides(image)) >= 3


def visible_images(doc: pymupdf.Document) -> list[tuple[int, int]]:
    """(page index, xref) of every image, once, on the page it is shown on."""
    found, seen = [], set()
    for page in doc:
        for info in page.get_image_info(xrefs=True):
            xref = info["xref"]
            if xref <= 0 or xref in seen:  # 0: inline image
                continue
            if (pymupdf.Rect(info["bbox"]) & page.rect).is_empty:
                continue
            seen.add(xref)
            found.append((page.number, xref))
    return found


def extract(pdf: Path, target: Path, prefix: str, min_side: int, threshold: float, dry_run: bool):
    doc = pymupdf.open(pdf)
    hashes = set()
    counts = {"products": 0, "scenes": 0, "small": 0, "duplicate": 0}
    for page_index, xref in visible_images(doc):
        raw = doc.extract_image(xref)
        data, ext = raw["image"], raw["ext"]
        if min(raw["width"], raw["height"]) < min_side:
            counts["small"] += 1
            continue
        digest = hashlib.md5(data).hexdigest()
        if digest in hashes:
            counts["duplicate"] += 1
            continue
        hashes.add(digest)
        if ext in ("jpeg", "jpg") and raw["colorspace"] == 3:
            image = Image.open(io.BytesIO(data))
        else:  # CMYK, indexed, JPX, ...: let pymupdf convert to RGB
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.Pixmap(doc, xref))
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            data, ext = None, "png"
        kind = "products" if is_product(image, threshold) else "scenes"
        counts[kind] += 1
        name = f"{prefix}_p{page_index + 1:03d}_{xref}.{'jpg' if ext == 'jpeg' else ext}"
        if dry_run:
            print(f"{kind:8} {image.width}x{image.height}  {name}")
            continue
        path = target / kind / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if data is not None:
            path.write_bytes(data)
        else:
            image.save(path)
    print(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("name", help="dataset folder name, under ROOT_DATASETS_FOLDER")
    parser.add_argument("--prefix", help="file name prefix (default: the dataset name)")
    parser.add_argument("--min-side", type=int, default=48, help="skip images smaller than this (pixels)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="fraction of near-white pixels that makes a side white (3 white sides: a product)")
    parser.add_argument("--dry-run", action="store_true", help="only print how every image would be sorted")
    args = parser.parse_args()
    target = env_config.ROOT_DATASETS_FOLDER / args.name
    extract(args.pdf, target, args.prefix or args.name, args.min_side, args.threshold, args.dry_run)
    if not args.dry_run:
        print(f"written to {target}")


if __name__ == "__main__":
    main()
