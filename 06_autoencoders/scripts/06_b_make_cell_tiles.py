"""Crop the micrographs into the square tiles the Part 2 autoencoder is trained on.

A micrograph is already a tiling of one texture, so cropping is not padding the
set with copies: each patch is a different piece of tissue, and the autoencoder
should see the texture, not the circular field of view and the slide edges that
dominate a whole frame.

Two crop sizes per image (half and a third of the short side) so the set holds
both "a few big cells" and "many small ones" -- that spread is what later gives
the relief its range from coarse bumps to fine grain. Candidates are scored on
contrast and edge energy, and flat or blown-out patches are dropped: a tile with
no structure becomes a dead flat square on the wall.

  uv run python scripts/06_b_make_cell_tiles.py
"""

from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

HERE = Path(__file__).resolve().parents[1]
SOURCES = [HERE / "data" / "source" / "cell",
           HERE.parent / "05_classifiers" / "data" / "train_pinterest" / "cell"]
OUT = HERE.parent / "05_classifiers" / "datasets" / "topography" / "cell"

TILE = 512            # saved size; the app resizes to 64/128 when it trains
SCALES = (0.5, 0.34)  # crop side as a fraction of the image's short side
STRIDE = 0.5          # candidate step, as a fraction of the crop side
PER_IMAGE = 3         # at most, so one specimen cannot dominate a cluster
OVERLAP = 0.3         # reject a candidate overlapping a kept one by more than this
TOTAL = 600

MIN_MEAN, MAX_MEAN = 14, 242    # drop near-black fields of view and blown-out slides
MIN_STD = 18                    # drop flat patches


def score(patch: Image.Image) -> float | None:
    """Contrast x edge energy, or None if the patch is empty of structure."""
    gray = patch.convert("L").resize((128, 128), Image.BILINEAR)
    stat = ImageStat.Stat(gray)
    mean, std = stat.mean[0], stat.stddev[0]
    if not (MIN_MEAN < mean < MAX_MEAN) or std < MIN_STD:
        return None
    edges = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0]
    return std * (1 + edges / 32)


def overlaps(box, kept) -> bool:
    ax, ay, aw = box
    for bx, by, bw in kept:
        ox = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        oy = max(0, min(ay + aw, by + bw) - max(ay, by))
        if ox * oy > OVERLAP * min(aw, bw) ** 2:
            return True
    return False


def candidates(img: Image.Image) -> list[tuple[float, tuple[int, int, int]]]:
    short = min(img.size)
    found = []
    for scale in SCALES:
        side = int(short * scale)
        if side < 256:
            continue
        step = max(1, int(side * STRIDE))
        for y in range(0, img.height - side + 1, step):
            for x in range(0, img.width - side + 1, step):
                s = score(img.crop((x, y, x + side, y + side)))
                if s is not None:
                    found.append((s, (x, y, side)))
    found.sort(key=lambda t: -t[0])
    return found


def pick(img: Image.Image) -> list[tuple[int, int, int]]:
    kept = []
    for _, box in candidates(img):
        if len(kept) >= PER_IMAGE:
            break
        if not overlaps(box, kept):
            kept.append(box)
    return kept


def main() -> None:
    paths = sorted(p for folder in SOURCES if folder.is_dir()
                   for p in folder.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not paths:
        raise SystemExit("no source images -- run 06_b_fetch_cell_source.py first")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.jpg"):
        old.unlink()

    # rank-major: every image gives its best tile before any image gives a second
    per_image = []
    for i, path in enumerate(paths, 1):
        try:
            img = Image.open(path)
            img.load()
            img = img.convert("RGB")
        except Exception:
            print(f"  ! unreadable: {path.name}")
            continue
        boxes = pick(img)
        per_image.append((path, img, boxes))
        print(f"  [{i:3d}/{len(paths)}] {path.name}  {img.width}x{img.height}  -> {len(boxes)} tiles")

    written = 0
    for rank in range(PER_IMAGE):
        for path, img, boxes in per_image:
            if written >= TOTAL:
                break
            if rank >= len(boxes):
                continue
            x, y, side = boxes[rank]
            tile = img.crop((x, y, x + side, y + side)).resize((TILE, TILE), Image.LANCZOS)
            written += 1
            tile.save(OUT / f"cell_{written:04d}.jpg", quality=92)
        if written >= TOTAL:
            break

    print(f"\n{written} tiles from {len(per_image)} source images -> {OUT}")


if __name__ == "__main__":
    main()
