"""Shrink the hw6 images in place so the long side is at most MAX_SIDE.

The autoencoder resizes and centre-crops to image_size (64 by default, 256 at
most in practice), so anything above a few hundred pixels is dead weight on
disk and decode time. Images already small enough are left alone.

    python 05_c_resize_hw6.py [max_side]
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "data" / "hw6"
MAX_SIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 512
CATEGORIES = ["cell", "galaxy", "arch_model"]


def shrink(path):
    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:
        return ("fail", path.name, str(exc))
    if max(img.size) <= MAX_SIDE:
        return ("kept", path.name, img.size)
    before = img.size
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    img.convert("RGB").save(path, quality=90)
    return ("shrunk", path.name, (before, img.size))


def main():
    for cat in CATEGORIES:
        folder = DATA / cat
        files = sorted(folder.glob("*.jpg"))
        before = sum(f.stat().st_size for f in files)
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(shrink, files))
        after = sum(f.stat().st_size for f in folder.glob("*.jpg"))
        counts = {}
        for kind, *_ in results:
            counts[kind] = counts.get(kind, 0) + 1
        print(
            f"{cat}: {len(files)} images, "
            f"{counts.get('shrunk', 0)} shrunk, {counts.get('kept', 0)} already small, "
            f"{counts.get('fail', 0)} failed  |  "
            f"{before / 1e6:.0f}MB -> {after / 1e6:.0f}MB"
        )
        for kind, name, info in results:
            if kind == "fail":
                print(f"  ! {name}: {info}")


if __name__ == "__main__":
    main()
