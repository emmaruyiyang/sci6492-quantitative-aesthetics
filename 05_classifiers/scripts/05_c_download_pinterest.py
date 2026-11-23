"""Download the Pinterest pins collected in data/pinterest_ids/*.txt.

Each line is a space-separated list of "<size>|<filename>" tokens, where
<size> is the CDN path segment ("o" for originals) and the first three byte
pairs of the filename give the CDN's shard directories. Pin IDs are scraped
from the logged-in search pages; the CDN itself needs no auth.

Images land in data/train_pinterest/<category>/ as JPEG, 20 per category.
The Openverse set in data/train/ is left untouched.
"""

import csv
import io
from pathlib import Path

import requests
from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "data"
IDS = DATA / "pinterest_ids"
OUT = DATA / "train_pinterest"

QUERIES = {
    "column": "column architecture",
    "table": "table interior",
    "arch_model": "architectural model",
    "window": "window architecture",
    "ordered": "order symmetry architecture",
    "disordered": "chaos clutter interior",
    "galaxy": "nebula galaxy astrophotography",
    "cell": "cell microscopy photography",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    "Referer": "https://www.pinterest.com/",
}


def urls_for(token):
    """Primary URL plus fallbacks: Pillow cannot read the .heic originals,
    and the CDN serves a .jpg at the resized paths for those pins."""
    size, fname = token.split("|", 1)
    seg = "originals" if size == "o" else size
    a, b, c = fname[0:2], fname[2:4], fname[4:6]
    urls = [f"https://i.pinimg.com/{seg}/{a}/{b}/{c}/{fname}"]
    if fname.lower().endswith((".heic", ".webp")):
        jpg = fname.rsplit(".", 1)[0] + ".jpg"
        urls += [
            f"https://i.pinimg.com/originals/{a}/{b}/{c}/{jpg}",
            f"https://i.pinimg.com/736x/{a}/{b}/{c}/{jpg}",
            f"https://i.pinimg.com/564x/{a}/{b}/{c}/{jpg}",
        ]
    return urls


def load(token):
    last = None
    for url in urls_for(token):
        try:
            r = requests.get(url, headers=HEADERS, timeout=40)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content))
            img.load()
            return img, url
        except Exception as exc:
            last = exc
    raise last


def main():
    rows = []
    for cat, query in QUERIES.items():
        tokens = (IDS / f"{cat}.txt").read_text().split()
        folder = OUT / cat
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.jpg"):
            old.unlink()
        print(f"\n== {cat} ({len(tokens)} pins) ==")
        kept = 0
        for token in tokens:
            try:
                img, url = load(token)
            except Exception as exc:
                print(f"  ! {token}: {exc}")
                continue
            kept += 1
            path = folder / f"{cat}_{kept:02d}.jpg"
            img.convert("RGB").save(path, quality=92)
            rows.append(
                {
                    "category": cat,
                    "file": path.name,
                    "query": query,
                    "source": "pinterest",
                    "width": img.width,
                    "height": img.height,
                    "image_url": url,
                }
            )
            print(f"  [{kept:2d}] {path.name}  {img.width}x{img.height}")
        if kept < 20:
            print(f"  ! {cat}: only {kept}")

    with open(DATA / "credits_pinterest.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} images -> {OUT}")


if __name__ == "__main__":
    main()
