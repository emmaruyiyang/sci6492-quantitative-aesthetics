"""Download the expanded hw6 sets (cell / galaxy) from the pin ids in
data/hw6_ids/<category>.txt.

Separate from the classifier sets: these land in data/hw6/<category>/ and
leave data/train/ and data/train_pinterest/ untouched.

    python 05_c_download_hw6.py galaxy [target]

Parallel downloads, perceptual-hash dedupe, minimum size filter, and a
credits csv per category.
"""

import csv
import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "data"
IDS = DATA / "hw6_ids"
OUT = DATA / "hw6"

TARGET = 500
MIN_SIDE = 300
WORKERS = 8
HAMMING = 6  # perceptual-hash distance below which two images count as the same

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    "Referer": "https://www.pinterest.com/",
}


def urls_for(token):
    size, fname = token.split("|", 1)
    seg = "originals" if size == "o" else size
    a, b, c = fname[0:2], fname[2:4], fname[4:6]
    urls = [f"https://i.pinimg.com/{seg}/{a}/{b}/{c}/{fname}"]
    if fname.lower().endswith((".heic", ".webp")):
        jpg = fname.rsplit(".", 1)[0] + ".jpg"
        urls += [
            f"https://i.pinimg.com/originals/{a}/{b}/{c}/{jpg}",
            f"https://i.pinimg.com/736x/{a}/{b}/{c}/{jpg}",
        ]
    return urls


def fetch(token):
    for url in urls_for(token):
        try:
            r = requests.get(url, headers=HEADERS, timeout=40)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content))
            img.load()
            return img, url
        except Exception:
            continue
    return None, None


def ahash(img):
    small = img.convert("L").resize((8, 8))
    px = list(small.getdata())
    mean = sum(px) / len(px)
    return "".join("1" if p > mean else "0" for p in px)


def main():
    cat = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv) > 2 else TARGET
    tokens = (IDS / f"{cat}.txt").read_text().split()
    folder = OUT / cat
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.jpg"):
        old.unlink()

    lock = threading.Lock()
    hashes = []
    rows = []
    stats = {"small": 0, "dup": 0, "fail": 0}

    def work(token):
        if len(rows) >= target:
            return
        img, url = fetch(token)
        if img is None:
            with lock:
                stats["fail"] += 1
            return
        if min(img.size) < MIN_SIDE:
            with lock:
                stats["small"] += 1
            return
        h = ahash(img)
        with lock:
            if len(rows) >= target:
                return
            if any(sum(a != b for a, b in zip(h, k)) <= HAMMING for k in hashes):
                stats["dup"] += 1
                return
            hashes.append(h)
            idx = len(rows) + 1
            rows.append(
                {
                    "category": cat,
                    "file": f"{cat}_{idx:04d}.jpg",
                    "source": "pinterest",
                    "width": img.width,
                    "height": img.height,
                    "image_url": url,
                }
            )
        img.convert("RGB").save(folder / f"{cat}_{idx:04d}.jpg", quality=90)
        if idx % 50 == 0:
            print(f"  {idx}/{target}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, tokens))

    with open(DATA / f"credits_hw6_{cat}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(
        f"{cat}: kept {len(rows)} of {len(tokens)} pins "
        f"(too small {stats['small']}, duplicate {stats['dup']}, failed {stats['fail']})"
    )


if __name__ == "__main__":
    main()
