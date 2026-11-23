"""Download a CC-licensed training set for the 05_C classifier assignment.

Two sets, scene-level images (no crops), 20 per category:
  concrete: column / table / arch_model / window
  abstract: ordered / disordered

Source: Openverse (openverse.org) -> Flickr, Wikimedia, museums. Everything
returned is Creative Commons or public domain, so the attributions in
credits.csv are enough to put the dataset in the report.
"""

import csv
import hashlib
import io
import time
from pathlib import Path

import requests
from PIL import Image

API = "https://api.openverse.org/v1/images/"
HEADERS = {"User-Agent": "sci6492-assignment/1.0 (coursework dataset)"}

OUT = Path(__file__).resolve().parents[1] / "data" / "train"

PER_CATEGORY = 20
MIN_SIDE = 400
MAX_PER_CREATOR = 2  # stops one Flickr photostream from filling a category
MAX_PER_QUERY = 6  # keeps each category visually varied

CATEGORIES = {
    # --- concrete set ---
    "column": [
        "colonnade",
        "stone columns interior",
        "concrete columns building",
        "pillars hall interior",
        "column capital temple",
        "columns portico",
    ],
    "table": [
        "wooden table room",
        "dining table interior",
        "desk office interior",
        "table and chairs kitchen",
        "meeting table conference room",
        "cafe table interior",
    ],
    "arch_model": [
        "architectural model",
        "architecture scale model",
        "maquette building",
        "massing model architecture",
        "urban planning model city",
        "cardboard study model architecture",
    ],
    "window": [
        "window interior light",
        "large window room",
        "window facade building",
        "stained glass window interior",
        "row of windows wall",
        "bay window house interior",
    ],
    # --- abstract set ---
    "ordered": [
        "symmetrical building facade",
        "grid facade office tower",
        "library shelves rows of books",
        "rows of empty chairs auditorium",
        "stacked chairs neat rows",
        "parking lot rows aerial",
        "solar panels rows",
        "warehouse shelving rows",
    ],
    "disordered": [
        "cluttered room",
        "messy desk papers",
        "chaotic pile of objects",
        "junk shop clutter",
        "disorganised storage room",
        "scrapyard pile metal",
        "tangled cables wires",
        "demolition debris rubble",
    ],
}


def search(query, page):
    r = requests.get(
        API,
        params={
            "q": query,
            "page_size": 20,
            "page": page,
            "license_type": "all-cc",
            "mature": "false",
        },
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("results", [])


def fetch_image(url):
    r = requests.get(url, headers=HEADERS, timeout=40)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    img.load()
    return img


def ahash(img):
    """Cheap perceptual hash, used to drop near-identical shots of one subject."""
    small = img.convert("L").resize((8, 8))
    pixels = list(small.getdata())
    mean = sum(pixels) / len(pixels)
    return "".join("1" if p > mean else "0" for p in pixels)


def collect(name, queries, rows):
    folder = OUT / name
    for old in folder.glob("*.jpg"):
        old.unlink()
    folder.mkdir(parents=True, exist_ok=True)

    seen_urls, hashes = set(), set()
    per_creator, per_query = {}, {}
    kept = 0

    for page in range(1, 5):
        for query in queries:
            if kept >= PER_CATEGORY:
                break
            if per_query.get(query, 0) >= MAX_PER_QUERY:
                continue
            try:
                results = search(query, page)
            except Exception as exc:
                print(f"  ! search failed ({query} p{page}): {exc}")
                continue
            for item in results:
                if kept >= PER_CATEGORY or per_query.get(query, 0) >= MAX_PER_QUERY:
                    break
                url = item.get("url")
                creator = item.get("creator") or "?"
                if not url or url in seen_urls:
                    continue
                if per_creator.get(creator, 0) >= MAX_PER_CREATOR:
                    continue
                seen_urls.add(url)
                try:
                    img = fetch_image(url)
                except Exception:
                    continue
                if min(img.size) < MIN_SIDE:
                    continue
                digest = ahash(img)
                if any(
                    sum(a != b for a, b in zip(digest, h)) <= 6 for h in hashes
                ):
                    continue
                hashes.add(digest)
                kept += 1
                per_creator[creator] = per_creator.get(creator, 0) + 1
                per_query[query] = per_query.get(query, 0) + 1
                path = folder / f"{name}_{kept:02d}.jpg"
                img.convert("RGB").save(path, quality=92)
                rows.append(
                    {
                        "category": name,
                        "file": path.name,
                        "query": query,
                        "title": item.get("title", ""),
                        "creator": creator,
                        "license": f"{item.get('license','')} {item.get('license_version','')}".strip(),
                        "source_page": item.get("foreign_landing_url", ""),
                        "image_url": url,
                    }
                )
                print(f"  [{kept:2d}/{PER_CATEGORY}] {path.name}  <- {query}")
                time.sleep(0.2)
        if kept >= PER_CATEGORY:
            break
    return kept


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, queries in CATEGORIES.items():
        print(f"\n== {name} ==")
        got = collect(name, queries, rows)
        if got < PER_CATEGORY:
            print(f"  ! only {got} images for {name}")
    with open(OUT.parent / "credits.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} images -> {OUT}")


if __name__ == "__main__":
    main()
