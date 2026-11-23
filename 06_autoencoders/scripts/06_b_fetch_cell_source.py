"""Collect high resolution micrographs for the 06 Part 2 topography exercise.

These are the *source* images, not the dataset: 06_b_make_cell_tiles.py crops
them into the square tiles the autoencoder is trained on. What matters here is
pixels and variety of texture, so the filters are size first, then one hash per
image to drop the duplicate scans Openverse returns for the same specimen.

Source: Openverse (openverse.org) -> Wikimedia, Flickr, museums, so everything
is Creative Commons or public domain and credits.csv can go in the report.

  uv run python scripts/06_b_fetch_cell_source.py
"""

import csv
import io
import time
from pathlib import Path

import requests
from PIL import Image

API = "https://api.openverse.org/v1/images/"
HEADERS = {"User-Agent": "sci6492-assignment/1.0 (coursework dataset)"}

OUT = Path(__file__).resolve().parents[1] / "data" / "source" / "cell"
CREDITS = OUT.parent / "cell_credits.csv"

TARGET = 260          # source images wanted, before cropping
MIN_SIDE = 800        # anything smaller cannot give a 512 tile with room to spare
MAX_LONG_SIDE = 3000  # downscale the huge museum scans, they cost disk for nothing
MAX_PER_QUERY = 10    # keeps one search from filling the set
MAX_PER_CREATOR = 4   # stops one photostream from doing the same
PAGES = 3

# Grouped only for reading; the script walks them flat. The spread matters more
# than the count: stains and SEM give completely different depth topographies.
QUERIES = [
    # histology, stained sections
    "light micrograph tissue section", "histology slide stained", "haematoxylin eosin section",
    "microscope slide cross section plant", "plant stem cross section micrograph",
    "leaf cross section microscope", "root cross section micrograph", "wood anatomy micrograph",
    "epidermis cells microscope", "onion cells microscope",
    # single celled, colonies
    "diatom microscope", "diatoms arrangement", "radiolaria microscope", "foraminifera shell",
    "algae microscope", "volvox microscope", "desmid microscope", "cyanobacteria micrograph",
    "paramecium microscope", "amoeba microscope", "yeast cells micrograph",
    # SEM / EM, strong relief
    "scanning electron micrograph pollen", "pollen grains SEM", "scanning electron micrograph surface",
    "transmission electron micrograph cell", "SEM micrograph crystal surface",
    "scanning electron micrograph insect", "electron micrograph bacteria",
    # cultures, fluorescence
    "fluorescence microscopy cells", "confocal microscopy cells", "immunofluorescence micrograph",
    "cell culture microscope", "stem cells microscope", "neuron microscope image",
    "blood smear microscope", "red blood cells micrograph", "bone tissue micrograph",
    "muscle tissue micrograph", "cartilage histology", "kidney histology",
    # anything that reads as packed cellular texture
    "cellular structure microscope", "micrograph texture", "mycelium microscope",
    "fungal spores microscope", "moss cells microscope", "fern spores microscope",
    "crystal thin section polarized light", "petrographic thin section",
]


def search(query: str, page: int) -> list[dict]:
    r = requests.get(API, params={"q": query, "page_size": 20, "page": page,
                                  "license_type": "all-cc", "mature": "false"},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])


def fetch_image(url: str) -> Image.Image:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    img.load()
    return img


def ahash(img: Image.Image) -> str:
    """Cheap perceptual hash, used to drop near-identical shots of one subject."""
    small = img.convert("L").resize((8, 8))
    pixels = list(small.getdata())
    mean = sum(pixels) / len(pixels)
    return "".join("1" if p > mean else "0" for p in pixels)


def near_duplicate(digest: str, seen: set[str]) -> bool:
    return any(sum(a != b for a, b in zip(digest, h)) <= 6 for h in seen)


def shrink(img: Image.Image) -> Image.Image:
    if max(img.size) <= MAX_LONG_SIDE:
        return img
    scale = MAX_LONG_SIDE / max(img.size)
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.jpg"):
        old.unlink()

    seen_urls, hashes = set(), set()
    per_creator, per_query = {}, {}
    rows, kept = [], 0

    for page in range(1, PAGES + 1):
        for query in QUERIES:
            if kept >= TARGET:
                break
            if per_query.get(query, 0) >= MAX_PER_QUERY:
                continue
            try:
                results = search(query, page)
            except Exception as exc:
                print(f"  ! search failed ({query} p{page}): {exc}")
                continue

            for item in results:
                if kept >= TARGET or per_query.get(query, 0) >= MAX_PER_QUERY:
                    break
                url, creator = item.get("url"), item.get("creator") or "?"
                if not url or url in seen_urls:
                    continue
                if per_creator.get(creator, 0) >= MAX_PER_CREATOR:
                    continue
                seen_urls.add(url)
                # the listed size is enough to skip the small ones without downloading
                if min(item.get("width") or 0, item.get("height") or 0) and \
                        min(item["width"], item["height"]) < MIN_SIDE:
                    continue
                try:
                    img = fetch_image(url)
                except Exception:
                    continue
                if min(img.size) < MIN_SIDE:
                    continue
                digest = ahash(img)
                if near_duplicate(digest, hashes):
                    continue

                hashes.add(digest)
                kept += 1
                per_creator[creator] = per_creator.get(creator, 0) + 1
                per_query[query] = per_query.get(query, 0) + 1
                path = OUT / f"cell_{kept:03d}.jpg"
                shrink(img).convert("RGB").save(path, quality=95)
                rows.append({"file": path.name, "query": query,
                             "title": item.get("title", ""), "creator": creator,
                             "license": f"{item.get('license', '')} {item.get('license_version', '')}".strip(),
                             "source_page": item.get("foreign_landing_url", ""),
                             "image_url": url,
                             "width": img.width, "height": img.height})
                print(f"  [{kept:3d}/{TARGET}] {path.name}  {img.width}x{img.height}  <- {query}")
                time.sleep(0.15)
        if kept >= TARGET:
            break

    with open(CREDITS, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{kept} source images -> {OUT}\ncredits -> {CREDITS}")


if __name__ == "__main__":
    main()
