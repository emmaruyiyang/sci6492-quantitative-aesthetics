"""Drop the off-topic images spotted on the contact sheets and refill each
category back to 20, skipping anything already kept or already rejected.

REJECTS lists files by their name at review time; rejected source URLs are
remembered in data/rejected_urls.txt so a refill never pulls them back.
"""

import csv
import importlib.util
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fetch", HERE / "05_c_fetch_dataset.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)

DATA = HERE.parents[0] / "data"
TRAIN = DATA / "train"
CREDITS = DATA / "credits.csv"
REJECTED = DATA / "rejected_urls.txt"

# Reviewed on the contact sheets: wrong subject, a crop instead of a scene,
# or an image that belongs to a different category in this set.
REJECTS = {
    "column": [3, 13, 14, 19],
    "table": [1, 2, 6, 20],
    "arch_model": [8, 9, 10, 11, 13, 18, 19, 20],
    "window": [7, 13, 15],
    "ordered": [6, 8, 9, 15, 16],
    "disordered": [4, 5, 6, 14, 15, 17, 18],
}

# Extra queries used only when refilling, to pull in subjects the first pass missed.
EXTRA = {
    "column": ["cloister courtyard columns", "hypostyle hall", "church nave columns"],
    "table": ["dining room table set", "workshop table interior", "classroom tables"],
    "arch_model": [
        "architecture studio model workshop",
        "wooden architectural model building",
        "site model architecture exhibition",
    ],
    "window": ["window seat interior", "casement window room", "clerestory windows hall"],
    "ordered": [
        "aligned rows of desks classroom",
        "stacked containers port rows",
        "cemetery rows of headstones",
        "vineyard rows aerial",
    ],
    "disordered": [
        "hoarder house clutter",
        "flea market stall piles",
        "construction site mess",
        "landfill waste pile",
    ],
}


def load_credits():
    with open(CREDITS) as fh:
        return list(csv.DictReader(fh))


def main():
    rows = load_credits()
    by_file = {(r["category"], r["file"]): r for r in rows}
    rejected_urls = set()
    if REJECTED.exists():
        rejected_urls = {ln.strip() for ln in REJECTED.read_text().splitlines() if ln.strip()}

    # 1. drop the rejects
    for cat, idxs in REJECTS.items():
        for i in idxs:
            name = f"{cat}_{i:02d}.jpg"
            row = by_file.pop((cat, name), None)
            if row:
                rejected_urls.add(row["image_url"])
            path = TRAIN / cat / name
            if path.exists():
                path.unlink()
            print(f"  - dropped {name}")

    kept = [r for r in rows if (r["category"], r["file"]) in by_file]
    REJECTED.write_text("\n".join(sorted(rejected_urls)) + "\n")

    # 2. refill each category
    for cat in fetch.CATEGORIES:
        current = [r for r in kept if r["category"] == cat]
        need = fetch.PER_CATEGORY - len(current)
        if need <= 0:
            continue
        print(f"\n== refill {cat}: need {need} ==")
        have_urls = {r["image_url"] for r in current} | rejected_urls
        hashes = set()
        for r in current:
            try:
                from PIL import Image

                hashes.add(fetch.ahash(Image.open(TRAIN / cat / r["file"])))
            except Exception:
                pass
        creators = {}
        for r in current:
            creators[r["creator"]] = creators.get(r["creator"], 0) + 1

        queries = EXTRA.get(cat, []) + fetch.CATEGORIES[cat]
        added = 0
        for page in range(1, 6):
            for query in queries:
                if added >= need:
                    break
                try:
                    results = fetch.search(query, page)
                except Exception as exc:
                    print(f"  ! {query} p{page}: {exc}")
                    continue
                for item in results:
                    if added >= need:
                        break
                    url = item.get("url")
                    creator = item.get("creator") or "?"
                    if not url or url in have_urls:
                        continue
                    if creators.get(creator, 0) >= fetch.MAX_PER_CREATOR:
                        continue
                    have_urls.add(url)
                    try:
                        img = fetch.fetch_image(url)
                    except Exception:
                        continue
                    if min(img.size) < fetch.MIN_SIDE:
                        continue
                    digest = fetch.ahash(img)
                    if any(sum(a != b for a, b in zip(digest, h)) <= 6 for h in hashes):
                        continue
                    hashes.add(digest)
                    creators[creator] = creators.get(creator, 0) + 1
                    added += 1
                    tmp = TRAIN / cat / f"new_{added:02d}.jpg"
                    img.convert("RGB").save(tmp, quality=92)
                    kept.append(
                        {
                            "category": cat,
                            "file": tmp.name,
                            "query": query,
                            "title": item.get("title", ""),
                            "creator": creator,
                            "license": f"{item.get('license','')} {item.get('license_version','')}".strip(),
                            "source_page": item.get("foreign_landing_url", ""),
                            "image_url": url,
                        }
                    )
                    print(f"  + {tmp.name}  <- {query}")
                    time.sleep(0.2)
            if added >= need:
                break
        if added < need:
            print(f"  ! {cat} short by {need - added}")

    # 3. renumber so every category reads <cat>_01..20
    final = []
    for cat in fetch.CATEGORIES:
        rowset = [r for r in kept if r["category"] == cat]
        for i, r in enumerate(rowset, 1):
            new = f"{cat}_{i:02d}.jpg"
            src = TRAIN / cat / r["file"]
            dst = TRAIN / cat / ("tmp_" + new)
            src.rename(dst)
            r["file"] = new
            final.append(r)
    for cat in fetch.CATEGORIES:
        for p in (TRAIN / cat).glob("tmp_*.jpg"):
            p.rename(p.with_name(p.name[4:]))

    with open(CREDITS, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(final[0].keys()))
        writer.writeheader()
        writer.writerows(final)
    for cat in fetch.CATEGORIES:
        print(f"{cat}: {len(list((TRAIN / cat).glob('*.jpg')))}")


if __name__ == "__main__":
    main()
