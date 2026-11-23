"""Ranking files: pairwise preferences between the images of a dataset, what a
RankNet trains on. No torch here.

A dataset folder can hold any number of them, one per person or question:

    <dataset folder>/ranking_<name>.json

    { "schema":   "diffdraw.taste.samples/1",
      "question": "which one would you hang on your wall?",
      "created", "updated",
      "images":   ["a.jpg", "sub/b.jpg", ...],    # what the indices below refer to
      "pairs":    [[a, b, pref], ...],             # pref 1 = a better, 0 = equal, -1 = b better
      "repeats":  [[a, b, first, second], ...] }   # pairs asked again, both answers

The same format as DiffDrawing's taste page (its samples files load here and
the other way round). "images" makes a file survive changes to the folder:
the indices are matched to the images by name, a pair whose image is gone is
dropped instead of silently comparing other images.

"repeats" measure how consistent the person ranking is: now and then a pair
answered earlier is asked again. If they agree with themselves 80% of the time,
a model that agrees with them 78% of the time has learned about all there is.

DiffDrawing's older files are read too (not written): ranked_pairs.json or
ranked_pairs_<user>.json, a bare list of pairs, with image_order.json next to
them for the names.
"""

import json
import time
from pathlib import Path

SCHEMA = "diffdraw.taste.samples/1"
PREFIX = "ranking_"
LEGACY_PREFIX = "ranked_pairs"
LEGACY_ORDER = "image_order.json"
NAME_CHARACTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")


def is_valid_name(name: str) -> bool:
    return bool(name) and set(name) <= NAME_CHARACTERS and not name.startswith(".")


def ranking_path(folder, name: str) -> Path:
    """The file of a new ranking called name"""
    if not is_valid_name(name):
        raise ValueError("a ranking name can only have letters, digits, _ - and . (not at the start)")
    return Path(folder) / f"{PREFIX}{name}.json"


def ranking_files(folder) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix == ".json"
                  and (p.name.startswith(PREFIX) or p.name.startswith(LEGACY_PREFIX)))


def is_legacy(path: Path) -> bool:
    return Path(path).name.startswith(LEGACY_PREFIX)


def label_of(path: Path) -> str:
    """ranking_pan.json -> pan, ranked_pairs_pan.json -> pan (old format)"""
    stem = Path(path).stem
    if is_legacy(path):
        return (stem[len(LEGACY_PREFIX):].lstrip("_") or "ranked_pairs") + " (old format)"
    return stem[len(PREFIX):]


def read_ranking(path) -> dict:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):           # DiffDrawing's older format
        order = path.parent / LEGACY_ORDER
        images = json.loads(order.read_text(encoding="utf-8")) if order.exists() else []
        return {"question": "", "created": None, "updated": None, "images": images,
                "pairs": raw, "repeats": []}
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path.name}: unknown ranking schema {raw.get('schema')!r}")
    return {"question": raw.get("question", ""), "created": raw.get("created"), "updated": raw.get("updated"),
            "images": raw.get("images") or [], "pairs": raw.get("pairs") or [],
            "repeats": raw.get("repeats") or []}


def remap(ranking: dict, names: list[str]) -> tuple[list[list], list[list], int]:
    """The pairs and repeats re-indexed onto names (the dataset's images as they
    are now). -> (pairs, repeats, number of pairs dropped: an image not in names)

    A file without "images" can only mean "made for exactly these images": its
    indices are taken as they are."""
    stored = ranking["images"]
    if stored:
        index_of = {name: i for i, name in enumerate(names)}
        index = {i: index_of[name] for i, name in enumerate(stored) if name in index_of}
    else:
        index = {i: i for i in range(len(names))}
    pairs, dropped = [], 0
    for entry in ranking["pairs"]:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3 or entry[0] not in index or entry[1] not in index:
            dropped += 1
            continue
        pairs.append([index[entry[0]], index[entry[1]], int(entry[2])])
    repeats = [[index[r[0]], index[r[1]], int(r[2]), int(r[3])] for r in ranking["repeats"]
               if isinstance(r, (list, tuple)) and len(r) >= 4 and r[0] in index and r[1] in index]
    return pairs, repeats, dropped


def self_consistency(repeats: list) -> dict | None:
    """How often the second answer to a repeated pair agreed with the first"""
    if not repeats:
        return None
    sign = lambda v: (v > 0) - (v < 0)
    agree = sum(1 for r in repeats if sign(r[2]) == sign(r[3]))
    return {"agree": agree, "total": len(repeats), "rate": agree / len(repeats)}


def summary(path, names: list[str]) -> dict:
    """What the page lists for a ranking file of a dataset"""
    path = Path(path)
    try:
        ranking = read_ranking(path)
    except (OSError, ValueError) as e:
        return {"file": path.name, "label": label_of(path), "error": str(e)}
    pairs, repeats, dropped = remap(ranking, names)
    return {"file": path.name, "label": label_of(path), "legacy": is_legacy(path),
            "question": ranking["question"], "updated": ranking["updated"],
            "pairs": len(pairs), "dropped": dropped, "strict": sum(1 for p in pairs if p[2] != 0),
            "consistency": self_consistency(repeats)}


def save_ranking(path, names: list[str], pairs: list, repeats: list, question: str = "",
                 created: str = None) -> None:
    """Write a ranking made on names (the dataset's images). Pairs the file
    already had between images that are not in names (made on another dataset
    of the same folder, or images moved away) are kept."""
    path = Path(path)
    images = list(names)
    kept_pairs, kept_repeats = [], []
    if path.exists() and not is_legacy(path):
        old = read_ranking(path)
        created = created or old["created"]
        current = set(names)
        position = {name: i for i, name in enumerate(images)}

        def index(name: str) -> int:
            if name not in position:
                position[name] = len(images)
                images.append(name)
            return position[name]

        stored = old["images"]
        outside = lambda e: stored and (stored[e[0]] not in current or stored[e[1]] not in current)
        valid = lambda e: (isinstance(e, (list, tuple)) and len(e) >= 3
                           and 0 <= e[0] < len(stored) and 0 <= e[1] < len(stored))
        for e in old["pairs"]:
            if valid(e) and outside(e):
                kept_pairs.append([index(stored[e[0]]), index(stored[e[1]]), e[2]])
        for e in old["repeats"]:
            if valid(e) and len(e) >= 4 and outside(e):
                kept_repeats.append([index(stored[e[0]]), index(stored[e[1]]), e[2], e[3]])
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data = {"schema": SCHEMA, "question": question, "created": created or now, "updated": now,
            "images": images, "pairs": [list(p) for p in pairs] + kept_pairs,
            "repeats": [list(r) for r in repeats] + kept_repeats}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)
