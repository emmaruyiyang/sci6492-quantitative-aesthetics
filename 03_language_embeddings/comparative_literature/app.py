"""Two Londons: the same words drawn in two word2vec models side by side.

AI-assisted assignment 2 (comparative literature). The models are trained by train_models.py on
    - The Complete Sherlock Holmes (Arthur Conan Doyle, 1887-1927)
    - The People of the Abyss (Jack London, 1903)

Run (from the sci6492_code folder):
    .venv/bin/python 03_language_embeddings/comparative_literature/app.py
then open http://127.0.0.1:5051
"""

from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"

BOOKS = [
    {
        "key": "holmes",
        "title": "The Complete Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "years": "1887–1927",
        "model": KeyedVectors.load(str(HERE / "sherlock_holmes.kv")),
    },
    {
        "key": "abyss",
        "title": "The People of the Abyss",
        "author": "Jack London",
        "years": "1903",
        "model": KeyedVectors.load(str(HERE / "people_of_the_abyss.kv")),
    },
]
for book in BOOKS:
    kv = book["model"]
    book["words"] = int(sum(kv.get_vecattr(w, "count") for w in kv.index_to_key))
    book["vocabulary"] = len(kv)

# Gehl's vocabulary of life between buildings, kept to words that occur at least 15 times in both books
GEHL_WORDS = [
    "street", "streets", "road", "court", "door", "room", "house", "home", "public", "city",
    "space", "crowd", "people", "night", "light", "walk", "sat", "talked", "look",
]

# neighbours that carry no meaning of their own are skipped
STOP_WORDS = set(ENGLISH_STOP_WORDS) | {"said", "mr", "mrs", "dr", "yes", "oh", "shall", "did", "don't", "i'm"}

# color terms and the hex values used to draw them; several spellings can share one color.
# each book only uses the color words it actually contains (at least min_count times)
COLOR_TERMS = {
    "red": "#e0474c", "orange": "#f08a3c", "yellow": "#f2c94c", "green": "#4caf6e",
    "blue": "#3d7fe0", "purple": "#9b5de5", "pink": "#f28dc0", "brown": "#a0714f",
    "grey": "#9aa0a6", "gray": "#9aa0a6", "white": "#ffffff", "black": "#000000",
    "gold": "#d4af37", "silver": "#c0c0c8",
}

# semantic axes for the "axis" encoding: (from word, to word), both frequent in both books
AXES = {
    "day-night": ("day", "night"),
    "clean-dirty": ("clean", "dirty"),
    "warm-cold": ("warm", "cold"),
    "inside-outside": ("inside", "outside"),
}

app = Flask(__name__)


def neighbours(kv, word, top_n):
    """The top_n nearest meaningful words to word in one model."""
    found = []
    for w, _ in kv.most_similar(word, topn=top_n * 4):
        if w in STOP_WORDS or len(w) < 3 or w.endswith("'s"):
            continue
        found.append(w)
        if len(found) == top_n:
            break
    return found


def radial_layout(query_vec, vecs, sims, lo, hi):
    """Query at the origin; radius from similarity on a scale shared by both books,
    angle from a PCA of the neighbours after removing their component along the query."""
    residual = vecs - np.outer(vecs @ query_vec, query_vec)
    if len(vecs) >= 2:
        xy = PCA(n_components=2).fit_transform(residual)
    else:
        xy = np.ones((len(vecs), 2))
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    t = np.clip((hi - sims) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(sims)
    radius = 0.28 + 0.72 * t
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def pca_layout(query_vec, vecs):
    xy = PCA(n_components=2).fit_transform(np.vstack([query_vec, vecs]))
    xy = xy[1:] - xy[0]
    return xy / np.abs(xy).max()


@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/web/<path:filename>")
def web_file(filename):
    return send_from_directory(WEB, filename)


@app.get("/api/compare")
def compare():
    word = request.args.get("word", "").strip().lower()
    source = request.args.get("source", "seed")
    top_n = max(5, min(request.args.get("top_n", 15, type=int), 40))
    layout = request.args.get("layout", "radial")
    axis_key = request.args.get("axis", "day-night")
    if axis_key not in AXES:
        axis_key = "day-night"

    if not word:
        return jsonify(error="Please enter a word."), 400
    missing = [b["title"] for b in BOOKS if word not in b["model"]]
    if missing:
        return jsonify(error=f"'{word}' is not in the vocabulary of {' and '.join(missing)}."), 404

    # the word set is the same for both books
    if source == "list":
        candidates = [w for w in GEHL_WORDS if w != word]
        own = {b["key"]: set() for b in BOOKS}
    else:
        own = {b["key"]: neighbours(b["model"], word, top_n) for b in BOOKS}
        candidates = list(dict.fromkeys(own["holmes"] + own["abyss"]))  # union, first-seen order
        own = {k: set(v) for k, v in own.items()}

    # each book draws the candidates it contains; the others become "absent" words
    present = {b["key"]: [w for w in candidates if w in b["model"]] for b in BOOKS}

    # one similarity scale for both panels, so ring radii mean the same on both sides
    all_sims = [b["model"].similarity(word, w) for b in BOOKS for w in present[b["key"]]]
    lo, hi = (float(min(all_sims)), float(max(all_sims))) if all_sims else (0.0, 1.0)

    # axis values on one scale for both panels
    axis_raw = {}
    for b in BOOKS:
        kv = b["model"]
        a, z = AXES[axis_key]
        direction = kv.get_vector(z, norm=True) - kv.get_vector(a, norm=True)
        direction /= np.linalg.norm(direction)
        axis_raw[b["key"]] = {w: float(kv.get_vector(w, norm=True) @ direction) for w in present[b["key"]] + [word]}
    axis_scale = max((abs(v) for d in axis_raw.values() for v in d.values()), default=1.0) or 1.0

    panels = []
    for b in BOOKS:
        kv = b["model"]
        other = next(o for o in BOOKS if o is not b)
        query_vec = kv.get_vector(word, norm=True)

        # this book's own palette: only the color words it contains
        palette = {c: h for c, h in COLOR_TERMS.items() if c in kv}
        color_names = list(palette)
        color_vecs = np.array([kv.get_vector(c, norm=True) for c in color_names])

        results = []
        words = present[b["key"]]
        if words:
            vecs = np.array([kv.get_vector(w, norm=True) for w in words])
            sims = vecs @ query_vec
            xy = pca_layout(query_vec, vecs) if layout == "pca" else radial_layout(query_vec, vecs, sims, lo, hi)
            color_sims = vecs @ color_vecs.T
            for i, w in enumerate(words):
                ci = int(np.argmax(color_sims[i]))
                if source == "list":
                    provenance = "list"
                elif w in own[b["key"]] and w in own[other["key"]]:
                    provenance = "both"
                elif w in own[b["key"]]:
                    provenance = "own"
                else:
                    provenance = "other"
                results.append({
                    "word": w,
                    "similarity": float(sims[i]),
                    # None / 0 when the other book does not contain the word
                    "other_similarity": float(other["model"].similarity(word, w)) if w in other["model"] else None,
                    "count": int(kv.get_vecattr(w, "count")),
                    "other_count": int(other["model"].get_vecattr(w, "count")) if w in other["model"] else 0,
                    "x": float(xy[i, 0]),
                    "y": float(xy[i, 1]),
                    "provenance": provenance,
                    "color_term": color_names[ci],
                    "color_term_sim": float(color_sims[i, ci]),
                    "axis": axis_raw[b["key"]][w] / axis_scale,
                })

        # words picked for the comparison that this book never uses (often enough to have a vector)
        absent = [w for w in candidates if w not in kv]

        panels.append({
            "key": b["key"],
            "title": b["title"],
            "author": b["author"],
            "years": b["years"],
            "words": b["words"],
            "vocabulary": b["vocabulary"],
            "query_count": int(kv.get_vecattr(word, "count")),
            "query_color_term": color_names[int(np.argmax(color_vecs @ query_vec))],
            "palette": palette,
            "absent": absent,
            "results": results,
        })

    return jsonify(
        word=word,
        source=source,
        layout=layout,
        axis={"key": axis_key, "from": AXES[axis_key][0], "to": AXES[axis_key][1]},
        axes=list(AXES),
        sim_range=[lo, hi],
        gehl_words=GEHL_WORDS,
        panels=panels,
    )


if __name__ == "__main__":
    # port 5000 is taken by AirPlay Receiver on macOS, 5050 by assignment 1
    print("Open http://127.0.0.1:5051")
    app.run(port=5051, debug=False)
