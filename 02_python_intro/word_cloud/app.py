"""Semantic word cloud: Flask + Gensim + scikit-learn.

Run (from the sci6492_code folder):
    .venv/bin/python 02_python_intro/word_cloud/app.py
then open http://127.0.0.1:5050
"""

import os
import ssl
from pathlib import Path

import gensim.downloader
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# avoid SSL certificate errors when gensim downloads a model (macOS)
ssl._create_default_https_context = ssl._create_unverified_context

WEB = Path(__file__).resolve().parent / "web"
MODEL_NAME = os.environ.get("MODEL", "glove-wiki-gigaword-50")

app = Flask(__name__)

# loaded once when the server starts, not once per request
print(f"loading {MODEL_NAME} ...")
model: KeyedVectors = gensim.downloader.load(MODEL_NAME)
print("model loaded")

# color terms and the hex values used to draw them.
# in "semantic color" mode each word takes the color whose *word vector* is closest to it.
COLOR_TERMS = {
    "red": "#e0474c",
    "orange": "#f08a3c",
    "yellow": "#f2c94c",
    "green": "#4caf6e",
    "blue": "#3d7fe0",
    "purple": "#9b5de5",
    "pink": "#f28dc0",
    "brown": "#a0714f",
    "gray": "#9aa0a6",
    "white": "#ffffff",
    "black": "#000000",
}
COLOR_VECS = np.array([model.get_vector(c, norm=True) for c in COLOR_TERMS])

# the front-end draws clusters in black and white with a typographic style per cluster;
# these colors are only kept in the API response for reference
CLUSTER_PALETTE = ["#f5f5f7", "#d2d2d7", "#aeaeb2", "#8e8e93", "#6e6e73", "#48484a"]


def radial_layout(query_vec, vecs, sims):
    """Place the query at the origin.

    radius : grows with semantic distance (1 - cosine similarity) to the query
    angle  : taken from a 2D PCA of the neighbours, so neighbours that are similar
             to each other end up in the same direction around the query
    """
    # remove the part of each neighbour that points at the query,
    # so PCA describes how neighbours differ from *each other*
    residual = vecs - np.outer(vecs @ query_vec, query_vec)
    xy = PCA(n_components=2).fit_transform(residual)
    angles = np.arctan2(xy[:, 1], xy[:, 0])

    dist = 1.0 - sims
    lo, hi = dist.min(), dist.max()
    t = (dist - lo) / (hi - lo) if hi > lo else np.zeros_like(dist)
    radius = 0.28 + 0.72 * t  # keep a ring of empty space around the query
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def pca_layout(query_vec, vecs):
    """Plain PCA of query + neighbours, shifted so the query sits at the origin."""
    xy = PCA(n_components=2).fit_transform(np.vstack([query_vec, vecs]))
    xy = xy[1:] - xy[0]
    return xy / np.abs(xy).max()


@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/web/<path:filename>")
def web_file(filename):
    return send_from_directory(WEB, filename)


@app.get("/api/cloud")
def cloud():
    word = request.args.get("word", "").strip().lower()
    top_n = max(5, min(request.args.get("top_n", 30, type=int), 100))
    layout = request.args.get("layout", "radial")
    n_clusters = max(2, min(request.args.get("clusters", 4, type=int), len(CLUSTER_PALETTE)))

    if not word:
        return jsonify(error="Please enter a word."), 400
    if word not in model:
        return jsonify(error=f"'{word}' is not in the model's vocabulary."), 404

    neighbours = model.most_similar(word, topn=top_n)
    words = [w for w, _ in neighbours]
    sims = np.array([s for _, s in neighbours])
    query_vec = model.get_vector(word, norm=True)
    vecs = np.array([model.get_vector(w, norm=True) for w in words])

    xy = pca_layout(query_vec, vecs) if layout == "pca" else radial_layout(query_vec, vecs, sims)

    # group neighbours into semantic clusters; name each cluster by the word closest to its centre
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(vecs)
    cluster_names = []
    for c in range(n_clusters):
        centre = km.cluster_centers_[c]
        centre = centre / np.linalg.norm(centre)
        members = np.where(km.labels_ == c)[0]
        best = members[np.argmax(vecs[members] @ centre)]
        cluster_names.append(words[best])

    # semantic color: the color term whose vector is closest to each word
    color_names = list(COLOR_TERMS)
    color_sims = vecs @ COLOR_VECS.T
    query_color = color_names[int(np.argmax(COLOR_VECS @ query_vec))]

    results = []
    for i, w in enumerate(words):
        ci = int(np.argmax(color_sims[i]))
        results.append(
            {
                "word": w,
                "similarity": float(sims[i]),
                "x": float(xy[i, 0]),
                "y": float(xy[i, 1]),
                "cluster": int(km.labels_[i]),
                "color_term": color_names[ci],
                "color_term_sim": float(color_sims[i, ci]),
            }
        )

    return jsonify(
        word=word,
        model=MODEL_NAME,
        layout=layout,
        query_color_term=query_color,
        clusters=[
            {"name": n, "color": CLUSTER_PALETTE[i]} for i, n in enumerate(cluster_names)
        ],
        color_terms=COLOR_TERMS,
        results=results,
    )


if __name__ == "__main__":
    # port 5000 is taken by AirPlay Receiver on macOS, so use 5050
    print("Open http://127.0.0.1:5050")
    app.run(port=5050, debug=False)
