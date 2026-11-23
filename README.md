<div align="center">

# Measuring Taste

<h3>Nine ways to put a number on an image</h3>

Coursework for **SCI 6492 · Quantitative Aesthetics** · Harvard GSD · Fall 2026 · Ruyi Yang

</div>


## 🎯 What this is

- Each assignment takes something usually argued about in words — a city, a style, a surface, a preference — and puts a number on it.
- Each one ends in something you can look at or click: a Flask app, a WebGL scene, a rendered report.
- Code is committed; datasets and weights are not (see [What's not here](#-whats-not-here)).


## 📦 Contents

| folder | what it does |
|---|---|
| `01_intro` · `02_python_intro` | Python warm-ups and a word-cloud Flask app |
| `03_language_embeddings` | **Two Londons** — word2vec trained separately on the Sherlock Holmes canon and Jack London's *People of the Abyss*; the same words drawn twice |
| `05_classifiers` | CNN and transfer-learning classifiers over scraped Pinterest sets, comparing ImageNet / CLIP / DINOv2 backbones on identical splits |
| `06_autoencoders` | autoencoders on image tiles, latent walks, and a WebGL app that reads luminance as height |
| `07_segmentation` | segmentation as censorship — masking bodies in *The Rape of the Sabine Women*, masks carried back onto a 3D rework |
| `08_ranknet_taste` | **RankNet and taste** — my score head (662 pairwise judgments) and a partner's, scoring the same 210 generated gift boxes. Rank correlation: **+0.02** |
| `09_optimization` | an optimization app over design parameters from earlier weeks |
| `final_project` | **scaleviz** — continuous zoom across ~4.5 decades, cell → river → galaxy |
| `assignment_deliverable` | the submitted report PDFs |


## 🔧 Running things

```bash
uv sync
```

- Python 3.12 via uv; the root `pyproject.toml` covers most assignments.
- `07_segmentation`, `08_ranknet_taste` and the autoencoder app carry their own `pyproject.toml` — run them from inside that folder.
- `final_project/scaleviz` is a Vite app:

```bash
cd final_project/scaleviz && npm install && npm run dev
```


## 🚫 What's not here

| excluded | why |
|---|---|
| scraped datasets (Pinterest, gift boxes, topography samples) | a few GB; the fetch scripts are committed |
| trained weights (`*.pt`, `*.kv`, `05_classifiers/models/`) | regenerable from the training scripts |
| `.3dm` files, zipped submissions, `node_modules/`, `.venv/` | large or rebuildable |
