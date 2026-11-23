# Semantic Constellation — AI-assisted assignment 1 (word cloud)

Flask back-end + Gensim (`glove-wiki-gigaword-50`) + scikit-learn, vanilla SVG front-end.

## Run

From the `sci6492_code` folder (the `.venv` there is Python 3.12 with flask, gensim, scikit-learn):

```bash
.venv/bin/python 02_python_intro/word_cloud/app.py
```

Open http://127.0.0.1:5050 (port 5000 is taken by AirPlay Receiver on macOS).
Use another model with e.g. `MODEL=glove-wiki-gigaword-100 .venv/bin/python ...`.

## How the map is built

- `model.most_similar(word, topn=N)` gives the N nearest neighbours.
- **radial layout** (default): the query sits at the centre.
  - radius = semantic distance `1 - cosine similarity`, rescaled, so the dashed rings are iso-similarity lines
  - angle = 2D PCA of the neighbours after removing their component along the query vector,
    so neighbours that resemble each other face the same direction
- **plain PCA**: PCA of query + neighbours, shifted so the query is at the origin.
- Labels are pushed apart by a small collision-relaxation loop with a spring back to their semantic position.

## Visual encodings

White-on-black interface in an Apple-like dark style (system font, hairlines, 4px corners).

- size and gray level: similarity to the centre word
- encoding, three modes
  - *Similarity*: gray ramp from dark gray (less similar) to white (more similar)
  - *Clusters*: k-means (k=4) on the 50-d word vectors; each cluster gets a typographic style
    (bold / italic / light / caps) and is named by the word nearest its centroid
  - *Color words*: a small dot in the color whose word vector is closest to the word (red, blue, green …)
- hairline rays from the centre, brighter for closer words
- click any word to re-centre on it; the breadcrumb trail records the path through the space
