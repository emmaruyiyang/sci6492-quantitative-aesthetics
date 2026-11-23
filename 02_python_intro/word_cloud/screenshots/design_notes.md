# Assignment 1 — Semantic Constellation (word cloud)

Flask back-end + Gensim (`glove-wiki-gigaword-50`, 400k words × 50 dims) + scikit-learn, SVG front-end.

## Design points

- **Distance = meaning.** Distance from the centre is `1 − cosine similarity`, with rings showing real values; direction groups neighbours that resemble each other.
- **Minimal white-on-black interface.** Size and brightness show similarity, typography marks clusters, and colour appears only when derived from the embedding.
- **A walkable map.** Click any word to re-centre on it; the breadcrumb records the path.

## Screenshots

1. `01_radial_similarity.png` — *architecture*, radial layout, similarity encoding. The rings let you read exact similarity; the neighbourhood splits into historical styles (left), art/culture (upper right) and design/concepts (bottom).
2. `02_radial_clusters.png` — same query, cluster encoding. The four k-means groups (neoclassical, art, designs, concepts) were computed in 50-d, yet they line up with the angular sectors of the 2D map, a sign the projection keeps the main structure.
3. `03_color_words_ocean.png` — *ocean*, colour-word encoding. The model says ocean is **blue**, but *sea*, *waters*, *coast* and *gulf* come out **red**, probably because of "Red Sea" and similar place names in the news corpus. The embedding knows usage, not perception.
4. `04_pca_clusters_ocean.png` — same query in plain PCA. With no radial constraint, distances to the centre are only approximate: *sea* sits closer to *ocean* than *oceans* does, and the rings disappear. Useful as a baseline to show what the radial layout adds.
5. `05_trail_garden.png` — after walking architecture → ocean → island → garden. The garden neighbourhood splits into cultural space (museum, exhibit, gallery), planted landscape (tree, grass, flowers), domestic rooms (DINING, KITCHEN, PLAYGROUND) and built material (brick, stone, cottage).

## Reflection (short)

- Distance to the centre is exact in rank, but min–max stretched, so differences between 0.66 and 0.90 look larger than they are.
- The 50-d → 2D angle is an approximation; two words that look adjacent on the map are only roughly similar to each other.
- GloVe captures association (garden–kitchen, ocean–storm) as much as similarity (garden–gardens), which the SimLex-999 reading in the lecture distinguishes.
