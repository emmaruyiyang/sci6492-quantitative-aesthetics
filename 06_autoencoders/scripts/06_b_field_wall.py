"""The wall: a generated field carrying the large scale, tiles carrying the grain.

A mosaic of independent depth maps cannot produce a landform on its own. Every
tile is normalized to its own full range, so the wall comes out as a few hundred
equally tall bumps with nothing joining them. The continuity has to come from
somewhere else, and here it comes from a field built by 06_b_tube_field.py.

The two scales are kept in separate channels:

  field   where the wall is high and where it falls away
  tiles   the grain on that surface, levelled so each tile adds only its
          texture and not the arbitrary average height its normalization gave it

Levelling is what makes the field legible. Without it the per-tile means --
noise from Depth Anything, not anything about the specimen -- sit in the same
channel as the landform and drown it.

Tiles are still placed by the autoencoder: the embedding map is matched onto
the cells of the silhouette by the assignment that moves every tile the least.

As in 06_b_tile_mosaic.py, the wall is also written in colour: `_albedo` keeps
the stained crops the depth maps came from, `_clusters_k<k>` paints each tile with
its kmeans colour, one file per cluster count in `k`.

    uv run python scripts/06_b_field_wall.py
    uv run python scripts/06_b_field_wall.py grain=0.3 subdivisions=40
"""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("tile_mosaic", HERE / "scripts" / "06_b_tile_mosaic.py")
tiles_mod = importlib.util.module_from_spec(spec)
sys.modules["tile_mosaic"] = tiles_mod
spec.loader.exec_module(tiles_mod)

ROOT = HERE.parent / "05_classifiers"
DATASETS = ROOT / "datasets" / "topography"
MODELS = ROOT / "models"
FIELDS = HERE / "data" / "topography" / "fields"

SETTINGS = {
    "field": "tube_amp0.2_r0.42-0.09_pinch2_height16.png",
    "source": "cell_depth",
    "model": "topo_cell_depth_cnn",
    "subdivisions": 36,
    "tile": 86,
    "grain": 0.22,      # how much of the height is the tiles, not the field
    "k": "3,8",         # cluster counts to paint; one colour map each
    "min_std": 18.0,
    "perplexity": 30.0,
}


def rank(values: np.ndarray) -> np.ndarray:
    out = np.empty(len(values))
    out[np.argsort(values)] = np.linspace(0, 1, len(values))
    return out


def load_tiles(folder: Path, min_std: float):
    paths, kept = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")), []
    for path in paths:
        image = tiles_mod.square(Image.open(path).convert("L"))
        if ImageStat.Stat(image.resize((96, 96))).stddev[0] >= min_std:
            kept.append((path, image))
    print(f"{len(kept)} of {len(paths)} tiles kept")
    return [p for p, _ in kept], [i for _, i in kept]


def cells_of(field: np.ndarray, n: int, count: int):
    """The `count` cells where the field stands highest -> the wall's footprint."""
    side = field.shape[0] // n
    coarse = field[:side * n, :side * n].reshape(n, side, n, side).mean(axis=(1, 3))
    flat = coarse.ravel()
    chosen = np.argpartition(-flat, count - 1)[:count]
    cy, cx = np.unravel_index(chosen, coarse.shape)
    return cx, cy, flat[chosen]


def place(images, cx, cy, s):
    """Match the embedding map onto the footprint's cells.

    Also returns the kmeans labels, so the wall can be painted by cluster and
    the grouping can be checked against what the specimens actually look like."""
    tiles_mod.MODEL = MODELS / s["model"]
    vectors = tiles_mod.embed(images)
    x = tiles_mod.app_server.prepare(vectors, "standardize")
    xy = tiles_mod.app_server.project(x, "tsne", s["perplexity"])
    # every k off the same latent vectors and the same map: the walls differ
    # only in how finely they are grouped, not in which tile sits where
    labels = {k: tiles_mod.app_server.cluster(x, "kmeans", k, 0.5, 5)
              for k in (int(v) for v in str(s["k"]).split(","))}

    want = np.stack([rank(xy[:, 0]), rank(xy[:, 1])], axis=1)
    have = np.stack([rank(cx.astype(float)), rank(cy.astype(float))], axis=1)
    cost = np.linalg.norm(want[:, None, :] - have[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    order = np.empty(len(images), dtype=int)
    order[rows] = cols
    return order, labels


def main() -> None:
    s = dict(SETTINGS)
    for arg in sys.argv[1:]:
        key, _, value = arg.partition("=")
        if key not in s:
            raise SystemExit(f"unknown setting {key} (have {', '.join(s)})")
        s[key] = type(SETTINGS[key])(value)

    field = np.asarray(Image.open(FIELDS / s["field"])).astype(np.float32)
    field /= field.max() or 1.0
    paths, images = load_tiles(DATASETS / s["source"], s["min_std"])

    n, tile_px = int(s["subdivisions"]), int(s["tile"])
    if len(images) > n * n:
        raise SystemExit(f"{len(images)} tiles do not fit in {n}x{n} cells")
    cx, cy, strength = cells_of(field, n, len(images))
    where, labels = place(images, cx, cy, s)

    size = n * tile_px
    big = np.asarray(Image.fromarray((field * 255).astype(np.uint8))
                     .resize((size, size), Image.BICUBIC), dtype=np.float32) / 255

    grain = np.zeros((size, size), dtype=np.float32)
    occupied = np.zeros((size, size), dtype=np.float32)
    for t, cell in enumerate(where):
        patch = np.asarray(images[t].resize((tile_px, tile_px), Image.LANCZOS),
                           dtype=np.float32) / 255
        patch = patch - patch.mean()          # levelled: texture only, no base height
        y, x = cy[cell] * tile_px, cx[cell] * tile_px
        grain[y:y + tile_px, x:x + tile_px] = patch
        occupied[y:y + tile_px, x:x + tile_px] = 1

    height = occupied * ((1 - s["grain"]) * big + s["grain"] * (grain + 0.5))
    height = np.clip(height, 0, None)
    height /= height.max() or 1.0

    out = HERE / "data" / "topography" / "walls"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{s['source']}_on_{Path(s['field']).stem.split('_height')[0]}_{n}x{n}_g{s['grain']:g}"
    raw = (height * 65535).round().astype("<u2")
    Image.frombytes("I;16", (size, size), raw.tobytes()).save(out / f"{stem}_height16.png")
    Image.fromarray((height * 255).round().astype(np.uint8)).save(out / f"{stem}_preview.png")
    tiles_mod.hillshade(height).save(out / f"{stem}_shaded.png")

    # the same wall in colour: the originals, and the clusters. The relief is
    # unchanged -- only what is painted on it differs.
    greys = tiles_mod.grey_tiles(images, tile_px)
    colour = tiles_mod.as_rgb(tiles_mod.colour_tiles(paths, tile_px,
                                                     DATASETS / s["source"]), greys)
    albedo = np.zeros((size, size, 3), dtype=np.float32)
    for t, cell in enumerate(where):
        y, x = cy[cell] * tile_px, cx[cell] * tile_px
        albedo[y:y + tile_px, x:x + tile_px] = colour[t]
    tiles_mod.save_rgb(albedo, out / f"{stem}_albedo.png")

    for k, label in labels.items():
        painted = tiles_mod.cluster_tiles(label, greys)
        canvas = np.zeros((size, size, 3), dtype=np.float32)
        for t, cell in enumerate(where):
            y, x = cy[cell] * tile_px, cx[cell] * tile_px
            canvas[y:y + tile_px, x:x + tile_px] = painted[t]
        tiles_mod.save_rgb(canvas, out / f"{stem}_clusters_k{k}.png")
        print(f"k = {k}")
        tiles_mod.report_palette(label)

    with open(out / f"{stem}_tiles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        ks = sorted(labels)
        w.writerow(["cell_x", "cell_y", "field"] + [f"cluster_k{k}" for k in ks]
                   + ["source"])
        for t, cell in enumerate(where):
            w.writerow([cx[cell], cy[cell], round(float(strength[cell]), 4)]
                       + [int(labels[k][t]) for k in ks] + [paths[t].name])

    print(f"\n{len(images)} tiles on {s['field']}, {n}x{n} cells, {size}x{size} px")
    print(f"-> {out / (stem + '_height16.png')}")


if __name__ == "__main__":
    main()
