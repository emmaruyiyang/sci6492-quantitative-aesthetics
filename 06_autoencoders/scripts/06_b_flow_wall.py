"""The galaxy wall as a braided stream: a form study for Part 2.

06_b_tile_mosaic.py lets the projection decide the outline, which comes out as
a t-SNE blob. Here the outline is the design: a braided channel winding down
the wall, the tiles laid into it, and the channel's own cross-section lifting
the core so the wall reads as one form from across the room and as 450 separate
specimens up close.

Two scales, kept separate on purpose:

  large   the stream profile -- where the wall is high and where it falls away
  fine    the tiles themselves -- the grain on that surface

PLACEMENT IS NOT SEMANTIC YET. There is no autoencoder for this dataset, so
the tiles are ordered by how bright and dense they are: the luminous ones go to
the core of the stream, the sparse ones to the fringe. That is a design rule,
not a found structure. `assign_by_density` is the only thing that has to change
once a galaxy autoencoder exists -- the geometry below does not.

    uv run python scripts/06_b_flow_wall.py
    uv run python scripts/06_b_flow_wall.py subdivisions=48 base=0.55 strands=6
"""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parents[1]

# the mosaic script owns the square crop and the hillshade; its module name
# starts with a digit, so it is loaded by path rather than imported
spec = importlib.util.spec_from_file_location("tile_mosaic", HERE / "scripts" / "06_b_tile_mosaic.py")
tiles_mod = importlib.util.module_from_spec(spec)
sys.modules["tile_mosaic"] = tiles_mod
spec.loader.exec_module(tiles_mod)

ROOT = HERE.parent / "05_classifiers"
DATASETS = ROOT / "datasets" / "topography"
MODELS = ROOT / "models"

SETTINGS = {
    "source": "cell_depth",         # folder under datasets/topography
    "model": "topo_cell_depth_cnn",  # "" falls back to ordering by density
    "subdivisions": 36,     # cells per side of the wall
    "tile": 86,             # pixels per cell
    "base": 0.12,           # how much of the height is the channel, not the tile
    "width": 0.085,         # half width of one strand, as a fraction of the wall
    "edge": 0.35,           # outer part of a strand that slopes; the rest is flat
    "amp": 0.2,             # how far the channel swings left and right
    "freq": 0.75,           # swings over the height of the wall
    "strands": 3,           # braids
    "braid": 0.09,          # how far the strands wander from the channel
    "weave": 0.9,           # crossings over the height; low = long, smooth curves
    "min_std": 18.0,        # drop tiles with no structure to show
    "perplexity": 30.0,
}


def centre(v: np.ndarray, s: dict) -> np.ndarray:
    """Where the channel runs at height v (0 = top, 1 = bottom)."""
    return 0.5 + s["amp"] * np.sin(2 * np.pi * s["freq"] * v + 0.45)


def nearness(u: np.ndarray, v: np.ndarray, s: dict) -> np.ndarray:
    """How deep inside the channel (u, v) is: 1 at a strand's line, falling off.

    Separate from flow() because that one is a plateau -- flat, and therefore
    equal, over most of a strand. Picking the strongest cells from a field full
    of ties leaves the order to the array layout, which fills whole rows. This
    one is strictly decreasing with distance, so the silhouette is the channel."""
    mid = centre(v, s)
    best = np.full(u.shape, np.inf)
    for k in range(int(s["strands"])):
        phase = 2 * np.pi * k / s["strands"]
        wander = s["braid"] * np.sin(2 * np.pi * s["weave"] * v + phase) * (0.35 + k / s["strands"])
        best = np.minimum(best, np.abs(u - (mid + wander)))
    return 1 / (1 + best / s["width"])


def flow(u: np.ndarray, v: np.ndarray, s: dict) -> np.ndarray:
    """How much channel there is at (u, v), 0..1.

    A strand is a *plateau*, not a ridge: flat across its middle, sloping only
    over the outer `edge` of its width. A gaussian section would make the wall
    a mountain, and the tiles would then differ more by where they sit on that
    slope than by what they hold -- the relief has to come from the tiles, so
    the channel only decides where the wall is and where the void is."""
    mid = centre(v, s)
    strength = np.zeros_like(u)
    for k in range(int(s["strands"])):
        phase = 2 * np.pi * k / s["strands"]
        wander = s["braid"] * np.sin(2 * np.pi * s["weave"] * v + phase) * (0.35 + k / s["strands"])
        d = np.abs(u - (mid + wander)) / s["width"]
        t = np.clip((1 - d) / max(s["edge"], 1e-6), 0, 1)
        strength = np.maximum(strength, t * t * (3 - 2 * t))      # smoothstep
    return strength


def load_tiles(folder: Path, min_std: float):
    """Square, grayscale tiles, with the featureless ones dropped."""
    paths, kept = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")), []
    for path in paths:
        image = tiles_mod.square(Image.open(path).convert("L"))
        if ImageStat.Stat(image.resize((96, 96))).stddev[0] >= min_std:
            kept.append((path, image))
    print(f"{len(kept)} of {len(paths)} tiles kept (flatter than std {min_std:g} dropped)")
    return [p for p, _ in kept], [i for _, i in kept]


def channel_cells(n: int, count: int, s: dict):
    """The `count` cells most inside the stream -> their grid and stream coords.

    Taking the strongest cells rather than thresholding means the silhouette is
    exactly as big as the number of tiles: no empty cells, no tiles left over."""
    j, i = np.mgrid[0:n, 0:n]
    u, v = (i + 0.5) / n, (j + 0.5) / n
    inside = nearness(u, v, s)

    flat = inside.ravel()
    chosen = np.argpartition(-flat, count - 1)[:count]
    cy, cx = np.unravel_index(chosen, inside.shape)

    core = flat[chosen]                                   # 1 at the middle of a strand
    across = u[cy, cx] - centre(v[cy, cx], s)
    span = np.abs(across).max() or 1.0
    return cx, cy, core, v[cy, cx], across / span


def assign_by_density(images, core):
    """Fallback when there is no model: densest tile to the deepest cell.

    A design rule, not a found structure -- say so if a wall is built on it."""
    density = np.array([np.asarray(im.resize((64, 64)), dtype=np.float32).mean() for im in images])
    cost = np.abs(rank(density)[:, None] - rank(core)[None, :])
    return solve(cost, len(images))


def assign_by_embedding(images, along, across, s):
    """The self-organizing part: the autoencoder's map, laid into the channel.

    The tiles are embedded and projected exactly as the Embedding Map does it,
    giving each tile a place on a square map. Each cell of the channel also has
    two coordinates -- how far along it sits and how far across -- so the square
    map is matched onto the channel by the assignment that moves every tile the
    least in total.

    The channel is long and narrow, so `across` holds only a handful of cells
    against forty-odd `along`: most of the 2D structure has to collapse onto the
    length of the channel. What survives is a gradient you read by walking
    along the wall, not a map you read face on."""
    tiles_mod.MODEL = MODELS / s["model"]
    vectors = tiles_mod.embed(images)
    x = tiles_mod.app_server.prepare(vectors, "standardize")
    xy = tiles_mod.app_server.project(x, "tsne", s["perplexity"])

    cells = np.stack([rank(along), rank(across)], axis=1)
    tiles = np.stack([rank(xy[:, 0]), rank(xy[:, 1])], axis=1)
    cost = np.linalg.norm(tiles[:, None, :] - cells[None, :, :], axis=2)
    return solve(cost, len(images))


def solve(cost: np.ndarray, n: int) -> np.ndarray:
    rows, cols = linear_sum_assignment(cost)
    order = np.empty(n, dtype=int)
    order[rows] = cols
    return order


def rank(values: np.ndarray) -> np.ndarray:
    """-> 0..1 by position in the sorted order, ties broken arbitrarily."""
    out = np.empty(len(values))
    out[np.argsort(values)] = np.linspace(0, 1, len(values))
    return out


def main() -> None:
    s = dict(SETTINGS)
    for arg in sys.argv[1:]:
        key, _, value = arg.partition("=")
        if key not in s:
            raise SystemExit(f"unknown setting {key} (have {', '.join(s)})")
        s[key] = type(SETTINGS[key])(value)

    source = DATASETS / s["source"]
    out = HERE / "data" / "topography" / s["source"].split("_")[0]
    paths, images = load_tiles(source, s["min_std"])
    n, tile_px = int(s["subdivisions"]), int(s["tile"])
    if len(images) > n * n:
        raise SystemExit(f"{len(images)} tiles do not fit in {n}x{n} cells")

    cx, cy, core, along, across = channel_cells(n, len(images), s)
    if s["model"]:
        where = assign_by_embedding(images, along, across, s)
        placed = f"embedding ({s['model']})"
    else:
        where = assign_by_density(images, core)
        placed = "density (no model)"
    print(f"placed by {placed}")

    # --- the fine scale: the tiles themselves -------------------------------
    size = n * tile_px
    detail = np.zeros((size, size), dtype=np.float32)
    occupied = np.zeros((size, size), dtype=np.float32)

    for t, cell in enumerate(where):
        patch = np.asarray(images[t].resize((tile_px, tile_px), Image.LANCZOS),
                           dtype=np.float32) / 255
        y, x = cy[cell] * tile_px, cx[cell] * tile_px
        detail[y:y + tile_px, x:x + tile_px] = patch
        occupied[y:y + tile_px, x:x + tile_px] = 1

    # --- the large scale: the stream's own section ---------------------------
    py, px = np.mgrid[0:size, 0:size]
    stream = flow((px + 0.5) / size, (py + 0.5) / size, s)

    height = occupied * ((1 - s["base"]) * detail + s["base"] * stream)

    out.mkdir(parents=True, exist_ok=True)
    stem = f"{s['source']}_flow_{n}x{n}_base{s['base']:g}_strands{int(s['strands'])}"
    raw = (height * 65535).round().astype("<u2")
    Image.frombytes("I;16", (size, size), raw.tobytes()).save(out / f"{stem}_height16.png")
    Image.fromarray((height * 255).round().astype(np.uint8)).save(out / f"{stem}_preview.png")
    tiles_mod.hillshade(height).save(out / f"{stem}_shaded.png")

    with open(out / f"{stem}_tiles.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell_x", "cell_y", "core", "along", "across", "source"])
        for t, cell in enumerate(where):
            writer.writerow([cx[cell], cy[cell], round(float(core[cell]), 4),
                             round(float(along[cell]), 4), round(float(across[cell]), 4),
                             paths[t].name])

    print(f"\n{len(images)} tiles in a {int(s['strands'])}-strand channel, "
          f"{n}x{n} cells, {size}x{size} px")
    print(f"-> {out / (stem + '_height16.png')}")


if __name__ == "__main__":
    main()
