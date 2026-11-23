"""The tile embedding map as a height field: Part 2's digital elevation model.

The Embedding Map tab draws this on screen, but it draws JPEG thumbnails and
exports a PDF of them. A displacement map needs the opposite: no colour, no
page furniture, and more tonal steps than 8 bits gives, or the relief comes out
of Rhino or Blender in visible terraces. So the map is rebuilt here from the
same pieces the app uses -- its autoencoder, its `prepare`, its `project`, and
the snapping its `quantizeCoords` does in the browser -- and written as one
16-bit grayscale PNG.

The tiles are the depth maps themselves, not the autoencoder's reconstructions:
the model only decides which tile sits next to which.

Three files come out of one run, all on the same grid: the height map, the same
mosaic in the specimens' own colour (`_albedo`), and the same mosaic with one
colour per cluster (`_clusters_k<k>`, one per cluster count). The relief is identical in all three -- only
what is painted on it changes, so the viewer can swap them on one mesh.

    uv run python scripts/06_b_tile_mosaic.py
    uv run python scripts/06_b_tile_mosaic.py subdivisions=28 gap=0.08 base=0.35

  subdivisions  cells per side of the map (granularity of the design)
  gap           map colour left between tiles, as a fraction of a cell
                (compactness: 0 = a mosaic, the tiles touch)
  tile          pixels per cell in the output
  base          0 = every tile keeps its own full depth range; higher mixes in
                a per-cluster base elevation, so the wall also reads at a
                distance instead of as equally deep squares
  k             cluster counts to paint, comma separated (kmeans on the
                latent vectors, as the app does); the first also drives `base`
  min_std       drop depth maps flatter than this; they become dead squares
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageStat

HERE = Path(__file__).resolve().parents[1]
APP = HERE / "autoencoder_app_v2" / "06_autoencoders"
sys.path.insert(0, str(APP))

import app_server                                  # noqa: E402  (needs APP on the path)
from autoencoder_model import Autoencoder          # noqa: E402

ROOT = HERE.parent / "05_classifiers"
DEPTHS = ROOT / "datasets" / "topography" / "cell_depth"
MODEL = ROOT / "models" / "topo_cell_depth_cnn"
OUT = HERE / "data" / "topography" / "cell"

SETTINGS = {
    "subdivisions": 24,
    "gap": 0.0,
    "tile": 128,
    "base": 0.0,
    "k": "3,8",
    "min_std": 30.0,
    "perplexity": 30.0,
}


def square(img: Image.Image) -> Image.Image:
    """Center crop to a square, the same crop the autoencoder's preprocess does."""
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    return img.crop((left, top, left + side, top + side))


def load(min_std: float) -> tuple[list[Path], list[Image.Image]]:
    paths, kept = sorted(DEPTHS.glob("*.png")), []
    for path in paths:
        img = square(Image.open(path).convert("L"))
        if ImageStat.Stat(img.resize((96, 96))).stddev[0] >= min_std:
            kept.append((path, img))
    print(f"{len(kept)} of {len(paths)} depth maps kept (flatter than "
          f"std {min_std:g} dropped)")
    return [p for p, _ in kept], [i for _, i in kept]


def embed(images: list[Image.Image]) -> np.ndarray:
    """Latent vector per tile. On the cpu: 500 images at 64px is a few seconds,
    and it keeps the gpu free for whatever else is running."""
    model = Autoencoder.from_folder(MODEL)
    print(model)
    vectors = []
    with torch.no_grad():
        for start in range(0, len(images), 64):
            batch = [im.convert("RGB") for im in images[start:start + 64]]
            vectors.append(model.embed(batch).numpy())
    return np.concatenate(vectors)


def snap(xy: np.ndarray, n: int) -> np.ndarray:
    """embedding_map.js quantizeCoords: every point to the center of its cell,
    the closest one keeps it, the rest take the nearest free cell."""
    cells = np.clip(np.floor(xy * n).astype(int), 0, n - 1)
    center = (cells + 0.5) / n
    order = np.argsort(np.linalg.norm(xy - center, axis=1))

    taken, out = set(), np.zeros((len(xy), 2), dtype=int)
    for i in order:
        wx, wy = cells[i]
        best, radius = None, 0
        while best is None:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    cx, cy = wx + dx, wy + dy
                    if not (0 <= cx < n and 0 <= cy < n) or (cx, cy) in taken:
                        continue
                    d = np.hypot(xy[i, 0] - (cx + 0.5) / n, xy[i, 1] - (cy + 0.5) / n)
                    if best is None or d < best[0]:
                        best = (d, cx, cy)
            radius += 1
        taken.add((best[1], best[2]))
        out[i] = (best[1], best[2])
    return out


def elevations(images: list[Image.Image], labels: np.ndarray, base: float,
               side: int) -> list[np.ndarray]:
    """Every tile as float 0..1, optionally lifted onto a per-cluster base.

    Depth Anything normalizes each image on its own, so without this every tile
    spans the full height of the wall and the relief has no large scale."""
    tiles = [np.asarray(im.resize((side, side), Image.LANCZOS), dtype=np.float32) / 255
             for im in images]
    if base <= 0:
        return tiles

    # a cluster's base height is how rough its tiles are, spread over 0..1
    rough = np.array([t.std() for t in tiles])
    per_cluster = {c: rough[labels == c].mean() for c in np.unique(labels)}
    lo, hi = min(per_cluster.values()), max(per_cluster.values())
    scaled = {c: (v - lo) / max(hi - lo, 1e-8) for c, v in per_cluster.items()}
    return [(1 - base) * t + base * scaled[c] for t, c in zip(tiles, labels)]


def mosaic(tiles: list[np.ndarray], cells: np.ndarray, n: int, tile_px: int,
           gap: float) -> np.ndarray:
    """-> float 0..1, [n * tile_px, n * tile_px], 0 (the map colour) in the gaps."""
    inner = max(1, round(tile_px * (1 - gap)))
    pad = (tile_px - inner) // 2
    canvas = np.zeros((n * tile_px, n * tile_px), dtype=np.float32)
    for tile, (cx, cy) in zip(tiles, cells):
        if inner != tile.shape[0]:
            tile = np.asarray(Image.fromarray((tile * 255).astype(np.uint8))
                              .resize((inner, inner), Image.LANCZOS), dtype=np.float32) / 255
        y, x = cy * tile_px + pad, cx * tile_px + pad
        canvas[y:y + inner, x:x + inner] = tile
    return canvas


# --- colour -----------------------------------------------------------------
#
# The height field answers "how deep", not "what of". Two questions from the
# brief need the other channel:
#
#   does a cluster stand for a material?  ->  flood each tile with its cluster's
#       colour and see whether the colours land in patches on the map or scatter
#   what if the surface kept the specimen?  ->  the depth map still drives the
#       relief, but the original stained crop is laid back over it as colour
#
# Both are written as ordinary 8-bit RGB next to the height map, on exactly the
# same grid, so the viewer can drop either one onto the same mesh.

# ten hues that stay apart on screen and in print -- a cluster map is read by
# which tiles share a colour, so the palette has to separate, not blend
PALETTE = [
    (0.91, 0.30, 0.24), (0.20, 0.60, 0.86), (0.18, 0.72, 0.49),
    (0.95, 0.73, 0.22), (0.61, 0.35, 0.71), (0.94, 0.50, 0.20),
    (0.35, 0.78, 0.78), (0.91, 0.46, 0.70), (0.55, 0.62, 0.25),
    (0.47, 0.51, 0.58),
]


def colour_folder(depths: Path) -> Path:
    """The folder of originals the depth maps were estimated from."""
    return depths.parent / depths.name.replace("_depth", "")


def colour_tiles(paths: list[Path], side: int,
                 depths: Path | None = None) -> list[np.ndarray | None]:
    """Each depth map's own source crop as float 0..1 RGB, None where missing.

    Same centre crop as `square`, so a tile's colour and its depth line up
    pixel for pixel."""
    folder = colour_folder(depths or DEPTHS)
    tiles, missing = [], 0
    for path in paths:
        hit = next((folder / (path.stem + ext)
                    for ext in (".jpg", ".jpeg", ".png", ".webp")
                    if (folder / (path.stem + ext)).is_file()), None)
        if hit is None:
            tiles.append(None)
            missing += 1
            continue
        img = square(Image.open(hit).convert("RGB")).resize((side, side), Image.LANCZOS)
        tiles.append(np.asarray(img, dtype=np.float32) / 255)
    if missing:
        print(f"  ! {missing} of {len(paths)} tiles have no original in {folder.name}")
    return tiles


def grey_tiles(images: list[Image.Image], side: int) -> list[np.ndarray]:
    return [np.asarray(im.resize((side, side), Image.LANCZOS), dtype=np.float32) / 255
            for im in images]


def as_rgb(colour: list[np.ndarray | None], greys: list[np.ndarray]) -> list[np.ndarray]:
    """Colour where there is one, the depth map itself where there is not."""
    return [c if c is not None else np.dstack([g] * 3) for c, g in zip(colour, greys)]


def cluster_tiles(labels: np.ndarray, greys: list[np.ndarray],
                  contrast: float = 0.45) -> list[np.ndarray]:
    """Every tile flooded with its cluster's colour.

    The tile's own depth stays in as brightness, or the wall turns into flat
    confetti and the relief underneath it stops reading."""
    out = []
    for grey, label in zip(greys, labels):
        rgb = np.array(PALETTE[int(label) % len(PALETTE)], dtype=np.float32)
        shade = (1 - contrast) + contrast * grey
        out.append(rgb[None, None, :] * shade[:, :, None])
    return out


def mosaic_rgb(tiles: list[np.ndarray], cells: np.ndarray, n: int, tile_px: int,
               gap: float) -> np.ndarray:
    """`mosaic` for three channels -> float 0..1, [n * tile_px, n * tile_px, 3]."""
    inner = max(1, round(tile_px * (1 - gap)))
    pad = (tile_px - inner) // 2
    canvas = np.zeros((n * tile_px, n * tile_px, 3), dtype=np.float32)
    for tile, (cx, cy) in zip(tiles, cells):
        if inner != tile.shape[0]:
            tile = np.asarray(Image.fromarray((tile * 255).astype(np.uint8))
                              .resize((inner, inner), Image.LANCZOS), dtype=np.float32) / 255
        y, x = cy * tile_px + pad, cx * tile_px + pad
        canvas[y:y + inner, x:x + inner] = tile
    return canvas


def save_rgb(canvas: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(canvas, 0, 1) * 255).round().astype(np.uint8)).save(path)


def report_palette(labels: np.ndarray) -> None:
    """Which colour is which cluster, for the report's legend."""
    for c in np.unique(labels):
        r, g, b = PALETTE[int(c) % len(PALETTE)]
        print(f"  cluster {int(c)}  #{round(r*255):02x}{round(g*255):02x}{round(b*255):02x}"
              f"  {(labels == c).sum():3d} tiles")


def hillshade(canvas: np.ndarray, exaggeration: float = 60.0) -> Image.Image:
    """The height field lit from the top left, to read the relief on screen.
    Not a render: just what the displacement would do to a flat surface."""
    z = canvas * exaggeration
    gy, gx = np.gradient(z)
    normals = np.dstack([-gx, -gy, np.ones_like(z)])
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    light = np.array([-0.5, -0.6, 0.62])
    light /= np.linalg.norm(light)
    shade = np.clip(normals @ light, 0, 1) ** 0.9
    return Image.fromarray((shade * 255).round().astype(np.uint8))


def main() -> None:
    s = dict(SETTINGS)
    for arg in sys.argv[1:]:
        key, _, value = arg.partition("=")
        if key not in s:
            raise SystemExit(f"unknown setting {key} (have {', '.join(s)})")
        s[key] = type(SETTINGS[key])(value)

    paths, images = load(s["min_std"])
    vectors = embed(images)

    # the app's own steps: standardize (its default for a cnn), t-SNE, kmeans
    x = app_server.prepare(vectors, "standardize")
    xy = app_server.project(x, "tsne", s["perplexity"])
    ks = [int(v) for v in str(s["k"]).split(",")]
    # the first k is the one `base` lifts the tiles by; the rest are only painted
    labels = app_server.cluster(x, "kmeans", ks[0], 0.5, 5)
    print(f"t-SNE + kmeans: {len(np.unique(labels))} clusters, "
          f"sizes {sorted((labels == c).sum() for c in np.unique(labels))[::-1]}")

    n = int(s["subdivisions"])
    if len(paths) > n * n:
        raise SystemExit(f"{len(paths)} tiles do not fit in {n}x{n} cells")
    cells = snap(xy, n)
    tile_px = int(s["tile"])
    tiles = elevations(images, labels, s["base"], tile_px)
    canvas = mosaic(tiles, cells, n, tile_px, s["gap"])

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"cell_{n}x{n}_gap{s['gap']:g}_base{s['base']:g}"
    # Pillow builds "I;16" from raw bytes, not from fromarray (which has no
    # 16-bit grayscale mode); numpy and "I;16" are both little endian here
    height = (canvas * 65535).round().astype("<u2")
    Image.frombytes("I;16", (height.shape[1], height.shape[0]),
                    height.tobytes()).save(OUT / f"{stem}_height16.png")
    Image.fromarray((canvas * 255).round().astype(np.uint8)).save(OUT / f"{stem}_preview.png")
    hillshade(canvas).save(OUT / f"{stem}_shaded.png")

    greys = grey_tiles(images, tile_px)
    save_rgb(mosaic_rgb(as_rgb(colour_tiles(paths, tile_px), greys),
                        cells, n, tile_px, s["gap"]), OUT / f"{stem}_albedo.png")
    for k in ks:
        label = labels if k == ks[0] else app_server.cluster(x, "kmeans", k, 0.5, 5)
        save_rgb(mosaic_rgb(cluster_tiles(label, greys), cells, n, tile_px, s["gap"]),
                 OUT / f"{stem}_clusters_k{k}.png")
        print(f"k = {k}")
        report_palette(label)

    with open(OUT / f"{stem}_tiles.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell_x", "cell_y", "cluster", "source"])
        for (cx, cy), c, path in zip(cells, labels, paths):
            writer.writerow([cx, cy, int(c), path.name])

    filled = len(paths) / (n * n)
    print(f"\n{n}x{n} cells, {len(paths)} filled ({filled:.0%}), "
          f"{canvas.shape[1]}x{canvas.shape[0]} px")
    print(f"-> {OUT / (stem + '_height16.png')}")


if __name__ == "__main__":
    main()
