"""Turn a height map into a mesh file, for Rhino, Blender or a print.

The three.js page displaces a plane in the browser and nothing of that survives
the tab closing. This writes the same surface to disk as OBJ: the height map
sampled on a grid, area-averaged down to the mesh resolution exactly as the
viewer does it, so the file matches what the preview shows.

Only the wall is written -- cells where the height field is zero are the void
around it, and their triangles are skipped, so the file carries the silhouette
instead of a full square sheet.

    uv run python scripts/06_b_export_mesh.py
    uv run python scripts/06_b_export_mesh.py resolution=800 exaggeration=0.2

  resolution    vertices per side before the void is dropped
  exaggeration  height as a fraction of the wall's width (the viewer's slider)
  smoothing     box blur passes on the mesh grid, as in the viewer
  skirt         1 = drop a wall from the silhouette down to the base plane
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
WALLS = HERE / "data" / "topography" / "walls"
OUT = HERE / "data" / "topography" / "mesh"

SETTINGS = {
    "height": "cell_depth_on_tube_amp0.2_r0.42-0.09_pinch2_36x36_g0.22_height16.png",
    "resolution": 700,
    "exaggeration": 0.14,
    "smoothing": 1,
    "skirt": 1,
}


def downsample(field: np.ndarray, n: int) -> np.ndarray:
    h, w = field.shape
    out = np.zeros((n, n), dtype=np.float32)
    for j in range(n):
        y0, y1 = j * h // n, max(j * h // n + 1, (j + 1) * h // n)
        for i in range(n):
            x0, x1 = i * w // n, max(i * w // n + 1, (i + 1) * w // n)
            out[j, i] = field[y0:y1, x0:x1].mean()
    return out


def blur(grid: np.ndarray, passes: int) -> np.ndarray:
    for _ in range(passes):
        p = np.pad(grid, 1, mode="edge")
        grid = sum(p[a:a + grid.shape[0], b:b + grid.shape[1]]
                   for a in range(3) for b in range(3)) / 9
    return grid


def main() -> None:
    s = dict(SETTINGS)
    for arg in sys.argv[1:]:
        key, _, value = arg.partition("=")
        if key not in s:
            raise SystemExit(f"unknown setting {key} (have {', '.join(s)})")
        s[key] = type(SETTINGS[key])(value)

    field = np.asarray(Image.open(WALLS / s["height"])).astype(np.float32)
    field /= field.max() or 1.0

    n = int(s["resolution"])
    raw = downsample(field, n)
    grid = blur(raw, int(s["smoothing"]))
    solid = raw > 0.01                       # the wall; everything else is void

    # vertices on a unit square, y up, matching the viewer's orientation
    u = (np.arange(n) + 0.5) / n - 0.5
    gx, gz = np.meshgrid(u, u)
    gy = grid * float(s["exaggeration"])

    index = -np.ones((n, n), dtype=int)
    verts = []
    for j in range(n):
        for i in range(n):
            if solid[j, i]:
                index[j, i] = len(verts)
                verts.append((gx[j, i], gy[j, i], gz[j, i]))

    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = index[j, i], index[j, i + 1], index[j + 1, i + 1], index[j + 1, i]
            if min(a, b, c, d) < 0:          # the quad touches the void
                continue
            faces.append((a + 1, b + 1, c + 1))
            faces.append((a + 1, c + 1, d + 1))

    if int(s["skirt"]):
        # a wall straight down wherever the surface meets the void, so the
        # shape closes into something with a visible edge instead of a sheet
        base = {}

        def under(j, i):
            if (j, i) not in base:
                base[(j, i)] = len(verts)
                verts.append((gx[j, i], 0.0, gz[j, i]))
            return base[(j, i)]

        for j in range(n - 1):
            for i in range(n - 1):
                for (j0, i0), (j1, i1) in (((j, i), (j, i + 1)), ((j, i), (j + 1, i))):
                    if solid[j0, i0] == solid[j1, i1]:
                        continue
                    a, b = (j0, i0) if solid[j0, i0] else (j1, i1), None
                    edge = (j0, i0) if solid[j0, i0] else (j1, i1)
                    other = (j1, i1) if solid[j0, i0] else (j0, i0)
                    top, foot = index[edge], under(edge[0], edge[1])
                    foot2 = under(other[0], other[1])
                    faces.append((top + 1, foot + 1, foot2 + 1))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (Path(s["height"]).stem.replace("_height16", "") + f"_{n}.obj")
    with open(path, "w") as fh:
        fh.write(f"# {s['height']}  {n}x{n}  exaggeration {s['exaggeration']}\n")
        for x, y, z in verts:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            fh.write(f"f {a} {b} {c}\n")

    print(f"{len(verts)} vertices, {len(faces)} triangles, "
          f"{solid.mean():.0%} of the grid is wall")
    print(f"-> {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
