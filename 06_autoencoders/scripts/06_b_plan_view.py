"""The wall seen straight down, rendered on the height map instead of in the browser.

The three.js page is a perspective camera, so its "plan" is never quite one --
the far edge foreshortens and the slab's own thickness hides the cells behind
it. Straight down is the view where the embedding map is actually legible, and
it costs nothing to compute: the relief is already a height field, so the shade
is its gradient and the colour is the mosaic that was written beside it.

    uv run python scripts/06_b_plan_view.py
    uv run python scripts/06_b_plan_view.py size=3000 relief=0.75

  stem     the wall or mosaic to draw, without the _height16.png
  folder   walls | cell | galaxy | fields
  size     pixels per side of the output
  relief   0 = flat colour, 1 = the shading carries everything
  azimuth  light direction in degrees, as in the viewer
  only     render one map instead of all of them
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
TOPO = HERE / "data" / "topography"
OUT = TOPO / "renders"

SETTINGS = {
    "stem": "cell_depth_on_tube_amp0.2_r0.42-0.09_pinch2_36x36_g0.22",
    "folder": "walls",
    "size": 2400,
    "relief": 0.62,
    "azimuth": 135.0,
    "elevation": 38.0,
    "exaggeration": 26.0,
    "only": "",         # one map only ("relief", "albedo", "clusters_k3"...),
                        # so a stronger light can be used for the bare relief
                        # without rewriting the painted ones
}

BACKGROUND = np.array([11, 12, 14], dtype=np.float32) / 255


def resize(img: Image.Image, size: int) -> np.ndarray:
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.float32)


def shade(height: np.ndarray, s: dict) -> np.ndarray:
    """Lambertian on the height field's own normals -- the same light the
    viewer uses, just evaluated per pixel instead of per vertex."""
    gy, gx = np.gradient(height * float(s["exaggeration"]))
    normals = np.dstack([-gx, -gy, np.ones_like(height)])
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    az = np.radians(float(s["azimuth"]))
    el = np.radians(float(s["elevation"]))
    light = np.array([np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), np.sin(el)])
    light /= np.linalg.norm(light)
    return np.clip(normals @ light, 0, 1) ** 0.85


def main() -> None:
    s = dict(SETTINGS)
    for arg in sys.argv[1:]:
        key, _, value = arg.partition("=")
        if key not in s:
            raise SystemExit(f"unknown setting {key} (have {', '.join(s)})")
        s[key] = type(SETTINGS[key])(value)

    folder = TOPO / s["folder"]
    size = int(s["size"])
    height = resize(Image.open(folder / f"{s['stem']}_height16.png"), size)
    height /= height.max() or 1.0
    solid = height > 0.004                      # everything else is the void
    lit = shade(height, s)

    OUT.mkdir(parents=True, exist_ok=True)
    # the bare relief, plus whatever colour maps were written beside it
    painted = sorted(folder.glob(f"{s['stem']}_albedo.png")) \
        + sorted(folder.glob(f"{s['stem']}_clusters_k*.png"))
    maps = {"relief": None}
    maps.update({path.name.removeprefix(s["stem"] + "_").removesuffix(".png"): path
                 for path in painted})
    if s["only"]:
        if s["only"] not in maps:
            raise SystemExit(f"no {s['only']} map (have {', '.join(maps)})")
        maps = {s["only"]: maps[s["only"]]}

    for tag, path in maps.items():
        if path is None:
            colour = np.ones((size, size, 3), dtype=np.float32) * 0.86
        else:
            colour = resize(Image.open(path).convert("RGB"), size) / 255
        # the shade is a multiplier, lifted so the colour is never crushed out
        frame = colour * ((1 - s["relief"]) + s["relief"] * lit)[:, :, None]
        frame = np.where(solid[:, :, None], frame, BACKGROUND)
        name = f"{s['stem']}_plan_{tag}.png"
        Image.fromarray((np.clip(frame, 0, 1) * 255).round().astype(np.uint8)).save(OUT / name)
        print(f"-> {OUT / name}")

    print(f"\n{size}x{size} px, {solid.mean():.0%} of the frame is wall")


if __name__ == "__main__":
    main()
