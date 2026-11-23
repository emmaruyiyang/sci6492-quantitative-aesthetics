"""The large-scale field on its own: a soft tube, for review before any tiles.

The wall needs continuity that a mosaic of independent depth maps cannot
produce -- every tile is its own little hill and nothing makes one agree with
the next. So the continuity is supplied here instead, as a smooth field the
tiles will later ride on.

The form is a bent tube: a curved centreline, a radius that swells at both
ends and pinches in the middle, and a circular section. Because the section is
circular, a pinched part is automatically *lower* as well as narrower, the way
a real hose behaves -- a band of constant height would read as a ribbon, not a
body.

    uv run python scripts/06_b_tube_field.py
    uv run python scripts/06_b_tube_field.py amp=0.24 r_end=0.13 r_mid=0.05

  amp / freq    how far and how often the centreline swings
  r_end/r_mid   radius at the two ends and at the waist, as a fraction of the map
  pinch         1 = waist exactly at the middle; higher = a longer slim section
  margin        how far the rounded caps stay inside the frame
"""

import sys
from pathlib import Path

import importlib.util
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("tile_mosaic", HERE / "scripts" / "06_b_tile_mosaic.py")
tiles_mod = importlib.util.module_from_spec(spec)
sys.modules["tile_mosaic"] = tiles_mod
spec.loader.exec_module(tiles_mod)

OUT = HERE / "data" / "topography" / "fields"

SETTINGS = {
    "size": 2048,       # output pixels per side
    "amp": 0.18,        # how far the centreline swings from the middle
    "freq": 0.8,        # swings over the length of the form
    "phase": 0.45,
    "strands": 7,       # filaments; 1 is a plain tube
    "spread": 0.26,     # how far apart the filaments get at the two ends
    "open": 3.0,        # 1 = they part steadily; higher = together longer, then fast
    "r_end": 0.045,     # filament radius at the ends
    "r_mid": 0.055,     # filament radius at the middle
    "pinch": 2.0,       # how the radius runs between the two
    "wobble": 0.02,     # a little waver so the filaments are not a perfect fan
    "margin": -0.04,    # negative: the form runs off the frame, no closed caps
    "samples": 420,     # points along each filament
}


def centreline(t: np.ndarray, s: dict) -> tuple[np.ndarray, np.ndarray]:
    """t along the tube, 0..1 -> its point on the map."""
    y = s["margin"] + t * (1 - 2 * s["margin"])
    x = 0.5 + s["amp"] * np.sin(2 * np.pi * s["freq"] * t + s["phase"])
    return x, y


def radius(t: np.ndarray, s: dict) -> np.ndarray:
    """Filament thickness along its length."""
    waist = np.abs(2 * t - 1) ** s["pinch"]          # 0 at the middle, 1 at the ends
    return s["r_mid"] + (s["r_end"] - s["r_mid"]) * waist


def strand(t: np.ndarray, k: int, s: dict) -> tuple[np.ndarray, np.ndarray]:
    """One filament: the centreline, pushed aside by an amount that is zero in
    the middle and largest at the two ends.

    This is what "opening out" means here -- not one tube getting fatter, but
    the body coming apart into separate strands as it reaches either end. In
    the middle they all lie on top of each other and read as a single mass."""
    n = max(int(s["strands"]), 1)
    place = 0.0 if n == 1 else (2 * k / (n - 1) - 1)      # -1 .. 1 across the fan

    cx, cy = centreline(t, s)

    # Each strand is the whole centreline turned by its own small angle about
    # the waist. Offsetting sideways instead makes the strands hook: near the
    # ends the sideways motion grows faster than the curve advances, so a
    # strand ends up travelling across the flow rather than along it. Turning
    # the curve keeps every strand running the same way, and they part in
    # proportion to how far they are from the waist -- a fan, not a whisk.
    mid = len(t) // 2
    ox, oy = cx[mid], cy[mid]
    angle = place * s["spread"]
    ca, sa = np.cos(angle), np.sin(angle)
    dx, dy = cx - ox, cy - oy
    rx, ry = ox + dx * ca - dy * sa, oy + dx * sa + dy * ca

    bow = s["wobble"] * np.sin(2 * np.pi * (1.3 * t + 0.37 * k)) * np.abs(2 * t - 1) ** s["open"]
    return rx + bow, ry


def field(size: int, s: dict) -> np.ndarray:
    """-> height 0..1. Distance to the centreline, read as a circular section.

    The distance is taken to the sampled polyline, so the two ends close
    themselves off as rounded caps with no extra work."""
    t = np.linspace(0, 1, int(s["samples"]))
    xs, ys, rs = [], [], []
    for k in range(max(int(s["strands"]), 1)):
        kx, ky = strand(t, k, s)
        xs.append(kx)
        ys.append(ky)
        rs.append(radius(t, s))
    cx, cy, cr = np.concatenate(xs), np.concatenate(ys), np.concatenate(rs)

    px = (np.arange(size) + 0.5) / size
    gx, gy = np.meshgrid(px, px)
    gx, gy = gx.ravel(), gy.ravel()

    # nearest sample per pixel, in chunks so the pairwise distances stay small
    best_d = np.full(gx.shape, np.inf, dtype=np.float32)
    best_r = np.zeros(gx.shape, dtype=np.float32)
    for start in range(0, len(cx), 64):      # every strand's samples, not just the first
        sx, sy, sr = cx[start:start + 64], cy[start:start + 64], cr[start:start + 64]
        d = np.hypot(gx[:, None] - sx[None, :], gy[:, None] - sy[None, :])
        k = d.argmin(axis=1)
        dk = d[np.arange(len(gx)), k]
        closer = dk < best_d
        best_d[closer] = dk[closer]
        best_r[closer] = sr[k][closer]

    # circular section: a thinner part of the tube is lower as well as narrower
    inside = np.maximum(best_r ** 2 - best_d ** 2, 0)
    return (np.sqrt(inside) / max(s["r_end"], s["r_mid"])).reshape(size, size).astype(np.float32)


def main() -> None:
    s = dict(SETTINGS)
    for arg in sys.argv[1:]:
        key, _, value = arg.partition("=")
        if key not in s:
            raise SystemExit(f"unknown setting {key} (have {', '.join(s)})")
        s[key] = type(SETTINGS[key])(value)

    size = int(s["size"])
    height = field(size, s)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"fan_s{int(s['strands'])}_sp{s['spread']:g}_op{s['open']:g}_r{s['r_mid']:g}"
    raw = (np.clip(height, 0, 1) * 65535).round().astype("<u2")
    Image.frombytes("I;16", (size, size), raw.tobytes()).save(OUT / f"{stem}_height16.png")
    Image.fromarray((np.clip(height, 0, 1) * 255).round().astype(np.uint8)).save(
        OUT / f"{stem}_preview.png")
    tiles_mod.hillshade(height, exaggeration=size / 60).save(OUT / f"{stem}_shaded.png")

    covered = (height > 0).mean()
    print(f"tube: ends r={s['r_end']:g}, waist r={s['r_mid']:g}, "
          f"covers {covered:.0%} of the map, {size}x{size} px")
    print(f"-> {OUT / (stem + '_height16.png')}")


if __name__ == "__main__":
    main()
