"""
P4 - cut single cells out of the 500 micrographs and bake them into a sprite
atlas.

The source images are whole microscope fields, not isolated cells, and they
are not one kind of image: bright-field histology (dark cells on white),
fluorescence (bright cells on black), phase contrast, stained sections. So
nothing here may assume "cells are the bright bits".

The pipeline is:

  background colour  <- median of a border ring (works for white AND black)
  foregroundness fg  <- colour distance from that background, normalised
  blob detection     <- multiscale |LoG| maxima on fg  (scale -> radius)
  scoring            <- contrast of the blob against its own annulus, and
                        how saturated it is, so flat background blobs lose
  crop + alpha       <- radial window x fg, so a crop taken out of dense
                        tissue still comes out as a round cell
  exposure           <- each tile normalised to a common mean luminance,
                        scaling RGB so hue survives

At most two cells are taken from any one photograph, so the atlas is wide
rather than 100 crops of the same slide.
"""
import json, math, sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "sample" / "topography" / "cell"
SRC_DEPTH = ROOT.parent / "sample" / "topography" / "cell_depth"
OUT = ROOT / "public" / "cells"

TILE = 128
COLS = 10
ROWS = 10
N_TILES = COLS * ROWS
PER_IMAGE = 3
WORK = 420            # longest side the detector runs at
SCALES = [5.0, 7.0, 9.5, 13.0, 18.0, 25.0]
R_MIN, R_MAX = 7.0, 46.0          # px at WORK; bigger blobs are whole organisms
TARGET_LUM = 0.42
MAX_GAIN = 1.9                    # beyond this, clipping turns pastel into neon


def load(path):
    im = Image.open(path).convert("RGB")
    k = WORK / max(im.size)
    if k < 1:
        im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def load_depth(name, shape):
    """The matching monocular depth map, resampled onto the working image.

    These are good: every cell comes out as a clean convex bump, which is
    exactly the height field the renderer wants. Missing or mismatched files
    are not fatal - that tile just gets a flat normal."""
    f = SRC_DEPTH / (name + ".png")
    if not f.exists():
        return None
    d = Image.open(f).convert("L").resize((shape[1], shape[0]), Image.LANCZOS)
    return np.asarray(d).astype(np.float32) / 255.0


def foreground(rgb):
    h, w, _ = rgb.shape
    b = max(3, int(min(h, w) * 0.06))
    ring = np.concatenate([
        rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
        rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    d = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
    p = np.percentile(d, 98)
    if p < 1e-3:
        return None, bg
    return np.clip(d / p, 0, 1), bg


def blobs(fg):
    """Multiscale LoG. Returns (y, x, radius, strength), strongest first."""
    best = None
    bests = None
    for s in SCALES:
        r = -ndi.gaussian_laplace(fg, s) * (s * s)     # + at bright blobs
        if best is None:
            best, bests = r, np.full(r.shape, s, np.float32)
        else:
            m = r > best
            best = np.where(m, r, best)
            bests = np.where(m, s, bests)
    mx = ndi.maximum_filter(best, size=9)
    peaks = (best == mx) & (best > 0.02)
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return []
    out = [(int(y), int(x), float(bests[y, x]) * math.sqrt(2), float(best[y, x]))
           for y, x in zip(ys, xs)]
    out.sort(key=lambda t: -t[3])
    return out[:400]


def disc(shape, cy, cx, r):
    yy = np.arange(shape[0])[:, None] - cy
    xx = np.arange(shape[1])[None, :] - cx
    return np.sqrt(yy * yy + xx * xx) / max(r, 1e-6)


def score(rgb, fg, lap, y, x, r):
    h, w, _ = rgb.shape
    if not (R_MIN <= r <= R_MAX):
        return None
    if r * 1.55 > min(y, x, h - 1 - y, w - 1 - x):
        return None                                   # too close to the edge
    rr = disc(fg.shape, y, x, r)
    inside = rr < 0.80
    ring = (rr > 1.25) & (rr < 1.80)
    if inside.sum() < 30 or ring.sum() < 30:
        return None
    fi, fo = fg[inside].mean(), fg[ring].mean()
    if fi < 0.18:
        return None

    # circularity: the blob should actually fill its own disc and the annulus
    # around it should not. Without this the detector happily returns a patch
    # of flat background between two real cells.
    fill = float((fg[inside] > 0.45).mean())
    halo = float((fg[ring] > 0.45).mean())
    if fill < 0.55:
        return None

    px = rgb[inside]
    mx, mn = px.max(axis=1), px.min(axis=1)
    sat = float(np.mean((mx - mn) / np.maximum(mx, 1e-3)))
    # detail, measured where the cell is, at the cell's own scale. A soft
    # out-of-focus ball and a cell with visible structure score the same on
    # contrast alone, and half the first atlas came out as soft balls.
    detail = float(lap[inside].mean()) / max(fi, 1e-3)

    return (fi - fo) * (0.25 + sat) * (0.15 + min(2.2, detail / 0.045)) * (1.0 - 0.6 * halo)


def normal_tile(depth, y, x, R, alpha):
    """Crop the depth patch to match the colour tile and turn it into a
    normal map.

    The gradient is normalised PER TILE against its own 90th percentile, so
    every cell gets the same relief strength however the depth model happened
    to scale that photograph - monocular depth is relative, and two tiles from
    two photographs are not on a common scale."""
    flat = np.dstack([np.full((TILE, TILE), 0.5, np.float32),
                      np.full((TILE, TILE), 0.5, np.float32),
                      np.zeros((TILE, TILE), np.float32)])
    if depth is None:
        return flat
    patch = depth[y - R:y + R, x - R:x + R]
    if patch.shape[0] != 2 * R or patch.shape[1] != 2 * R:
        return flat
    # BILINEAR, not LANCZOS: the depth maps are 8-bit and a cell often spans
    # only 20-40 levels, so they arrive already terraced. Lanczos rings on
    # every one of those steps and the gradient turns the terraces into
    # corduroy - which is exactly what the first bake looked like.
    h = np.asarray(Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8))
                   .resize((TILE, TILE), Image.BILINEAR)).astype(np.float32) / 255.0

    m = alpha > 0.4
    if m.sum() > 50:
        lo, hi = np.percentile(h[m], [5, 95])
    else:
        lo, hi = float(h.min()), float(h.max())
    h = np.clip((h - lo) / max(hi - lo, 1e-3), 0, 1)
    # A cell's bump spans ~100 px of the tile, so smoothing at sigma 3.5 costs
    # nothing of the shape and removes the quantisation steps entirely.
    h = ndi.gaussian_filter(h, 3.5)

    gx = ndi.sobel(h, axis=1) / 8.0
    gy = ndi.sobel(h, axis=0) / 8.0
    mag = np.sqrt(gx * gx + gy * gy)
    ref = float(np.percentile(mag[m], 90)) if m.sum() > 50 else float(mag.max())
    k = 0.8 / max(ref, 1e-4)
    nx, ny = -gx * k, -gy * k
    # Clamp the LENGTH, not each component: clamping separately lets nx,ny
    # leave the unit disc, and then the shader's nz = sqrt(1-nx^2-ny^2) has
    # to be faked.
    ln = np.sqrt(nx * nx + ny * ny)
    sc = np.minimum(1.0, 0.95 / np.maximum(ln, 1e-6))
    nx *= sc
    ny *= sc
    return np.dstack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, h])


def cut(rgb, fg, y, x, r):
    """Crop a square and build the RGBA tile."""
    R = int(round(r * 1.42))
    y0, x0 = y - R, x - R
    patch = rgb[y0:y0 + 2 * R, x0:x0 + 2 * R]
    fpatch = fg[y0:y0 + 2 * R, x0:x0 + 2 * R]
    if patch.shape[0] != 2 * R or patch.shape[1] != 2 * R:
        return None

    im = Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8)).resize(
        (TILE, TILE), Image.LANCZOS)
    col = np.asarray(im).astype(np.float32) / 255.0
    fv = np.asarray(Image.fromarray((fpatch * 255).astype(np.uint8)).resize(
        (TILE, TILE), Image.LANCZOS)).astype(np.float32) / 255.0

    # alpha: how much this pixel is "not background", windowed to a disc so a
    # crop taken out of solid tissue still reads as one round cell.
    p = np.percentile(fv, 80)
    a = np.clip(fv / max(p, 0.10), 0, 1) ** 0.9
    rr = disc((TILE, TILE), TILE / 2 - 0.5, TILE / 2 - 0.5, TILE * 0.5)
    window = np.clip((1.0 - rr) / 0.16, 0, 1)
    window = window * window * (3 - 2 * window)
    a = a * window
    if a.max() < 0.35 or (a > 0.5).mean() < 0.06:
        return None

    # exposure: equalise mean luminance across tiles, scaling RGB so hue and
    # internal contrast survive. Histology slides are pale, fluorescence is
    # dark; untouched they sit two stops apart in the same tissue.
    lum = (col * np.array([0.2126, 0.7152, 0.0722])).sum(axis=2)
    m = float((lum * a).sum() / max(a.sum(), 1e-6))
    if m < 0.02:
        return None
    # Equalising exposure by multiplying and clipping is what turned pale
    # histology lavender into neon magenta: the two low channels stay put
    # while the high one pins at 1.0, which IS a hue rotation. Cap the gain
    # and roll the top off smoothly instead of clipping it.
    col = col * float(np.clip(TARGET_LUM / m, 0.55, MAX_GAIN))
    K = 0.80
    hi = col > K
    col = np.where(hi, K + (1 - K) * (1 - np.exp(-(col - K) / (1 - K))), col)
    # and pull back anything that is still a primary
    mx2, mn2 = col.max(axis=2, keepdims=True), col.min(axis=2, keepdims=True)
    sat2 = (mx2 - mn2) / np.maximum(mx2, 1e-3)
    g = np.clip((sat2 - 0.80) / 0.20, 0, 1)
    l2 = (col * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(axis=2, keepdims=True)
    col = col * (1 - 0.45 * g) + l2 * (0.45 * g)

    out = np.concatenate([col, a[:, :, None]], axis=2)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), a, R


def main():
    files = sorted(SRC.glob("*.jpg"))
    if not files:
        sys.exit(f"no images in {SRC}")
    cands = []
    for n, f in enumerate(files):
        if n % 50 == 0:
            print(f"  {n}/{len(files)} scanned, {len(cands)} candidates", flush=True)
        try:
            rgb = load(f)
        except Exception as e:
            print(f"  skip {f.name}: {e}")
            continue
        fg, bg = foreground(rgb)
        if fg is None:
            continue
        depth = None          # loaded lazily, only if a blob survives scoring
        taken = []
        lum = (rgb * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(axis=2)
        lap = np.abs(ndi.laplace(ndi.gaussian_filter(lum, 1.0)))
        for (y, x, r, _) in blobs(fg):
            if len(taken) >= PER_IMAGE:
                break
            if any((y - ty) ** 2 + (x - tx) ** 2 < (r + tr) ** 2 for ty, tx, tr in taken):
                continue
            s = score(rgb, fg, lap, y, x, r)
            if s is None or s < 0.06:
                continue
            got = cut(rgb, fg, y, x, r)
            if got is None:
                continue
            tile, alpha, R = got
            if depth is None:
                depth = load_depth(f.stem, rgb.shape[:2])
            nrm = normal_tile(depth, y, x, R, alpha)
            taken.append((y, x, r))
            cands.append({"score": float(s), "src": f.name, "r": float(r),
                          "tile": tile, "normal": nrm})

    cands.sort(key=lambda c: -c["score"])
    # spread the pick over source images: first pass takes one per photograph
    picked, seen = [], set()
    for c in cands:
        if c["src"] in seen:
            continue
        seen.add(c["src"])
        picked.append(c)
        if len(picked) >= N_TILES:
            break
    print(f"{len(cands)} candidates from {len(seen)} photographs -> {len(picked)} tiles")
    if not picked:
        sys.exit("no usable cells found")

    atlas = np.zeros((ROWS * TILE, COLS * TILE, 4), np.uint8)
    normals = np.zeros((ROWS * TILE, COLS * TILE, 3), np.uint8)
    for i, c in enumerate(picked):
        cy, cx = (i // COLS) * TILE, (i % COLS) * TILE
        atlas[cy:cy + TILE, cx:cx + TILE] = c["tile"]
        normals[cy:cy + TILE, cx:cx + TILE] = (np.clip(c["normal"], 0, 1) * 255).astype(np.uint8)

    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas, "RGBA").save(OUT / "cell_atlas.png", optimize=True)
    Image.fromarray(normals, "RGB").save(OUT / "cell_normal.png", optimize=True)
    # Alpha-weighted mean colour of each tile. The renderer needs it to
    # recolour a cell IMPOSTOR to the tissue it is standing in for: the atlas
    # only holds 12 baked variants, so without this every chunk past the
    # full-detail threshold would turn the same average beige.
    means = []
    for c in picked:
        t = c["tile"].astype(np.float32) / 255.0
        a = t[:, :, 3]
        w = max(float(a.sum()), 1e-6)
        means.append([round(float((t[:, :, k] * a).sum() / w), 4) for k in range(3)])

    meta = {
        "tile": TILE, "cols": COLS, "rows": ROWS, "count": len(picked),
        "means": means, "normals": "cell_normal.png",
        "sources": [c["src"] for c in picked],
        "scores": [round(c["score"], 4) for c in picked],
    }
    (OUT / "cells.json").write_text(json.dumps(meta))
    sz = (OUT / "cell_atlas.png").stat().st_size
    nz = (OUT / "cell_normal.png").stat().st_size
    withDepth = sum(1 for c in picked if float(np.std(c["normal"][:, :, 2])) > 1e-3)
    print(f"wrote {OUT/'cell_atlas.png'}  {COLS*TILE}x{ROWS*TILE}  {sz/1024:.0f} KB")
    print(f"wrote {OUT/'cell_normal.png'}  {nz/1024:.0f} KB  ({withDepth}/{len(picked)} tiles have depth)")
    print(f"score range {picked[-1]['score']:.3f} .. {picked[0]['score']:.3f}")


if __name__ == "__main__":
    main()
