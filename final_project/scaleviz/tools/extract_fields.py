"""
P1 - density field extraction.

Reads the two reference images and turns them into compact fields the
runtime samples from. Nothing here runs in the browser; the app only ever
loads the small binaries written to public/fields/.

galaxy_ref.jpg  ->  a luminous band, a warm core, dark dust lanes, stars
    diffuse.bin   u8  low-pass luminance      -> where the glow lives
    stars.bin     u8  high-pass luminance     -> where discrete stars live
    color.bin     u8  rgb, blurred            -> colour at each position

river_ref.jpg   ->  a single meandering channel on white
    mask.bin      u8  inside/outside
    dist.bin      u8  normalised distance to bank -> channel width field
    spine.json        ordered centreline + half-width + colour

Run:
  uv run --with numpy --with pillow --with scipy python tools/extract_fields.py
"""

import json
import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_closing, binary_opening

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "refs")
OUT = os.path.join(ROOT, "public", "fields")
os.makedirs(OUT, exist_ok=True)

meta = {}


def write_u8(name, arr):
    a = np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
    p = os.path.join(OUT, name)
    a.tofile(p)
    return {"file": "fields/" + name, "bytes": a.nbytes}


def fit(shape, longest):
    """Target grid preserving aspect, long side == longest."""
    h, w = shape[:2]
    if h >= w:
        return (longest, max(1, int(round(longest * w / h))))
    return (max(1, int(round(longest * h / w))), longest)


def resize(a, hw):
    im = Image.fromarray(a)
    return np.asarray(im.resize((hw[1], hw[0]), Image.LANCZOS))


# --------------------------------------------------------------------------
# GALAXY
# --------------------------------------------------------------------------

g = Image.open(os.path.join(REFS, "galaxy_ref.jpg")).convert("RGB")
grgb = np.asarray(g).astype(np.float32) / 255.0
print(f"galaxy_ref {g.size[0]}x{g.size[1]}")

lum = grgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Split the photograph into the two things it actually contains.
# Low pass = the band of unresolved light, including the dust lanes that
# carve it. High pass = the resolved stars sitting on top of it.
diffuse = gaussian_filter(lum, sigma=6.0)
stars = np.clip(lum - gaussian_filter(lum, sigma=1.6), 0.0, None)

# Stretch the glow so sampling follows the band rather than the whole frame.
d = diffuse - diffuse.min()
d /= max(d.max(), 1e-6)
d = np.power(d, 1.45)

s = stars / max(stars.max(), 1e-6)
s = np.power(s, 0.75)

# The photograph has a rectangular frame; a galaxy does not. Feather the
# density to zero at the border so the chunk has no visible edge.
yy = np.linspace(-1, 1, lum.shape[0])[:, None]
xx = np.linspace(-1, 1, lum.shape[1])[None, :]
edge = np.minimum(1 - np.abs(yy), 1 - np.abs(xx))
feather = np.clip(edge / 0.18, 0, 1) ** 1.5
d *= feather
s *= feather

GH, GW = fit(lum.shape, 384)
d_small = resize((d * 255).astype(np.uint8), (GH, GW)).astype(np.float32) / 255.0
s_small = resize((s * 255).astype(np.uint8), (GH, GW)).astype(np.float32) / 255.0

# Colour: blur hard so a point picks up the local colour of the band
# (warm core, cool rim) and not the colour of one pixel of one star.
col = np.stack([gaussian_filter(grgb[:, :, k], sigma=9.0) for k in range(3)], -1)
col = col / max(col.max(), 1e-6)
col = np.power(col, 0.85)
CH, CW = fit(lum.shape, 128)
col_small = resize((col * 255).astype(np.uint8), (CH, CW))

meta["galaxy"] = {
    "w": GW, "h": GH,
    "diffuse": write_u8("galaxy_diffuse.bin", d_small * 255),
    "stars": write_u8("galaxy_stars.bin", s_small * 255),
    "color": {"w": CW, "h": CH, **write_u8("galaxy_color.bin", col_small)},
    "diffuseMean": float(d_small.mean()),
    "starsMean": float(s_small.mean()),
}
print(f"  grid {GW}x{GH}  diffuse mean {d_small.mean():.3f}  stars mean {s_small.mean():.4f}")

# --------------------------------------------------------------------------
# RIVER
# --------------------------------------------------------------------------

r = Image.open(os.path.join(REFS, "river_ref.jpg")).convert("RGB")
rrgb = np.asarray(r).astype(np.float32) / 255.0
print(f"river_ref  {r.size[0]}x{r.size[1]}")

mx = rrgb.max(-1)
mn = rrgb.min(-1)
sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
# white page = high value, no saturation. the water is both darker and coloured.
mask = (sat > 0.10) | (mx < 0.88)
mask = binary_closing(mask, np.ones((5, 5)))
mask = binary_opening(mask, np.ones((3, 3)))

# keep only the largest connected blob (drops jpeg speckle on the page)
from scipy.ndimage import label
lab, n = label(mask)
if n > 1:
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    mask = lab == sizes.argmax()
print(f"  mask covers {mask.mean()*100:.1f}% of the frame, {n} blobs -> 1")

dist = distance_transform_edt(mask).astype(np.float32)
dmax = float(dist.max())

# Centreline: the river runs top to bottom, so for every row the ridge of the
# distance transform is the middle of the channel. No skeletonisation needed.
H, W = mask.shape
spine = []
for y in range(H):
    row = dist[y]
    if row.max() <= 1.0:
        continue
    x = int(row.argmax())
    spine.append((y, x, float(row[x])))

# resample to a fixed number of stations along arc length
spine = np.array(spine, dtype=np.float32)
seg = np.sqrt(np.diff(spine[:, 1]) ** 2 + np.diff(spine[:, 0]) ** 2)
arc = np.concatenate([[0.0], np.cumsum(seg)])
total = float(arc[-1])
N = 220
t = np.linspace(0, total, N)
sx = np.interp(t, arc, spine[:, 1])
sy = np.interp(t, arc, spine[:, 0])
sw = np.interp(t, arc, spine[:, 2])

# light smoothing: keep the meander, drop row-quantisation stair steps
k = np.array([1, 4, 6, 4, 1], dtype=np.float32)
k /= k.sum()
sx = np.convolve(np.pad(sx, 2, mode="edge"), k, mode="valid")
sy = np.convolve(np.pad(sy, 2, mode="edge"), k, mode="valid")
sw = np.convolve(np.pad(sw, 2, mode="edge"), k, mode="valid")

# normalise into [-1,1] with the long axis spanning the range
scale = 2.0 / max(H, W)
cx, cy = W / 2.0, H / 2.0
nx = (sx - cx) * scale
ny = -(sy - cy) * scale
nw = sw * scale

# turning angle per unit length -> this is the meander signature we reuse
dx = np.gradient(nx)
dy = np.gradient(ny)
head = np.unwrap(np.arctan2(dy, dx))
curv = np.gradient(head)
sinuosity = total / max(float(np.hypot(sx[-1] - sx[0], sy[-1] - sy[0])), 1e-6)

# colour along the channel, sampled from the photograph
cols = []
for i in range(N):
    yy = int(np.clip(sy[i], 0, H - 1))
    xx = int(np.clip(sx[i], 0, W - 1))
    patch = rrgb[max(0, yy - 6):yy + 7, max(0, xx - 6):xx + 7]
    m = mask[max(0, yy - 6):yy + 7, max(0, xx - 6):xx + 7]
    cols.append(patch[m].mean(0) if m.any() else patch.reshape(-1, 3).mean(0))
cols = np.array(cols, dtype=np.float32)

MH, MW = fit(mask.shape, 256)
mask_small = resize((mask * 255).astype(np.uint8), (MH, MW))
dist_small = resize((dist / max(dmax, 1e-6) * 255).astype(np.uint8), (MH, MW))

spine_json = {
    "n": N,
    "x": [round(float(v), 5) for v in nx],
    "y": [round(float(v), 5) for v in ny],
    "halfWidth": [round(float(v), 5) for v in nw],
    "curvature": [round(float(v), 5) for v in curv],
    "color": [[round(float(c), 4) for c in rgb] for rgb in cols],
    "sinuosity": round(float(sinuosity), 4),
    # arc length in the same normalised units as x/y/halfWidth, so the runtime
    # can keep width-to-length proportion when it regrows the channel
    "arcLength": round(float(total * scale), 5),
    "curvRms": round(float(np.sqrt((curv ** 2).mean())), 5),
    "meanHalfWidth": round(float(nw.mean()), 5),
}
with open(os.path.join(OUT, "river_spine.json"), "w") as f:
    json.dump(spine_json, f)

meta["river"] = {
    "w": MW, "h": MH,
    "mask": write_u8("river_mask.bin", mask_small),
    "dist": write_u8("river_dist.bin", dist_small),
    "spine": {"file": "fields/river_spine.json", "n": N},
    "sinuosity": spine_json["sinuosity"],
    "curvRms": spine_json["curvRms"],
    "meanHalfWidth": spine_json["meanHalfWidth"],
}
print(f"  spine {N} stations  sinuosity {sinuosity:.3f}  curv rms {spine_json['curvRms']:.4f}"
      f"  mean half-width {nw.mean():.4f}")

with open(os.path.join(OUT, "fields.json"), "w") as f:
    json.dump(meta, f, indent=2)

total_bytes = sum(
    os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)
)
print(f"\nwrote {len(os.listdir(OUT))} files, {total_bytes/1024:.0f} KB total -> public/fields/")
