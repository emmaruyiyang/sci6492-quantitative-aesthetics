"""Textured meshes for optimization.py's MESH_TEXTURE field: a Rhino mesh, prerendered once from
fixed cameras into float G-buffers, then shaded differentiably with an optimized texture.

Prerender (no gradient, once per run): every view is rasterized (rasterize: a z-buffered triangle
rasterizer in plain torch, any device) at the canvas size x supersample into per-pixel maps --
mask, world position, shading normal (turned to the camera: meshes are seen two-sided), the uv
as a grid_sample grid, and for bump mapping the screen-space derivatives of the position. The
geometry never changes, so this is all of it the optimization ever sees.

Shading (every step, differentiable w.r.t. the texture and the colours): the texture is sampled
at each pixel's uv; per mode

    color       the texture is the RGB albedo
    monochrome  one channel, the albedo in gray
    bump        one channel, a height map over the surface (bump_depth: the height of value 1, as
                % of the mesh's bounding radius); its screen-space gradient tilts the normals
                (surface gradient bump mapping, Mikkelsen 2010), the albedo is the material colour

lit (shading.mode "lit") by one directional light, relative to each camera or fixed in the world,
jittered for the models' views: Lambert + Blinn-Phong. Over the background, then averaged down by
the supersample factor, which antialiases the silhouettes.

UVs: the mesh's own texture coordinates (wrapped into 0..1 where they tile), or a projection
computed per pixel from the world position, so a cylindrical / spherical one has no smear at its
seam: planar (as Rhino's Front view shows the mesh), cylindrical or spherical about the bounding
box's centre, z up; u = 1/2 faces the front.

Cameras orbit the bounding box's centre in Rhino's frame (z up; azimuth 0 = the Front view, looking
along +y; 90 = the Right view), evenly spaced in azimuth from `azimuth`, elevations alternating
+- elevation_variation around `elevation`; at the distance where the bounding sphere fills `fill`
of the canvas' short side. fov 0 = orthographic.

write_rhino saves the merged mesh with its uvs and a material: the texture as its bitmap, or (bump)
as its bump map, referenced and embedded.
"""
import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from surface_of_revolution import _choice, _clamped, _filled, _hex

MODES = ["color", "monochrome", "bump"]
UV_SOURCES = ["mesh", "planar", "cylindrical", "spherical"]
NORMALS = ["smooth", "flat"]
SHADINGS = ["lit", "flat"]
LIGHT_SPACES = ["camera", "world"]
WRAPPED_UVS = ("cylindrical", "spherical")  # u wraps around: the texture is sampled circularly
MAX_VIEWS = 32
RASTER_BUDGET = 1 << 22    # candidate pixels per rasterizer chunk (memory ~ 40 B each)
DEPTH_BITS = 30            # depth's resolution in the z-buffer's packed keys
EDGE_TOLERANCE = 1e-6      # barycentric slack, so pixels on a shared edge fall in one triangle or both
DISCONTINUITY = 4.0        # a neighbour farther than this x the median pixel footprint is across an edge
_EPS = 1e-12


# ── Config ─────────────────────────────────────────────────────────────────────

def default_config() -> dict:
    return normalize_config({})


def normalize_config(cfg: dict | None) -> dict:
    """A MESH_TEXTURE config, defaults filled in and validated:
    {path (a .3dm), mode, uv, normals ("smooth": the mesh's vertex normals | "flat": per face),
     supersample (1..4: px per canvas px along each side),
     texture: {resolution [w, h], levels, lr_decay},
     views: {count, per_step (views the models see each step), azimuth, elevation,
       elevation_variation (degrees), fov (degrees, 0 = orthographic), fill, canvas_view (index)},
     shading: {mode, space ("camera": azimuth, elevation relative to each camera, 0, 0 = from it,
       positive = from the right / above | "world": azimuth 0 = from the front, 90 = from the right),
       azimuth, elevation, ambient, specular, shininess, jitter (degrees, the models' views)},
     color, optimize_color (bump: the material), bump_depth (% of the bounding radius)}."""
    cfg = _filled({"path": "", "mode": "color", "uv": "mesh", "normals": "smooth", "supersample": 2,
                   "color": "#bbbbbb", "optimize_color": False, "bump_depth": 0.5}, cfg)
    texture = _filled({"resolution": [1024, 1024], "levels": 4, "lr_decay": 1.0}, cfg.get("texture"))
    w, h = (int(v) for v in texture["resolution"])
    if not (8 <= w <= 8192 and 8 <= h <= 8192):
        raise ValueError(f"Bad texture resolution {w}x{h}")
    shading = _filled({"mode": "lit", "space": "camera", "azimuth": -30.0, "elevation": 30.0, "ambient": 0.35,
                       "specular": 0.0, "shininess": 32.0, "jitter": 20.0}, cfg.get("shading"))
    return {
        "path": str(cfg["path"]).strip(), "mode": _choice(cfg["mode"], MODES, "mesh texture mode"),
        "uv": _choice(cfg["uv"], UV_SOURCES, "uv source"), "normals": _choice(cfg["normals"], NORMALS, "normals"),
        "supersample": int(_clamped(cfg["supersample"], 1, 4)),
        "texture": {"resolution": [w, h], "levels": int(_clamped(texture["levels"], 1, 8)),
                    "lr_decay": _clamped(texture["lr_decay"], 0.05, 1.0)},
        "views": normalize_views(cfg.get("views")),
        "shading": {"mode": _choice(shading["mode"], SHADINGS, "shading"),
                    "space": _choice(shading["space"], LIGHT_SPACES, "light space"),
                    "azimuth": float(shading["azimuth"]), "elevation": _clamped(shading["elevation"], -90.0, 90.0),
                    "ambient": _clamped(shading["ambient"], 0.0, 1.0), "specular": _clamped(shading["specular"], 0.0, 4.0),
                    "shininess": _clamped(shading["shininess"], 1.0, 1024.0),
                    "jitter": _clamped(shading["jitter"], 0.0, 180.0)},
        "color": _hex(cfg["color"], "#bbbbbb"), "optimize_color": bool(cfg["optimize_color"]),
        "bump_depth": _clamped(cfg["bump_depth"], 0.0, 100.0),
    }


def normalize_views(given: dict | None) -> dict:
    """An orbit of cameras (see cameras), defaults filled in and validated: {count, per_step (views
    the models see each step), azimuth, elevation, elevation_variation (degrees), fov (degrees,
    0 = orthographic), fill, canvas_view (index)}."""
    views = _filled({"count": 8, "per_step": 4, "azimuth": 0.0, "elevation": 15.0, "elevation_variation": 15.0,
                     "fov": 30.0, "fill": 0.9, "canvas_view": 0}, given)
    count = int(_clamped(views["count"], 1, MAX_VIEWS))
    return {"count": count, "per_step": int(_clamped(views["per_step"], 1, count)),
            "azimuth": float(views["azimuth"]), "elevation": _clamped(views["elevation"], -89.0, 89.0),
            "elevation_variation": _clamped(views["elevation_variation"], 0.0, 89.0),
            "fov": _clamped(views["fov"], 0.0, 150.0), "fill": _clamped(views["fill"], 0.05, 4.0),
            "canvas_view": int(_clamped(views["canvas_view"], 0, count - 1))}


def schema() -> dict:
    """The choices, for the page."""
    return {"modes": MODES, "uv_sources": UV_SOURCES, "normals": NORMALS, "shadings": SHADINGS,
            "light_spaces": LIGHT_SPACES, "max_views": MAX_VIEWS}


def channels(cfg: dict) -> int:
    """The texture's channels: RGB for "color", else one (gray, or the bump's height)."""
    return 3 if cfg["mode"] == "color" else 1


# ── Mesh ───────────────────────────────────────────────────────────────────────

@dataclass
class Mesh:
    """Every mesh of a .3dm, merged; world units."""
    positions: np.ndarray      # (V, 3)
    normals: np.ndarray        # (V, 3) unit
    uvs: np.ndarray | None     # (V, 2); None when any object has no texture coordinates
    triangles: np.ndarray      # (F, 3)
    objects: int
    without_uvs: int           # how many objects have no texture coordinates

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.positions.min(0), self.positions.max(0)

    def describe(self) -> str:
        return (f"{self.objects} object{'s' if self.objects != 1 else ''}, {len(self.positions)} vertices, "
                f"{len(self.triangles)} triangles" + (", no uvs" if self.uvs is None else ""))


def _object_meshes(geometry) -> list:
    """The rhino3dm meshes of one object: a mesh itself, or the render meshes cached in a Brep /
    Extrusion (a file saved with shaded or rendered views has them)."""
    import rhino3dm as rhino
    if isinstance(geometry, rhino.Mesh):
        return [geometry]
    if isinstance(geometry, rhino.Extrusion):
        mesh = geometry.GetMesh(rhino.MeshType.Any)
        return [mesh] if mesh is not None else []
    if isinstance(geometry, rhino.Brep):
        faces = geometry.Faces
        return [m for m in (faces[i].GetMesh(rhino.MeshType.Any) for i in range(len(faces))) if m is not None]
    return []


def _vertex_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """(V, 3) area-weighted vertex normals."""
    a, b, c = (positions[triangles[:, i]] for i in range(3))
    face = np.cross(b - a, c - a)
    out = np.zeros_like(positions)
    for i in range(3):
        np.add.at(out, triangles[:, i], face)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), _EPS)


def read_mesh(path: str | Path) -> Mesh:
    """Every mesh in a .3dm (meshes, and Breps' / Extrusions' cached render meshes), merged; quads
    split in two. ValueError when there is none."""
    import rhino3dm as rhino
    model = rhino.File3dm.Read(str(path))
    if model is None:
        raise ValueError(f"Could not read {path}")
    positions, normals, uvs, triangles = [], [], [], []
    objects = without_uvs = 0
    offset = 0
    for obj in model.Objects:
        meshes = _object_meshes(obj.Geometry)
        if not meshes:
            continue
        objects += 1
        lacking = False
        for mesh in meshes:
            n = len(mesh.Vertices)
            if n == 0 or len(mesh.Faces) == 0:
                continue
            p = np.array([(v.X, v.Y, v.Z) for v in (mesh.Vertices[i] for i in range(n))], dtype=np.float64)
            tris = []
            for i in range(len(mesh.Faces)):
                a, b, c, d = mesh.Faces[i]
                tris.append((a, b, c))
                if c != d:
                    tris.append((a, c, d))
            t = np.array(tris, dtype=np.int64)
            nn = (np.array([(v.X, v.Y, v.Z) for v in (mesh.Normals[i] for i in range(n))], dtype=np.float64)
                  if len(mesh.Normals) == n else _vertex_normals(p, t))
            nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), _EPS)
            if len(mesh.TextureCoordinates) == n:
                uvs.append(np.array([(q.X, q.Y) for q in (mesh.TextureCoordinates[i] for i in range(n))],
                                    dtype=np.float64))
            else:
                uvs.append(None)
                lacking = True
            positions.append(p)
            normals.append(nn)
            triangles.append(t + offset)
            offset += n
        without_uvs += lacking
    if not positions:
        raise ValueError(f"No meshes in {Path(path).name} (a Brep or Extrusion needs its render mesh: "
                         "save the file from a shaded view, or convert it with _Mesh)")
    return Mesh(np.concatenate(positions), np.concatenate(normals),
                None if any(u is None for u in uvs) else np.concatenate(uvs), np.concatenate(triangles),
                objects, without_uvs)


_MESHES: OrderedDict = OrderedDict()  # (path, mtime) -> Mesh, the latest few


def load_mesh(path: str | Path) -> Mesh:
    """read_mesh, cached by path and modification time."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Mesh file not found: {path}")
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    if key in _MESHES:
        _MESHES.move_to_end(key)
        return _MESHES[key]
    mesh = read_mesh(path)
    _MESHES[key] = mesh
    while len(_MESHES) > 4:
        _MESHES.popitem(last=False)
    return mesh


# ── Cameras ────────────────────────────────────────────────────────────────────

@dataclass
class Camera:
    eye: torch.Tensor       # (3,)
    rotation: torch.Tensor  # (3, 3) rows: right, up, forward
    focal: float            # px: perspective, at depth 1; orthographic, per world unit
    orthographic: bool


def orbit_direction(azimuth: float, elevation: float) -> tuple[float, float, float]:
    """Unit vector from the target towards an orbiting eye, degrees: azimuth 0 = in front (-y),
    90 = to the right (+x), elevation up (+z)."""
    a, e = math.radians(azimuth), math.radians(elevation)
    return math.cos(e) * math.sin(a), -math.cos(e) * math.cos(a), math.sin(e)


def view_angles(views: dict, i: int) -> tuple[float, float]:
    """(azimuth, elevation) of camera i of a views config: evenly around from `azimuth`, every
    other one elevation_variation higher, the rest as much lower."""
    swing = views["elevation_variation"] * (1 if i % 2 == 0 else -1) if views["count"] > 1 else 0.0
    return views["azimuth"] + 360.0 * i / views["count"], max(-89.0, min(89.0, views["elevation"] + swing))


def orbit_camera(views: dict, azimuth: float, elevation: float, center, radius: float,
                 size: tuple[int, int]) -> Camera:
    """A camera at (azimuth, elevation) degrees around `center` (3,), framing a sphere of `radius`
    per a views config's fov (0 = orthographic) and fill, for images of size (w, h)."""
    short = min(size)
    direction = torch.tensor(orbit_direction(azimuth, max(-89.9, min(89.9, elevation))), dtype=torch.float64)
    if views["fov"] > 0:
        half = math.radians(views["fov"]) / 2
        distance = radius / math.sin(math.atan(views["fill"] * math.tan(half)))
        focal, ortho = (short / 2) / math.tan(half), False
    else:
        distance, focal, ortho = 3 * radius, (short / 2) * views["fill"] / radius, True
    eye = torch.as_tensor(np.asarray(center, dtype=np.float64)) + distance * direction
    forward = -direction
    right = F.normalize(torch.linalg.cross(forward, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)), dim=0)
    up = torch.linalg.cross(right, forward)
    return Camera(eye.float(), torch.stack([right, up, forward]).float(), focal, ortho)


def cameras(cfg: dict, center: np.ndarray, radius: float, size: tuple[int, int]) -> list[Camera]:
    """The config's views of a bounding sphere, for images of size (w, h)."""
    v = cfg["views"]
    return [orbit_camera(v, *view_angles(v, i), center, radius, size) for i in range(v["count"])]


# ── Rasterizer ─────────────────────────────────────────────────────────────────

def _barycentric(fx, fy, sa, sb, sc, area):
    """Screen-space barycentrics of points (fx, fy) in triangles a, b, c (signed area `area`)."""
    l0 = ((sc[..., 0] - sb[..., 0]) * (fy - sb[..., 1]) - (sc[..., 1] - sb[..., 1]) * (fx - sb[..., 0])) / area
    l2 = ((sb[..., 0] - sa[..., 0]) * (fy - sa[..., 1]) - (sb[..., 1] - sa[..., 1]) * (fx - sa[..., 0])) / area
    return l0, 1 - l0 - l2, l2


def rasterize(screen: torch.Tensor, depth: torch.Tensor, triangles: torch.Tensor, size: tuple[int, int],
              orthographic: bool, near: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The visible triangle at each pixel centre: screen (V, 2) px (x right, y down), depth (V,)
    along the view, triangles (F, 3), size (w, h). Triangles with a vertex nearer than `near` are
    dropped; both sides are drawn. -> (pixel indices (N,) row-major, their triangles (N,),
    perspective-correct barycentrics (N, 3)). Triangles are sorted by their bounding box and
    rasterized in chunks of about RASTER_BUDGET candidate pixels; a z-buffer of packed
    (depth, triangle) keys keeps the nearest with one scatter_reduce per chunk."""
    w, h = size
    device = screen.device
    sa, sb, sc = screen[triangles[:, 0]], screen[triangles[:, 1]], screen[triangles[:, 2]]
    za, zb, zc = depth[triangles[:, 0]], depth[triangles[:, 1]], depth[triangles[:, 2]]
    area = (sb[:, 0] - sa[:, 0]) * (sc[:, 1] - sa[:, 1]) - (sb[:, 1] - sa[:, 1]) * (sc[:, 0] - sa[:, 0])
    lo, hi = torch.minimum(torch.minimum(sa, sb), sc), torch.maximum(torch.maximum(sa, sb), sc)
    x0 = torch.ceil(lo[:, 0] - 0.5).clamp(0, w).long()
    x1 = torch.floor(hi[:, 0] - 0.5).clamp(-1, w - 1).long()
    y0 = torch.ceil(lo[:, 1] - 0.5).clamp(0, h).long()
    y1 = torch.floor(hi[:, 1] - 0.5).clamp(-1, h - 1).long()
    keep = (area.abs() > _EPS) & (x0 <= x1) & (y0 <= y1) & (torch.minimum(torch.minimum(za, zb), zc) > near)
    ids = keep.nonzero()[:, 0]
    side = (torch.maximum(x1 - x0, y1 - y0) + 1)[ids]
    order = side.argsort()
    ids, side = ids[order], side[order]

    empty = torch.iinfo(torch.int64).max
    zbuf = torch.full((h * w,), empty, dtype=torch.int64, device=device)
    zlo = depth.min()
    zscale = ((1 << DEPTH_BITS) - 1) / (depth.max() - zlo).clamp_min(_EPS)
    sides = side.cpu().numpy().astype(np.int64)
    start, total = 0, len(ids)
    while start < total:
        window = sides[start:start + RASTER_BUDGET]
        cost = np.arange(1, len(window) + 1) * window ** 2  # chunk padded to its largest box: increasing
        end = start + max(1, int(np.searchsorted(cost, RASTER_BUDGET, side="right")))
        cid, s = ids[start:end], int(sides[end - 1])
        start = end
        offsets = torch.arange(s, device=device)
        px = x0[cid, None, None] + offsets[None, None, :]
        py = y0[cid, None, None] + offsets[None, :, None]
        valid = (px <= x1[cid, None, None]) & (py <= y1[cid, None, None])
        l0, l1, l2 = _barycentric(px + 0.5, py + 0.5, sa[cid, None, None], sb[cid, None, None], sc[cid, None, None],
                                  area[cid, None, None])
        inside = valid & (l0 >= -EDGE_TOLERANCE) & (l1 >= -EDGE_TOLERANCE) & (l2 >= -EDGE_TOLERANCE)
        if orthographic:
            z = l0 * za[cid, None, None] + l1 * zb[cid, None, None] + l2 * zc[cid, None, None]
        else:
            z = 1 / (l0 / za[cid, None, None] + l1 / zb[cid, None, None] + l2 / zc[cid, None, None]).clamp_min(_EPS)
        quantized = ((z[inside] - zlo) * zscale).clamp(0, (1 << DEPTH_BITS) - 1).long()
        tri = cid[:, None, None].expand_as(inside)[inside]
        zbuf.scatter_reduce_(0, (py * w + px)[inside], (quantized << 32) | tri, reduce="amin")

    pixels = (zbuf != empty).nonzero()[:, 0]
    tri = zbuf[pixels] & 0xFFFFFFFF
    fx, fy = (pixels % w).float() + 0.5, (pixels // w).float() + 0.5
    l = torch.stack(_barycentric(fx, fy, sa[tri], sb[tri], sc[tri], area[tri]), dim=-1).clamp_min(0)
    if not orthographic:
        l = l / torch.stack([za[tri], zb[tri], zc[tri]], dim=-1)
    return pixels, tri, l / l.sum(-1, keepdim=True).clamp_min(_EPS)


def project(positions: torch.Tensor, camera: Camera, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(V, 2) screen px (x right, y down) and (V,) depth of world positions (V, 3)."""
    w, h = size
    cam = (positions - camera.eye.to(positions)) @ camera.rotation.to(positions).T
    z = cam[:, 2]
    scale = camera.focal if camera.orthographic else camera.focal / z.clamp_min(_EPS)
    return torch.stack([w / 2 + scale * cam[:, 0], h / 2 - scale * cam[:, 1]], dim=-1), z


# ── G-buffers ──────────────────────────────────────────────────────────────────

def projected_uv(kind: str, p: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """(N, 2) uvs of world points (N, 3) by a projection about the bounding box (lo, hi)."""
    extent = (hi - lo).clamp_min(_EPS)
    v = (p[:, 2] - lo[2]) / extent[2]
    if kind == "planar":
        return torch.stack([(p[:, 0] - lo[0]) / extent[0], v], dim=-1)
    d = p - (lo + hi) / 2
    u = torch.atan2(d[:, 0], -d[:, 1]) / (2 * math.pi) + 0.5
    if kind == "spherical":
        v = 0.5 + torch.asin((d[:, 2] / d.norm(dim=-1).clamp_min(_EPS)).clamp(-1, 1)) / math.pi
    return torch.stack([u, v], dim=-1)


def uv_grid(uv: torch.Tensor, wrapped: bool, texture_width: int) -> torch.Tensor:
    """(..., 2) grid_sample coordinates of uvs (..., 2): image top = v 1; uvs outside 0..1 tile.
    wrapped: for a texture padded by one column each side (circularly), so that u wraps."""
    u, v = uv.unbind(-1)
    u = torch.where((u < 0) | (u > 1), u - u.floor(), u)
    v = torch.where((v < 0) | (v > 1), v - v.floor(), v)
    gx = 2 * (u * texture_width + 1) / (texture_width + 2) - 1 if wrapped else 2 * u - 1
    return torch.stack([gx, 1 - 2 * v], dim=-1)


def _neighbour_weights(p: torch.Tensor, mask: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(1, H, W) x2: whether the pixel after / before along `dim` (-1: x, -2: y) is on the same
    surface (both covered, not across a depth discontinuity), as 0 / 1."""
    n = p.shape[dim]
    d = p.narrow(dim, 1, n - 1) - p.narrow(dim, 0, n - 1)
    both = mask.narrow(dim, 1, n - 1) * mask.narrow(dim, 0, n - 1)
    length = d.norm(dim=0, keepdim=True)
    covered = length[both > 0]
    limit = DISCONTINUITY * covered.median() if covered.numel() else torch.tensor(0.0, device=p.device)
    ok = both * (length <= limit)
    pad_after, pad_before = ((0, 1), (1, 0)) if dim == -1 else ((0, 0, 0, 1), (0, 0, 1, 0))
    return F.pad(ok, pad_after), F.pad(ok, pad_before)


def screen_derivative(x: torch.Tensor, after: torch.Tensor, before: torch.Tensor, dim: int) -> torch.Tensor:
    """d x / d pixel along `dim` of (..., C, H, W), from the neighbours on the same surface
    (weights `after`, `before`, broadcastable to x): their mean, one of them, or 0."""
    n = x.shape[dim]
    d = x.narrow(dim, 1, n - 1) - x.narrow(dim, 0, n - 1)
    pad_after, pad_before = ((0, 1), (1, 0)) if dim == -1 else ((0, 0, 0, 1), (0, 0, 1, 0))
    return (after * F.pad(d, pad_after) + before * F.pad(d, pad_before)) / (after + before).clamp_min(1)


@dataclass
class GBuffer:
    """Every view's maps at the canvas size x supersample, stacked (V, C, H, W), on one device."""
    mask: torch.Tensor       # (V, 1, H, W) 0 / 1
    position: torch.Tensor   # (V, 3, H, W) world
    normal: torch.Tensor     # (V, 3, H, W) unit, towards the camera; 0 off the mesh
    grid: torch.Tensor       # (V, H, W, 2) the uvs as grid_sample coordinates
    uv: torch.Tensor         # (V, 2, H, W) the uvs (for the page)
    rotation: torch.Tensor   # (V, 3, 3) the cameras' rows: right, up, forward
    eye: torch.Tensor        # (V, 3)
    orthographic: bool
    supersample: int
    bounds: tuple[torch.Tensor, torch.Tensor]  # the mesh's bounding box, (3,) each
    radius: float            # its bounding sphere's
    wrapped: bool            # u wraps around (cylindrical / spherical uvs)
    # Bump mapping only (else None): the surface gradient's screen-space frame, sign-folded --
    # r_x = sign(det) dP/dy x n, r_y = sign(det) n x dP/dx, |det| = |dP/dx . (dP/dy x n)| -- and
    # which neighbours each pixel differentiates with ((V, 2, H, W): after, before).
    r_x: torch.Tensor | None = None
    r_y: torch.Tensor | None = None
    det: torch.Tensor | None = None
    weights_x: torch.Tensor | None = None
    weights_y: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.mask)

    def maps(self, view: int) -> dict[str, torch.Tensor]:
        """(1, 3, h, w) images in [0, 1] of one view at the canvas size, by name: mask, normals,
        position (in the bounding box), uv (red = u, green = v)."""
        s = self.supersample
        mask = self.mask[view:view + 1]
        lo, hi = self.bounds
        position = (self.position[view:view + 1] - lo.view(1, 3, 1, 1)) / (hi - lo).clamp_min(_EPS).view(1, 3, 1, 1)
        uv = torch.cat([self.uv[view:view + 1] % 1.0, torch.zeros_like(mask)], dim=1)
        maps = {"mask": mask.expand(-1, 3, -1, -1), "normals": (self.normal[view:view + 1] * 0.5 + 0.5) * mask,
                "position": position * mask, "uv": uv * mask}
        return {k: F.avg_pool2d(v, s) if s > 1 else v for k, v in maps.items()}


def prerender(mesh: Mesh, cfg: dict, size: tuple[int, int], device) -> GBuffer:
    """The G-buffers of a config's views of `mesh` for a canvas of size (w, h) (rendered at size x
    supersample). Rasterized on the GPU when `device` is CUDA, else on the CPU (MPS lacks the
    int64 scatter_reduce), then moved to `device`."""
    device = torch.device(device)
    raster = device if device.type == "cuda" else torch.device("cpu")
    s = cfg["supersample"]
    w, h = size[0] * s, size[1] * s
    bump = cfg["mode"] == "bump"
    wrapped = cfg["uv"] in WRAPPED_UVS
    if cfg["uv"] == "mesh" and mesh.uvs is None:
        raise ValueError(f"{mesh.without_uvs} of the mesh's {mesh.objects} objects have no texture coordinates: "
                         f"choose a projected uv source ({', '.join(UV_SOURCES[1:])})")
    positions = torch.from_numpy(mesh.positions).float().to(raster)
    normals = torch.from_numpy(mesh.normals).float().to(raster)
    triangles = torch.from_numpy(mesh.triangles).to(raster)
    uvs = torch.from_numpy(mesh.uvs).float().to(raster) if mesh.uvs is not None else None
    lo, hi = positions.min(0).values, positions.max(0).values
    center, radius = ((lo + hi) / 2), max(float((hi - lo).norm()) / 2, 1e-9)
    if cfg["normals"] == "flat":
        a, b, c = (positions[triangles[:, i]] for i in range(3))
        face_normals = F.normalize(torch.linalg.cross(b - a, c - a), dim=-1)
    texture_width = cfg["texture"]["resolution"][0]

    out = {k: [] for k in ("mask", "position", "normal", "grid", "uv", "r_x", "r_y", "det", "wx", "wy")}
    views = cameras(cfg, center.cpu().numpy(), radius, (w, h))
    for camera in views:
        screen, depth = project(positions, camera, (w, h))
        pixels, tri, bary = rasterize(screen, depth, triangles, (w, h), camera.orthographic, near=1e-3 * radius)
        corners = triangles[tri]  # (N, 3)

        def interpolate(values):
            return (values[corners] * bary[..., None]).sum(1)

        p = interpolate(positions)
        n = face_normals[tri] if cfg["normals"] == "flat" else F.normalize(interpolate(normals), dim=-1)
        eye = camera.eye.to(raster)
        to_eye = -camera.rotation[2].to(raster).expand_as(p) if camera.orthographic else eye - p
        n = torch.where(((n * to_eye).sum(-1, keepdim=True) < 0), -n, n)
        uv = interpolate(uvs) if cfg["uv"] == "mesh" else projected_uv(cfg["uv"], p, lo, hi)

        def dense(values):  # (N, C) at the pixels -> (C, H, W), 0 elsewhere
            image = values.new_zeros(values.shape[1], h * w)
            image[:, pixels] = values.T
            return image.view(-1, h, w)

        mask = dense(torch.ones_like(p[:, :1]))
        position, normal, uv_map = dense(p), dense(n), dense(uv)
        out["mask"].append(mask)
        out["position"].append(position)
        out["normal"].append(normal)
        out["uv"].append(uv_map)
        out["grid"].append(uv_grid(uv_map.permute(1, 2, 0), wrapped, texture_width))
        if bump:
            wx = _neighbour_weights(position, mask, -1)
            wy = _neighbour_weights(position, mask, -2)
            dp_dx = screen_derivative(position, *wx, -1)
            dp_dy = screen_derivative(position, *wy, -2)
            r_x = torch.linalg.cross(dp_dy, normal, dim=0)
            r_y = torch.linalg.cross(normal, dp_dx, dim=0)
            det = (dp_dx * r_x).sum(0, keepdim=True)
            sign = torch.where(det < 0, -1.0, 1.0)
            out["r_x"].append(sign * r_x)
            out["r_y"].append(sign * r_y)
            out["det"].append(det.abs())
            out["wx"].append(torch.cat(wx))
            out["wy"].append(torch.cat(wy))

    stack = lambda key: torch.stack(out[key]).to(device) if out[key] else None
    return GBuffer(stack("mask"), stack("position"), stack("normal"), stack("grid"), stack("uv"),
                   torch.stack([c.rotation for c in views]).to(device), torch.stack([c.eye for c in views]).to(device),
                   views[0].orthographic, s, (lo.to(device), hi.to(device)), radius, wrapped,
                   stack("r_x"), stack("r_y"), stack("det"), stack("wx"), stack("wy"))


# ── Shading ────────────────────────────────────────────────────────────────────

def light_directions(gbuffer: GBuffer, shading: dict, views: torch.Tensor,
                     jitter: float = 0.0) -> torch.Tensor:
    """(n, 3) unit vectors towards the light, world space, for views (n,) (indices), each with its
    azimuth and elevation moved by up to +- jitter degrees."""
    n = len(views)
    device = gbuffer.mask.device
    azimuth = torch.full((n,), shading["azimuth"], device=device)
    elevation = torch.full((n,), shading["elevation"], device=device)
    if jitter > 0:
        azimuth = azimuth + jitter * (2 * torch.rand(n, device=device) - 1)
        elevation = (elevation + jitter * (2 * torch.rand(n, device=device) - 1)).clamp(-90, 90)
    a, e = torch.deg2rad(azimuth)[:, None], torch.deg2rad(elevation)[:, None]
    if shading["space"] == "world":
        return torch.cat([e.cos() * a.sin(), -e.cos() * a.cos(), e.sin()], dim=-1)
    right, up, forward = gbuffer.rotation[views].unbind(1)
    return F.normalize(e.cos() * (a.cos() * -forward + a.sin() * right) + e.sin() * up, dim=-1)


def sample(texture: torch.Tensor, gbuffer: GBuffer, views: torch.Tensor) -> torch.Tensor:
    """(n, C, H, W) the (1, C, TH, TW) texture at every pixel of views (n,)."""
    texture = texture.expand(len(views), -1, -1, -1)
    if gbuffer.wrapped:
        texture = F.pad(texture, (1, 1, 0, 0), mode="circular")
    return F.grid_sample(texture, gbuffer.grid[views], mode="bilinear", padding_mode="border", align_corners=False)


def bumped_normals(gbuffer: GBuffer, views: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    """(n, 3, H, W) the normals tilted by a (n, 1, H, W) height over the surface (world units).
    Where the screen-space frame is degenerate (a lone pixel, edge on) they stay as they are."""
    wx, wy = gbuffer.weights_x[views], gbuffer.weights_y[views]
    dh_dx = screen_derivative(height, wx[:, :1], wx[:, 1:], -1)
    dh_dy = screen_derivative(height, wy[:, :1], wy[:, 1:], -2)
    det, normal = gbuffer.det[views], gbuffer.normal[views]
    tilted = F.normalize(det * normal - (dh_dx * gbuffer.r_x[views] + dh_dy * gbuffer.r_y[views]), dim=1)
    return torch.where(det > 0, tilted, normal)


def shade(gbuffer: GBuffer, cfg: dict, views: torch.Tensor, texture: torch.Tensor, color: torch.Tensor,
          background: torch.Tensor, lights: torch.Tensor | None = None) -> torch.Tensor:
    """(n, 3, h, w) views (n,) (indices) at the canvas size: the (1, C, TH, TW) texture in [0, 1]
    on the mesh per cfg's mode (bump: the material `color` (3,)), lit by `lights` (n, 3) (None:
    the config's, unjittered), over `background` (1, 3, h, w)."""
    shading = cfg["shading"]
    texel = sample(texture, gbuffer, views)
    normal = gbuffer.normal[views]
    if cfg["mode"] == "bump":
        depth = cfg["bump_depth"] / 100 * gbuffer.radius
        normal = bumped_normals(gbuffer, views, texel * depth)
        albedo = color.view(1, 3, 1, 1)
    else:
        albedo = texel.expand(-1, 3, -1, -1)
    if shading["mode"] == "lit":
        if lights is None:
            lights = light_directions(gbuffer, shading, views)
        l = lights[:, :, None, None]
        diffuse = (normal * l).sum(1, keepdim=True).clamp_min(0)
        rgb = albedo * (shading["ambient"] + (1 - shading["ambient"]) * diffuse)
        if shading["specular"] > 0:
            if gbuffer.orthographic:
                to_eye = -gbuffer.rotation[views][:, 2, :, None, None].expand_as(normal)
            else:
                to_eye = F.normalize(gbuffer.eye[views][:, :, None, None] - gbuffer.position[views], dim=1)
            half = F.normalize(to_eye + l, dim=1)
            highlight = (normal * half).sum(1, keepdim=True).clamp_min(0) ** shading["shininess"]
            rgb = rgb + shading["specular"] * highlight * (diffuse > 0)
    else:
        rgb = albedo.expand(len(views), -1, *normal.shape[-2:])
    mask = gbuffer.mask[views]
    rgb = rgb * mask
    s = gbuffer.supersample
    if s > 1:
        rgb, mask = F.avg_pool2d(rgb, s), F.avg_pool2d(mask, s)
    return rgb + background * (1 - mask)


# ── Preview ────────────────────────────────────────────────────────────────────

_PREVIEWS: OrderedDict = OrderedDict()  # prerender's key -> GBuffer on the CPU, the latest few


def cached_prerender(mesh: Mesh, cfg: dict, size: tuple[int, int], key: tuple) -> GBuffer:
    """prerender on the CPU, cached: the page's previews re-render while only the texture or the
    lights change."""
    geometry = {k: cfg[k] for k in ("uv", "normals", "supersample", "views")}
    full = (key, repr(geometry), cfg["mode"] == "bump", cfg["texture"]["resolution"][0], tuple(size))
    if full in _PREVIEWS:
        _PREVIEWS.move_to_end(full)
        return _PREVIEWS[full]
    gbuffer = prerender(mesh, cfg, size, "cpu")
    _PREVIEWS[full] = gbuffer
    while len(_PREVIEWS) > 3:
        _PREVIEWS.popitem(last=False)
    return gbuffer


def sheet_layout(count: int, size: tuple[int, int]) -> tuple[int, int, tuple[int, int]]:
    """(cols, rows, tile size): `count` tiles with the canvas' aspect, in about a square."""
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))
    return cols, rows, (max(8, size[0] // cols), max(8, size[1] // cols))


def tile_sheet(tiles: torch.Tensor, size: tuple[int, int], base: torch.Tensor, highlight: int | None = None,
               ) -> torch.Tensor:
    """(1, 3, H, W): tiles (N, 3, th, tw) of sheet_layout(N, size)'s tile size, row-major on `base`
    (1, 3, H, W) (a copy of it), centred vertically; tile `highlight` outlined (when N > 1)."""
    count = len(tiles)
    w, h = size
    cols, rows, (tw, th) = sheet_layout(count, size)
    sheet = base.clone()
    top, left = max(0, (h - rows * th) // 2), (w - cols * tw) // 2
    border = torch.zeros(1, th, tw, dtype=torch.bool, device=tiles.device)
    border[:, :2], border[:, -2:], border[:, :, :2], border[:, :, -2:] = True, True, True, True
    accent = torch.tensor([0.9, 0.25, 0.2], device=tiles.device).view(3, 1, 1)
    for i, image in enumerate(tiles):
        if i == highlight and count > 1:
            image = torch.where(border, accent, image)
        y, x = top + (i // cols) * th, left + (i % cols) * tw
        sheet[0, :, y:y + th, x:x + tw] = image[:, :min(th, h - y), :min(tw, w - x)]
    return sheet


def preview(cfg: dict, size: tuple[int, int], texture: torch.Tensor, color: torch.Tensor,
            background: Callable[[tuple[int, int]], torch.Tensor]) -> tuple[torch.Tensor, str]:
    """(1, 3, H, W) every view as a run starts, tiled on a canvas of size (w, h) (row-major, the
    canvas view outlined), and a description of the mesh. background(size) -> (1, 3, h, w). CPU."""
    cfg = normalize_config(cfg)
    if not cfg["path"]:
        raise ValueError("No mesh file")
    mesh = load_mesh(cfg["path"])
    path = Path(cfg["path"])
    count = cfg["views"]["count"]
    _, _, tile = sheet_layout(count, size)
    gbuffer = cached_prerender(mesh, cfg, tile, (str(path.resolve()), path.stat().st_mtime_ns))
    with torch.no_grad():
        tiles = shade(gbuffer, cfg, torch.arange(count), texture, color, background(tile))
        sheet = tile_sheet(tiles, size, background(size), cfg["views"]["canvas_view"])
    return sheet, mesh.describe()


# ── Export ─────────────────────────────────────────────────────────────────────

def vertex_uvs(mesh: Mesh, kind: str) -> np.ndarray:
    """(V, 2) per-vertex uvs: the mesh's, or the projection's (a cylindrical / spherical one smears
    across its seam here, unlike in the per-pixel render)."""
    if kind == "mesh" and mesh.uvs is not None:
        return mesh.uvs
    lo, hi = (torch.from_numpy(b) for b in mesh.bounds)
    return projected_uv(kind, torch.from_numpy(mesh.positions), lo, hi).numpy()


def write_rhino(path: Path, mesh: Mesh, cfg: dict, color: tuple[int, int, int], texture: Path,
                notes: dict[str, str] | None = None) -> None:
    """A .3dm with the merged mesh on a layer of its own, its uvs, and a material: the image
    `texture` as its bitmap (color, monochrome) or bump map over `color` (bump), referenced by its
    path and embedded in the file. notes: user strings on the mesh."""
    import rhino3dm as rhino
    out = rhino.Mesh()
    for x, y, z in mesh.positions.tolist():
        out.Vertices.Add(x, y, z)
    for x, y, z in mesh.normals.tolist():
        out.Normals.Add(x, y, z)
    for a, b, c in mesh.triangles.tolist():
        out.Faces.AddFace(a, b, c)
    uvs = vertex_uvs(mesh, cfg["uv"])
    # As surface_of_revolution.write_rhino: rhino3dm can only replace coordinates computed from a mapping.
    unit = rhino.Interval(0, 1)
    out.SetTextureCoordinates(rhino.TextureMapping.CreatePlaneMapping(rhino.Plane.WorldXY(), unit, unit, unit),
                              rhino.Transform.Identity(), False)
    if len(out.TextureCoordinates) != len(uvs):
        raise RuntimeError(f"rhino3dm made {len(out.TextureCoordinates)} texture coordinates for {len(uvs)} vertices")
    for i, (u, v) in enumerate(uvs.tolist()):
        out.TextureCoordinates[i] = rhino.Point2f(u, v)

    model = rhino.File3dm()
    material = rhino.Material()
    material.Name = "mesh_texture"
    path_text = str(Path(texture).resolve())
    if cfg["mode"] == "bump":
        material.DiffuseColor = (*color, 255)
        material.SetBumpTexture(path_text)
    else:
        material.DiffuseColor = (255, 255, 255, 255)
        material.SetBitmapTexture(path_text)
    model.EmbeddedFiles.Add(rhino.EmbeddedFile.Read(path_text))
    attributes = rhino.ObjectAttributes()
    attributes.Name = "mesh_texture"
    attributes.LayerIndex = model.Layers.AddLayer("mesh_texture", (*color, 255))
    attributes.MaterialSource = rhino.ObjectMaterialSource.MaterialFromObject
    attributes.MaterialIndex = model.Materials.Add(material)
    for key, value in (notes or {}).items():
        attributes.SetUserString(key, str(value))
    model.Objects.AddMesh(out, attributes)
    if not model.Write(str(path), 8):
        raise RuntimeError(f"Could not write {path}")
