"""3D Gaussian splats for optimization.py's GAUSSIANS_3D field, and the splat compositing that
parametric_surface's SplatRenderer shares.

A scene of N anisotropic 3D Gaussians -- centre, three scales (sigmoid, up to max_size), a rotation
(quaternion), an opacity (sigmoid) and a colour (sigmoid; RGB, or one gray channel) -- inside a unit
sphere, seen by orbiting cameras (mesh_texture's: Rhino's frame, z up, azimuth 0 = the Front view).
Each view projects the Gaussians to the screen (EWA: the 3D covariance through the projection's
Jacobian, plus a small low pass), sorts them by depth and alpha-composites them front to back per
pixel, as in 3D Gaussian splatting (Kerbl et al. 2023), in plain torch on any device. The colours
are view independent: no spherical harmonics, no densification -- the count is fixed.

composite does the per-pixel part: every splat's pixel window is found without autograd, sized by
its own projected radius (splats grouped by the next power of two, so a few big ones don't blow up
the windows of all the small ones), and only the (splat, pixel) entries it covers carry gradients.

Exports: a .ply in the 3D Gaussian splatting layout (most splat viewers open it) and a .3dm point
cloud of the centres in their colours.
"""
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import mesh_texture
from surface_of_revolution import _clamped, _filled

LOW_PASS = 0.3            # px^2 added to every projected covariance (anti-aliasing)
NEAR = 1e-3               # x the scene radius: nearer splats are dropped
WINDOW_BUDGET = 1 << 24   # (splat, pixel) candidates per batch while finding the windows
FRUSTUM_SLACK = 1.3       # x the half field of view: the Jacobian is evaluated no further out
SH_C0 = 0.28209479177387814  # the 0th spherical harmonic, for the .ply's colours


# ── Compositing ────────────────────────────────────────────────────────────────

def composite(xy: torch.Tensor, conic: torch.Tensor, radius: torch.Tensor, depth: torch.Tensor,
              color: torch.Tensor, opacity: torch.Tensor | float, size: tuple[int, int],
              background: torch.Tensor, max_radius: int, near: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Screen-space Gaussians alpha-composited front to back over a background: centres xy (N, 2) px,
    conics (N, 3) (the inverse 2-D covariance: xx, xy, yy), radii (N,) px (3 sigma; the window is
    capped at max_radius), depths (N,), colours (N, C), opacities (N,) or one for all; size (w, h),
    background (C, H, W). Splats nearer than `near` or off screen are culled. -> ((C, H, W) image in
    [0, 1], stats {visible, entries})."""
    w, h = size
    device = xy.device
    if not torch.is_tensor(opacity):
        opacity = xy.new_full((len(xy),), float(opacity))
    with torch.no_grad():
        r = radius.clamp(max=max_radius)
        visible = (depth > near) & (xy[:, 0] + r > 0) & (xy[:, 0] - r < w) & (xy[:, 1] + r > 0) & (xy[:, 1] - r < h)
        idx = visible.nonzero().squeeze(1)
    if not len(idx):
        return background.clone(), {"visible": 0, "entries": 0}
    pick = lambda x: torch.index_select(x, 0, idx)
    xy, conic, color, opacity, z = pick(xy), pick(conic), pick(color), pick(opacity), pick(depth).detach()

    with torch.no_grad():
        n = len(z)
        cutoff = (2 * torch.log(255 * opacity.detach().clamp_min(1e-6))).clamp(0, 9)  # alpha >= 1/255 inside
        window = pick(r).ceil().long().clamp(1, max_radius)
        bucket = torch.where(window > 1, 2 ** torch.ceil(torch.log2(window.float())).long(), window).clamp(max=max_radius)
        corner = xy.floor().long()
        splats, xs, ys = [], [], []
        for size_r in bucket.unique().tolist():
            members = (bucket == size_r).nonzero().squeeze(1)
            offsets = torch.arange(-size_r, size_r + 1, device=device)
            oy, ox = (o.reshape(1, -1) for o in torch.meshgrid(offsets, offsets, indexing="ij"))
            per_batch = max(1, WINDOW_BUDGET // ox.shape[1])
            for batch in members.split(per_batch):
                px, py = corner[batch, :1] + ox, corner[batch, 1:] + oy           # (B, K)
                dx, dy = px + 0.5 - xy[batch, :1], py + 0.5 - xy[batch, 1:]
                c = conic[batch]
                q = c[:, :1] * dx ** 2 + 2 * c[:, 1:2] * dx * dy + c[:, 2:] * dy ** 2
                keep = (px >= 0) & (px < w) & (py >= 0) & (py < h) & (q < cutoff[batch, None])
                row, k = keep.nonzero(as_tuple=True)
                splats.append(batch[row])
                xs.append(px[row, k])
                ys.append(py[row, k])
        splat, px, py = torch.cat(splats), torch.cat(xs), torch.cat(ys)
        pixel = py * w + px
        rank = torch.empty(n, dtype=torch.long, device=device)
        rank[torch.argsort(z)] = torch.arange(n, device=device)
        order = torch.argsort(pixel * n + rank[splat])
        splat, pixel = splat[order], pixel[order]
        offset = torch.stack([px[order], py[order]], dim=-1) + 0.5
    # One gather for everything per entry (index_select: its backward is a plain index_add).
    packed = torch.index_select(torch.cat([xy, conic, opacity[:, None], color], dim=-1), 0, splat)
    d = offset - packed[:, :2]
    c = packed[:, 2:5]
    q = c[:, 0] * d[:, 0] ** 2 + 2 * c[:, 1] * d[:, 0] * d[:, 1] + c[:, 2] * d[:, 1] ** 2
    alpha = packed[:, 5] * torch.exp(-0.5 * q)
    # float64: the cumsum runs over every entry of the frame. MPS has none, so there it runs on the CPU.
    scan = torch.device("cpu") if alpha.device.type == "mps" else alpha.device
    log_t = torch.log1p(-alpha.clamp(max=0.999)).to(scan).double()
    before = torch.cumsum(log_t, 0) - log_t                     # exclusive, global
    start = torch.ones_like(pixel, dtype=torch.bool)
    start[1:] = pixel[1:] != pixel[:-1]
    segment = torch.cumsum(start.long(), 0) - 1
    transmittance = torch.exp(before - before[start.to(scan)][segment.to(scan)]).float().to(device)
    weight = transmittance * alpha                              # sums to the pixel's coverage
    channels = color.shape[1]
    img = torch.zeros(h * w, channels, device=device).index_add(0, pixel, weight[:, None] * packed[:, 6:])
    coverage = torch.zeros(h * w, device=device).index_add(0, pixel, weight)
    img = img + (1 - coverage)[:, None] * background.reshape(channels, -1).T
    return img.clamp(0, 1).T.reshape(channels, h, w), {"visible": n, "entries": len(splat)}


def conic_and_radius(sxx: torch.Tensor, sxy: torch.Tensor, syy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The conic (N, 3) and 3-sigma radius (N,) of 2-D covariances."""
    det = (sxx * syy - sxy ** 2).clamp(min=1e-8)
    conic = torch.stack([syy / det, -sxy / det, sxx / det], dim=-1)
    mid = (sxx + syy) / 2
    return conic, 3 * (mid + (mid ** 2 - det).clamp(min=0).sqrt()).sqrt()


# ── Config ─────────────────────────────────────────────────────────────────────

def default_config() -> dict:
    return normalize_config({})


def normalize_config(cfg: dict | None) -> dict:
    """A GAUSSIANS_3D config, defaults filled in and validated (lengths in scene radii: the cameras
    frame a unit sphere at the origin):
    {count, monochrome, spread (radius of the ball the centres start in), size (starting scale),
     max_size (the scales' bound), opacity (starting), position_lr, shape_lr (x the learning rate:
     centres; scales and rotations), max_radius (px: the largest pixel window of a splat),
     jitter (degrees: the models' cameras move by up to this, in azimuth and elevation),
     views: mesh_texture.normalize_views,
     mesh: {resolution (voxels along the longest side; 0 = no isosurface), level} (the export's
       isosurface of the summed density)}."""
    cfg = _filled({"count": 2000, "monochrome": False, "spread": 0.6, "size": 0.04, "max_size": 0.25, "opacity": 0.5,
                   "position_lr": 1.0, "shape_lr": 1.0, "max_radius": 48, "jitter": 10.0}, cfg)
    mesh = _filled({"resolution": 128, "level": 0.5}, cfg.get("mesh"))
    max_size = _clamped(cfg["max_size"], 1e-3, 2.0)
    return {"count": int(_clamped(cfg["count"], 1, 200_000)), "monochrome": bool(cfg["monochrome"]),
            "spread": _clamped(cfg["spread"], 0.0, 2.0), "size": _clamped(cfg["size"], 1e-4, 0.99 * max_size),
            "max_size": max_size, "opacity": _clamped(cfg["opacity"], 0.01, 0.99),
            "position_lr": _clamped(cfg["position_lr"], 0.0, 100.0), "shape_lr": _clamped(cfg["shape_lr"], 0.0, 100.0),
            "max_radius": int(_clamped(cfg["max_radius"], 4, 256)), "jitter": _clamped(cfg["jitter"], 0.0, 180.0),
            "views": mesh_texture.normalize_views(cfg.get("views")),
            "mesh": {"resolution": int(_clamped(mesh["resolution"], 0, 512)),
                     "level": _clamped(mesh["level"], 1e-3, 100.0)}}


def schema() -> dict:
    return {"max_views": mesh_texture.MAX_VIEWS}


# ── The scene ──────────────────────────────────────────────────────────────────

def _logit(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-3, 1 - 1e-3)
    return torch.log(x / (1 - x))


def rotation_matrices(q: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) from quaternions (N, 4) (w, x, y, z; any length)."""
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1).view(-1, 3, 3)


class Splats:
    """The optimizable Gaussians of a (normalized) config, leaves on `device`. Drawn from torch's
    global generator: seed it first for a repeatable start. colors(means) -> (N, C) in [0, 1], the
    starting colours (None: mid grey)."""

    def __init__(self, cfg: dict, device, colors: Callable[[torch.Tensor], torch.Tensor] | None = None):
        self.cfg = cfg
        n = cfg["count"]
        direction = F.normalize(torch.randn(n, 3), dim=-1)
        means = direction * torch.rand(n, 1) ** (1 / 3) * cfg["spread"]
        scales = (cfg["size"] * torch.exp(0.2 * torch.randn(n, 3))).clamp(max=0.99 * cfg["max_size"])
        channels = 1 if cfg["monochrome"] else 3
        start = colors(means) if colors is not None else torch.full((n, channels), 0.5)
        self.means = means.to(device).requires_grad_(True)
        self.scales = _logit(scales / cfg["max_size"]).to(device).requires_grad_(True)
        self.quats = F.normalize(torch.randn(n, 4), dim=-1).to(device).requires_grad_(True)
        self.opacities = _logit(torch.full((n,), cfg["opacity"])).to(device).requires_grad_(True)
        self.colors = _logit(start.to(torch.float32)).to(device).requires_grad_(True)

    def param_groups(self, lr: float) -> list[dict]:
        return [{"params": [self.means], "lr": lr * self.cfg["position_lr"]},
                {"params": [self.scales, self.quats], "lr": lr * self.cfg["shape_lr"]},
                {"params": [self.opacities, self.colors], "lr": lr}]

    def scale(self) -> torch.Tensor:
        return self.cfg["max_size"] * torch.sigmoid(self.scales)

    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacities)

    def color(self) -> torch.Tensor:
        """(N, 3) in [0, 1] (monochrome: the gray, three times)."""
        return torch.sigmoid(self.colors).expand(-1, 3)

    def covariances(self) -> torch.Tensor:
        """(N, 3, 3) world space."""
        m = rotation_matrices(self.quats) * self.scale()[:, None, :]
        return m @ m.transpose(1, 2)

    def render(self, camera: mesh_texture.Camera, size: tuple[int, int], background: torch.Tensor,
               covariances: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """(3, H, W) one camera's view over `background` (3, H, W), and composite's stats.
        covariances: this step's, when several views share them."""
        w, h = size
        cov = self.covariances() if covariances is None else covariances
        rot, eye = camera.rotation.to(self.means), camera.eye.to(self.means)
        cam = (self.means - eye) @ rot.T
        x, y, z = cam.unbind(-1)
        f = camera.focal
        if camera.orthographic:
            xy = torch.stack([w / 2 + f * x, h / 2 - f * y], dim=-1)
            jac = cam.new_tensor([[f, 0.0, 0.0], [0.0, -f, 0.0]]).expand(len(cam), 2, 3)
        else:
            zs = z.clamp_min(NEAR)
            xy = torch.stack([w / 2 + f * x / zs, h / 2 - f * y / zs], dim=-1)
            # As 3DGS: the Jacobian no further out than a little past the frustum.
            tx = (x / zs).clamp(-FRUSTUM_SLACK * w / (2 * f), FRUSTUM_SLACK * w / (2 * f)) * zs
            ty = (y / zs).clamp(-FRUSTUM_SLACK * h / (2 * f), FRUSTUM_SLACK * h / (2 * f)) * zs
            zero = torch.zeros_like(zs)
            jac = torch.stack([torch.stack([f / zs, zero, -f * tx / zs ** 2], -1),
                               torch.stack([zero, -f / zs, f * ty / zs ** 2], -1)], dim=1)
        t = jac @ rot
        s2 = t @ cov @ t.transpose(1, 2)
        conic, radius = conic_and_radius(s2[:, 0, 0] + LOW_PASS, s2[:, 0, 1], s2[:, 1, 1] + LOW_PASS)
        near = -math.inf if camera.orthographic else NEAR
        return composite(xy, conic, radius, z, self.color(), self.opacity(), size, background,
                         self.cfg["max_radius"], near)

    def values(self) -> dict:
        """Summary numbers, for the page and run.json."""
        with torch.no_grad():
            opacity, scale = self.opacity(), self.scale()
            return {"mean opacity": round(opacity.mean().item(), 4),
                    "opaque (> 0.5)": round((opacity > 0.5).float().mean().item(), 4),
                    "mean size": round(scale.mean().item(), 5),
                    "extent": round(self.means.norm(dim=-1).quantile(0.95).item(), 4)}


def view_camera(views: dict, i: int, size: tuple[int, int], jitter: float = 0.0) -> mesh_texture.Camera:
    """Camera i of an orbit around the unit sphere, its angles moved by up to +- jitter degrees."""
    azimuth, elevation = mesh_texture.view_angles(views, i)
    if jitter > 0:
        azimuth += jitter * (2 * torch.rand(1).item() - 1)
        elevation += jitter * (2 * torch.rand(1).item() - 1)
    return mesh_texture.orbit_camera(views, azimuth, elevation, (0.0, 0.0, 0.0), 1.0, size)


def project_points(points: torch.Tensor, camera: mesh_texture.Camera, size: tuple[int, int]) -> torch.Tensor:
    """(N, 2) grid_sample coordinates (-1..1 across the image) of world points (N, 3)."""
    xy, _ = mesh_texture.project(points, camera, size)
    w, h = size
    return torch.stack([2 * xy[:, 0] / w - 1, 2 * xy[:, 1] / h - 1], dim=-1)


def start_colors(start: torch.Tensor | None, camera: mesh_texture.Camera, size: tuple[int, int],
                 init: str, noise_std: float, monochrome: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """Splats' colors() per a CANVAS_INIT value: "start": the (1, C, H, W) start image where each
    centre falls in `camera`'s view (mid grey off it); "noise": as initial_image's; else mid grey."""
    channels = 1 if monochrome else 3

    def colors(means: torch.Tensor) -> torch.Tensor:
        n = len(means)
        if init == "noise":
            return torch.sigmoid(noise_std * torch.randn(n, channels))
        if init != "start" or start is None:
            return torch.full((n, channels), 0.5)
        grid = project_points(means, camera, size)
        image = start.to(means)
        image = image.mean(1, keepdim=True) if monochrome and image.shape[1] == 3 else image
        sampled = F.grid_sample(image, grid[None, :, None], align_corners=False, padding_mode="zeros")[0, :, :, 0].T
        inside = (grid.abs() <= 1).all(-1, keepdim=True)
        return torch.where(inside, sampled, torch.full_like(sampled, 0.5))
    return colors


# ── Preview ────────────────────────────────────────────────────────────────────

def preview(cfg: dict, size: tuple[int, int], background: Callable[[tuple[int, int]], torch.Tensor],
            start: torch.Tensor | None, init: str, noise_std: float, seed: int) -> torch.Tensor:
    """(1, 3, H, W) every view of the scene as a run starts, tiled on a canvas of size (w, h) (the
    canvas view outlined). background(size) -> (1, 3, h, w); start: the (1, C, H, W) start image,
    framed on the canvas. CPU."""
    cfg = normalize_config(cfg)
    views = cfg["views"]
    canvas = view_camera(views, views["canvas_view"], size)
    torch.manual_seed(seed)
    splats = Splats(cfg, "cpu", start_colors(start, canvas, size, init, noise_std, cfg["monochrome"]))
    return all_views(splats, size, background(size))


def all_views(splats: Splats, size: tuple[int, int], background: torch.Tensor) -> torch.Tensor:
    """(1, 3, H, W) every camera's unjittered view, tiled on a canvas of size (w, h) over `background`
    (1, 3, H, W) (each tile over it shrunk), the canvas view outlined."""
    views = splats.cfg["views"]
    count = views["count"]
    _, _, tile = mesh_texture.sheet_layout(count, size)
    with torch.no_grad():
        cov = splats.covariances()
        back = F.interpolate(background, size=(tile[1], tile[0]), mode="area")[0]
        tiles = torch.stack([splats.render(view_camera(views, i, tile), tile, back, cov)[0] for i in range(count)])
        return mesh_texture.tile_sheet(tiles, size, background, views["canvas_view"])


# ── Export ─────────────────────────────────────────────────────────────────────

GRID_BUDGET = 1 << 24       # (splat, voxel) candidates per batch while summing the splats on a grid
GRID_MAX_RADIUS = 48        # voxels: a splat's window half size is capped here
GRID_MIN_OPACITY = 0.02     # fainter splats don't count towards the isosurface's box


def density_grid(splats: Splats, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The splats summed on a voxel grid: density = sum of opacity x Gaussian, colour = their
    density-weighted mean. The grid is cubic voxels over the box of the splats' 3-sigma extents (the
    faint ones left out), `resolution` along its longest side. Each splat only within its own window,
    batched by window size, as composite. -> (density (D, H, W), colour (3, D, H, W) in [0, 1],
    lo (3,), hi (3,)), numpy."""
    with torch.no_grad():
        means, scale, opacity = splats.means.detach(), splats.scale(), splats.opacity()
        color = splats.color()
        keep = opacity > GRID_MIN_OPACITY
        if not keep.any():
            keep = torch.ones_like(keep)
        reach = 3 * scale.max(-1).values
        lo = (means - reach[:, None])[keep].min(0).values
        hi = (means + reach[:, None])[keep].max(0).values
        voxel = float((hi - lo).max()) / resolution
        dims = ((hi - lo) / voxel).ceil().long().clamp_min(1)          # (x, y, z) counts
        hi = lo + dims * voxel
        nx, ny, nz = dims.tolist()
        rot = rotation_matrices(splats.quats)
        inverse = rot @ torch.diag_embed(1 / scale ** 2) @ rot.transpose(1, 2)  # (N, 3, 3)
        center = (means - lo) / voxel - 0.5                            # in voxel index units
        window = (reach / voxel).ceil().long().clamp(1, GRID_MAX_RADIUS)
        bucket = torch.where(window > 1, 2 ** torch.ceil(torch.log2(window.float())).long(), window)
        bucket = bucket.clamp(max=GRID_MAX_RADIUS)
        total = torch.zeros(nz * ny * nx, device=means.device)
        weighted = torch.zeros(3, nz * ny * nx, device=means.device)
        base = center.round().long()
        for r in bucket[keep].unique().tolist():
            members = (keep & (bucket == r)).nonzero().squeeze(1)
            o = torch.arange(-r, r + 1, device=means.device)
            offsets = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).view(-1, 3)  # (K, 3) x, y, z
            per_batch = max(1, GRID_BUDGET // len(offsets))
            for batch in members.split(per_batch):
                idx = base[batch, None] + offsets[None]                      # (B, K, 3)
                d = (idx - center[batch, None]) * voxel                     # world offsets
                q = torch.einsum("bki,bij,bkj->bk", d, inverse[batch], d)
                inside = (idx >= 0).all(-1) & (idx[..., 0] < nx) & (idx[..., 1] < ny) & (idx[..., 2] < nz) & (q < 9)
                row, k = inside.nonzero(as_tuple=True)
                value = opacity[batch[row]] * torch.exp(-0.5 * q[row, k])
                cell = idx[row, k]
                flat = (cell[:, 2] * ny + cell[:, 1]) * nx + cell[:, 0]
                total.index_add_(0, flat, value)
                weighted.index_add_(1, flat, color[batch[row]].T * value)
        mean_color = weighted / total.clamp_min(1e-6)
        return (total.view(nz, ny, nx).cpu().numpy(), mean_color.view(3, nz, ny, nx).cpu().numpy(),
                lo.cpu().double().numpy(), hi.cpu().double().numpy())


def isosurface(splats: Splats, resolution: int, level: float):
    """The isosurface of the splats' summed density (density_grid) at `level`, coloured by their
    mean colour there (a volume_mesh.Isosurface)."""
    import volume_mesh
    density, color, lo, hi = density_grid(splats, resolution)
    surface = volume_mesh.marching_cubes(density, level, lo, hi)
    if len(surface.vertices):
        surface.colors = volume_mesh.sample_colors(color, surface.vertices, lo, hi)
    return surface

def write_ply(path: Path, splats: Splats) -> None:
    """The splats in the 3D Gaussian splatting .ply layout: x y z, nx ny nz (0), f_dc_0..2 (the
    colour as a 0th-order spherical harmonic), opacity (logit), scale_0..2 (log), rot_0..3 (w x y z)."""
    with torch.no_grad():
        means = splats.means.cpu().numpy()
        f_dc = ((splats.color() - 0.5) / SH_C0).cpu().numpy()
        opacity = splats.opacities.cpu().numpy()
        scale = torch.log(splats.scale()).cpu().numpy()
        rot = F.normalize(splats.quats, dim=-1).cpu().numpy()
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate([means, np.zeros_like(means), f_dc, opacity[:, None], scale, rot], axis=1).astype("<f4")
    header = "ply\nformat binary_little_endian 1.0\n" f"element vertex {len(data)}\n" \
             + "".join(f"property float {n}\n" for n in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def write_rhino(path: Path, splats: Splats, notes: dict[str, str] | None = None) -> None:
    """A .3dm with the splats' centres as a point cloud in their colours, on a layer of its own, and
    (mesh.resolution > 0) the isosurface of their summed density at mesh.level as a vertex-coloured
    mesh on another. notes: user strings on both."""
    import rhino3dm as rhino
    import volume_mesh
    with torch.no_grad():
        means = splats.means.cpu().double().numpy().tolist()
        colors = (splats.color() * 255).round().int().cpu().numpy().tolist()
    cloud = rhino.PointCloud()
    for (x, y, z), (r, g, b) in zip(means, colors):
        cloud.Add(rhino.Point3d(x, y, z), (r, g, b, 255))
    model = rhino.File3dm()
    attributes = rhino.ObjectAttributes()
    attributes.Name = "gaussians"
    attributes.LayerIndex = model.Layers.AddLayer("gaussians", (0, 0, 0, 255))
    for key, value in (notes or {}).items():
        attributes.SetUserString(key, str(value))
    model.Objects.AddPointCloud(cloud, attributes)
    mesh_cfg = splats.cfg["mesh"]
    if mesh_cfg["resolution"] > 0:
        surface = isosurface(splats, mesh_cfg["resolution"], mesh_cfg["level"])
        if len(surface):
            mesh_attributes = rhino.ObjectAttributes()
            mesh_attributes.Name = "isosurface"
            mesh_attributes.LayerIndex = model.Layers.AddLayer("isosurface", (128, 128, 128, 255))
            for key, value in {**(notes or {}), "level": mesh_cfg["level"]}.items():
                mesh_attributes.SetUserString(key, str(value))
            model.Objects.AddMesh(volume_mesh.to_rhino(surface), mesh_attributes)
    if not model.Write(str(path), 8):
        raise RuntimeError(f"Could not write {path}")
