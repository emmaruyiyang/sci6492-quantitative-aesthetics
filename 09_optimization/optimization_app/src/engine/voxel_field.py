"""Voxel volumes for optimization.py's VOXELS field, rendered by dvolume's volume renderer.

The scene is the cube -1..1 (world x, y, z; z up), sampled as R^3 voxels: a density in [0, 1] and,
per mode, a colour --

    color       an RGB colour volume
    monochrome  a gray volume
    density     no colour volume: the material colour (optimized with optimize_color)

-- each the sigmoid of a Laplacian pyramid of 3D tensors (R, R/2, R/4, ...: coarse levels for broad
changes, as the pixel fields' pyramids), all optimized. dvolume (shared.dvolume)
renders a view: its camera's frustum is sampled in `samples` depth layers
(jittered by depth_noise), the volume resampled there (grid_sample) and the layers alpha-composited
back to front, each alpha = density x the per-layer opacity -- or, with a contrast, dvolume's sigmoid
((density - 0.5) x contrast) x the opacity: sharper surfaces, at the density's 0.5. "lit" shading multiplies the colour by
dvolume's half-Lambert diffuse on the density's gradient.

Cameras are mesh_texture's orbit (Rhino's frame, azimuth 0 = the Front view), around the unit
sphere. The density starts as a ball, smooth noise, empty, or a .3dm's meshes (dvolume's signed
distance, the meshes fitted into the cube); the colour per CANVAS_INIT (the start image seen from
the canvas camera, through the volume).

Exports: the density and colour as dvolume .npz fields, and the density's isosurface (volume_mesh:
VTK's marching cubes) as a .3dm mesh coloured from the colour volume.
"""
import math
import warnings
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import mesh_texture
import volume_mesh
from surface_of_revolution import _choice, _clamped, _filled, _hex

MODES = ["color", "monochrome", "density"]
INITS = ["ball", "noise", "empty", "mesh"]
SHADINGS = ["lit", "flat"]
LIGHT_SPACES = mesh_texture.LIGHT_SPACES
OPACITY_LAYERS = 64       # the opacity is that of a layer 1/64 of the view's depth range thick
INIT_LOGIT = 6.0          # |logit| of a solid / empty voxel at the start
_DV = None


def dvolume():
    """The dvolume package (shared.dvolume), imported on first use (it pulls in VTK)."""
    global _DV
    if _DV is None:
        from shared import dvolume as dv
        _DV = dv
    return _DV


# ── Config ─────────────────────────────────────────────────────────────────────

def default_config() -> dict:
    return normalize_config({})


def normalize_render(cfg: dict | None) -> dict:
    """The volume rendering settings every dvolume-rendered field shares (VOXELS, and solid_fields'
    ASSEMBLY, SOLIDS_ON_OFF, EXTRUSIONS), defaults filled in and validated:
    {resolution (the scene's voxels along each side), samples (depth layers per view), opacity (of a
     dense layer 1/64 of the depth range thick), contrast (dvolume's density sigmoid factor in the view
     frustum: 0 = off, the density as it is), depth_noise (layer jitter, in layer spacings),
     render_scale (views rendered at this x the canvas, then upsampled), color, optimize_color (the
     material, where there is no colour volume), shading: {mode, space, azimuth, elevation}, jitter
     (degrees: the models' cameras), iso_level (the exported isosurface's density),
     views: mesh_texture.normalize_views}."""
    given = cfg or {}
    cfg = _filled({"resolution": 64, "samples": 64, "opacity": 0.3, "contrast": 0.0, "depth_noise": 0.5,
                   "render_scale": 1.0, "color": "#bbbbbb", "optimize_color": False, "jitter": 10.0,
                   "iso_level": 0.5}, given)
    shading = _filled({"mode": "lit", "space": "camera", "azimuth": -30.0, "elevation": 30.0}, given.get("shading"))
    return {"resolution": int(_clamped(cfg["resolution"], 8, 256)), "samples": int(_clamped(cfg["samples"], 4, 512)),
            "opacity": _clamped(cfg["opacity"], 0.001, 1.0), "contrast": _clamped(cfg["contrast"], 0.0, 1000.0),
            "depth_noise": _clamped(cfg["depth_noise"], 0.0, 4.0),
            "render_scale": _clamped(cfg["render_scale"], 0.1, 1.0),
            "color": _hex(cfg["color"], "#bbbbbb"), "optimize_color": bool(cfg["optimize_color"]),
            "shading": {"mode": _choice(shading["mode"], SHADINGS, "shading"),
                        "space": _choice(shading["space"], LIGHT_SPACES, "light space"),
                        "azimuth": float(shading["azimuth"]), "elevation": _clamped(shading["elevation"], -90.0, 90.0)},
            "jitter": _clamped(cfg["jitter"], 0.0, 180.0), "iso_level": _clamped(cfg["iso_level"], 0.01, 0.99),
            "views": mesh_texture.normalize_views(given.get("views"))}


def normalize_config(cfg: dict | None) -> dict:
    """A VOXELS config, defaults filled in and validated: normalize_render's settings, and
    {levels, lr_decay (pyramid, as the pixel fields'), mode, init, init_radius (ball: in the cube's
     half sides), init_mesh (a .3dm), smoothness (weight of the density's 3D total variation),
     blur: {every (steps; 0 = off), kernel (odd, voxels), strength (0..1), color (the colour volume
       too)}: dvolume's box blur of the finest pyramid levels between steps (Voxels.blur)}.
    The material (color, optimize_color) is density mode's."""
    render = normalize_render(cfg)
    cfg = _filled({"levels": 3, "lr_decay": 1.0, "mode": "color", "init": "ball", "init_radius": 0.5, "init_mesh": "",
                   "smoothness": 0.0}, cfg)
    blur = _filled({"every": 0, "kernel": 3, "strength": 0.5, "color": False}, cfg.get("blur"))
    kernel = int(_clamped(blur["kernel"], 3, 15))
    resolution = render["resolution"]
    return {**render, "levels": int(_clamped(cfg["levels"], 1, max(1, int(math.log2(resolution)) - 1))),
            "lr_decay": _clamped(cfg["lr_decay"], 0.05, 1.0), "mode": _choice(cfg["mode"], MODES, "voxel mode"),
            "init": _choice(cfg["init"], INITS, "voxel init"), "init_radius": _clamped(cfg["init_radius"], 0.01, 1.8),
            "init_mesh": str(cfg["init_mesh"]).strip(), "smoothness": _clamped(cfg["smoothness"], 0.0, 1e4),
            "blur": {"every": int(_clamped(blur["every"], 0, 100_000)), "kernel": kernel if kernel % 2 else kernel + 1,
                     "strength": _clamped(blur["strength"], 0.0, 1.0), "color": bool(blur["color"])}}


def schema() -> dict:
    return {"modes": MODES, "inits": INITS, "shadings": SHADINGS, "light_spaces": LIGHT_SPACES,
            "max_views": mesh_texture.MAX_VIEWS}


def channels(cfg: dict) -> int:
    """The colour volume's channels: 3, 1, or 0 (density mode: none)."""
    return {"color": 3, "monochrome": 1, "density": 0}[cfg["mode"]]


# ── Volumes ────────────────────────────────────────────────────────────────────

def voxel_centers(resolution: int, device) -> torch.Tensor:
    """(R, R, R, 3) world (x, y, z) of the voxel centres, indexed [z, y, x] (grid_sample's
    align_corners=False layout)."""
    c = (torch.arange(resolution, device=device, dtype=torch.float32) + 0.5) / resolution * 2 - 1
    z, y, x = torch.meshgrid(c, c, c, indexing="ij")
    return torch.stack([x, y, z], dim=-1)


_MESH_SDF: OrderedDict = OrderedDict()  # (path, mtime, resolution) -> (R, R, R) signed distance


def mesh_logits(path: str, resolution: int) -> torch.Tensor:
    """(R, R, R) density logits of a .3dm's meshes, fitted into the cube (dvolume's signed distance,
    unit box), steep over about two voxels. CPU, cached."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"Voxel init mesh not found: {path}")
    key = (str(p.resolve()), p.stat().st_mtime_ns, resolution)
    if key not in _MESH_SDF:
        dv = dvolume()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            field = dv.VolumeField.signedDistanceFromRhinoFile(str(p), torch.device("cpu"), resolution, unit_box=True,
                                                               use_cache=False, union=True)
        _MESH_SDF[key] = field.tensor[0, 0].float()
        while len(_MESH_SDF) > 2:
            _MESH_SDF.popitem(last=False)
    # dvolume samples at corners (-1..1 inclusive): close enough to the centres at this resolution.
    return (-_MESH_SDF[key] / (2 / resolution) * 2).clamp(-INIT_LOGIT, INIT_LOGIT)


def start_density(cfg: dict) -> torch.Tensor:
    """(R, R, R) the density's starting logits, per cfg's init. CPU; noise from torch's generator."""
    r = cfg["resolution"]
    if cfg["init"] == "empty":
        return torch.full((r, r, r), -INIT_LOGIT)
    if cfg["init"] == "mesh":
        return mesh_logits(cfg["init_mesh"], r)
    if cfg["init"] == "noise":
        coarse = torch.randn(1, 1, *(3 * [max(2, r // 8)]))
        smooth = F.interpolate(coarse, size=(r, r, r), mode="trilinear", align_corners=False)[0, 0]
        return (smooth * 2 * INIT_LOGIT / max(smooth.abs().max().item(), 1e-6) - 1.0).clamp(-INIT_LOGIT, INIT_LOGIT)
    distance = voxel_centers(r, "cpu").norm(dim=-1)
    return ((cfg["init_radius"] - distance) / (2 / r) * 2).clamp(-INIT_LOGIT, INIT_LOGIT)


def start_color(cfg: dict, start: torch.Tensor | None, camera, size, init: str, noise_std: float) -> torch.Tensor:
    """(C, R, R, R) the colour's start in [0, 1]: "start": the (1, C, H, W) start image where each voxel
    falls in `camera`'s view (a mesh_texture.Camera; mid grey off it); "noise": as initial_image's;
    else mid grey. CPU."""
    r, c = cfg["resolution"], channels(cfg)
    if init == "noise":
        return torch.sigmoid(noise_std * torch.randn(c, r, r, r))
    if init != "start" or start is None:
        return torch.full((c, r, r, r), 0.5)
    points = voxel_centers(r, "cpu").view(-1, 3)
    xy, _ = mesh_texture.project(points, camera, size)
    w, h = size
    grid = torch.stack([2 * xy[:, 0] / w - 1, 2 * xy[:, 1] / h - 1], dim=-1)
    image = start.float()
    image = image.mean(1, keepdim=True) if c == 1 and image.shape[1] == 3 else image
    sampled = F.grid_sample(image, grid[None, :, None], align_corners=False)[0, :, :, 0]  # (C, N)
    inside = (grid.abs() <= 1).all(-1)[None]
    return torch.where(inside, sampled, torch.full_like(sampled, 0.5)).view(c, r, r, r)


def _logit(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-3, 1 - 1e-3)
    return torch.log(x / (1 - x))


class Pyramid:
    """A (1, C, R, R, R) volume in [0, 1]: the sigmoid of the sum of `levels` tensors at R, R/2, ...,
    trilinearly upsampled. The finest starts as the given logits, the others at 0."""

    def __init__(self, logits: torch.Tensor, levels: int, device):
        r = logits.shape[-1]
        self.size = r
        self.levels = [logits.to(device).clone().requires_grad_(True)]
        for i in range(1, levels):
            n = max(2, r >> i)
            self.levels.append(torch.zeros(1, logits.shape[1], n, n, n, device=device, requires_grad=True))

    def param_groups(self, lr: float, decay: float) -> list[dict]:
        # As the pixel fields': the coarsest level at the full rate, each finer one x decay more.
        last = len(self.levels) - 1
        return [{"params": [level], "lr": lr * decay ** (last - i)} for i, level in enumerate(self.levels)]

    def __call__(self) -> torch.Tensor:
        total = self.levels[0]
        for level in self.levels[1:]:
            total = total + F.interpolate(level, size=(self.size,) * 3, mode="trilinear", align_corners=False)
        return torch.sigmoid(total)


class VolumeRenderer:
    """dvolume's model of a (1, 1, R, R, R) density and optional colour volume over the cube, and its
    views of it, per a config with normalize_render's settings."""

    def __init__(self, cfg: dict, device):
        self.cfg = cfg
        self.device = torch.device(device)
        dv = dvolume()
        r = cfg["resolution"]
        self.model = dv.VolumeModel([r, r, r], device=self.device)

    def set(self, density: torch.Tensor, color: torch.Tensor | None) -> None:
        """This step's density (1, 1, R, R, R) and colour (1, C, R, R, R) (1 channel: gray) or None."""
        dv = dvolume()
        if color is not None and color.shape[1] == 1:
            color = color.expand(-1, 3, -1, -1, -1)
        self.model.density = dv.VolumeField(density)
        self.model.color = dv.VolumeField(color) if color is not None else None
        self.model.invalidateDensity()

    def render(self, camera: mesh_texture.Camera, size: tuple[int, int], background: torch.Tensor,
               material: torch.Tensor, light: torch.Tensor | None) -> torch.Tensor:
        """(3, H, W) the model of the last update() seen by `camera` over `background` (3, H, W);
        material (3,): the colour without a colour volume (unused with one); light (3,): unit,
        towards the light, world (None: unlit)."""
        dv = dvolume()
        cfg = self.cfg
        w, h = size
        s = cfg["render_scale"]
        rw, rh = max(8, round(w * s)), max(8, round(h * s))
        eye = camera.eye.to(self.device)
        far = float(eye.norm()) + 2.0
        views = cfg["views"]
        if camera.orthographic:
            # The short side spans the unit sphere / fill; dvolume's half width is horizontal.
            half_short = 1.0 / views["fill"]
            half_width = half_short * (w / h if w >= h else 1.0)
            dv_camera = dv.VolumeCamera(eye, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 30.0, w / h, 0.01, far, self.device,
                                        orthographic=True, ortho_half_width=half_width)
        else:
            # mesh_texture's fov spans the short side; dvolume's the height.
            fov = views["fov"] if w >= h else math.degrees(2 * math.atan(math.tan(math.radians(views["fov"]) / 2) * h / w))
            dv_camera = dv.VolumeCamera(eye, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], fov, w / h, 0.01, far, self.device)
        viewport = dv.VolumeViewport(dv_camera, rw, rh, cfg["samples"], cfg["depth_noise"])
        # dvolume's normals: the density's negative gradient (outwards), dotted with the light.
        lighting = dv.VolLighting(light.to(self.device)) if light is not None else None
        if self.model.color is not None:
            material = torch.ones(3, device=self.device)  # the colour volume's colours as they are
        alpha =1 - (1 - cfg["opacity"]) ** (OPACITY_LAYERS / cfg["samples"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # dvolume's grid_sample calls leave align_corners unset
            gb = viewport.renderVolume(self.model, torch.zeros(3, rh, rw, device=self.device),
                                       volume_color=material, compositing_opacity=alpha, density_sigmoid_factor=cfg["contrast"],
                                       lighting=lighting, recompute_depth_bounds=True, require_alpha=True)
        color, coverage = gb.final_color[None], gb.alpha[None]  # colour premultiplied over black
        if (rw, rh) != (w, h):
            color = F.interpolate(color, size=(h, w), mode="bilinear", align_corners=False)
            coverage = F.interpolate(coverage, size=(h, w), mode="bilinear", align_corners=False)
        return (color + background[None] * (1 - coverage))[0].clamp(0, 1)


class Voxels:
    """The optimizable volumes of a (normalized) VOXELS config on `device`, and their renderer."""

    def __init__(self, cfg: dict, device, color_start: torch.Tensor | None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.density = Pyramid(start_density(cfg)[None, None], cfg["levels"], device)
        self.color = (Pyramid(_logit(color_start)[None], cfg["levels"], device)
                      if channels(cfg) and color_start is not None else None)
        self.renderer = VolumeRenderer(cfg, device)

    def param_groups(self, lr: float) -> list[dict]:
        decay = self.cfg["lr_decay"]
        return [*self.density.param_groups(lr, decay), *(self.color.param_groups(lr, decay) if self.color else [])]

    def update(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Evaluates the pyramids into the renderer's model for this step -> (density (1, 1, R, R, R),
        colour (1, 3, R, R, R) or None)."""
        density = self.density()
        color = self.color() if self.color is not None else None
        if color is not None and color.shape[1] == 1:
            color = color.expand(-1, 3, -1, -1, -1)
        self.renderer.set(density, color)
        return density, color

    def render(self, *args) -> torch.Tensor:
        """VolumeRenderer.render of the last update()."""
        return self.renderer.render(*args)

    def blur(self) -> None:
        """In place, outside autograd: the finest level of the density (and with blur.color, of the
        colour) moved towards dvolume's box blur of it (VolumeField.smooth), by blur.strength. The levels
        are logits, so they are padded by replication first, not by smooth's zeros (logit 0, density
        0.5, would grow walls on the cube's faces), and blurred a channel at a time (smooth's kernel sums
        the channels)."""
        dv = dvolume()
        cfg = self.cfg["blur"]
        k = cfg["kernel"]
        pad = k // 2
        levels = [self.density.levels[0]] + ([self.color.levels[0]] if cfg["color"] and self.color else [])
        with torch.no_grad():
            for level in levels:
                for c in range(level.shape[1]):
                    padded = F.pad(level[:, c:c + 1], (pad,) * 6, mode="replicate")
                    field = dv.VolumeField(padded)
                    field.smooth(k, cfg["strength"])
                    level[:, c:c + 1] = field.tensor[..., pad:-pad, pad:-pad, pad:-pad]

    def values(self) -> dict:
        with torch.no_grad():
            density = self.density()
            return {"mean density": round(density.mean().item(), 4),
                    "occupied (> iso)": round((density > self.cfg["iso_level"]).float().mean().item(), 4)}


def light_direction(cfg: dict, camera: mesh_texture.Camera) -> torch.Tensor | None:
    """(3,) unit, towards the light, world: the config's shading, relative to `camera` or the world.
    None when unlit."""
    shading = cfg["shading"]
    if shading["mode"] != "lit":
        return None
    a, e = math.radians(shading["azimuth"]), math.radians(shading["elevation"])
    if shading["space"] == "world":
        return torch.tensor([math.cos(e) * math.sin(a), -math.cos(e) * math.cos(a), math.sin(e)])
    right, up, forward = camera.rotation
    return F.normalize(math.cos(e) * (math.cos(a) * -forward + math.sin(a) * right) + math.sin(e) * up, dim=0)


def view_camera(cfg: dict, i: int, size: tuple[int, int], jitter: float = 0.0) -> mesh_texture.Camera:
    """Camera i of the orbit around the unit sphere, its angles moved by up to +- jitter degrees."""
    views = cfg["views"]
    azimuth, elevation = mesh_texture.view_angles(views, i)
    if jitter > 0:
        azimuth += jitter * (2 * torch.rand(1).item() - 1)
        elevation += jitter * (2 * torch.rand(1).item() - 1)
    return mesh_texture.orbit_camera(views, azimuth, elevation, (0.0, 0.0, 0.0), 1.0, size)


def total_variation_3d(volume: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference between neighbouring voxels, along the three axes of (..., D, H, W)."""
    return sum((volume.narrow(d, 1, volume.shape[d] - 1) - volume.narrow(d, 0, volume.shape[d] - 1)).abs().mean()
               for d in (-3, -2, -1))


def slices(density: torch.Tensor, color: torch.Tensor | None) -> torch.Tensor:
    """(1, 3, R, 3R) the volume's three central sections, side by side: front (x, z), right (y, z), top
    (x, y), each image up = z (top: y). The colour where there is one, faded by the density."""
    vol = (color if color is not None else density.expand(-1, 3, -1, -1, -1)) * density
    r = density.shape[-1]
    m = r // 2
    front = vol[0, :, :, m, :].flip(1)            # (3, z, x), z up
    right = vol[0, :, :, :, m].flip(1)            # (3, z, y)
    top = vol[0, :, m, :, :].flip(1)              # (3, y, x), y up
    return torch.cat([front, right, top], dim=-1)[None]


# ── Preview ────────────────────────────────────────────────────────────────────

def preview(cfg: dict, size: tuple[int, int], background: Callable[[tuple[int, int]], torch.Tensor],
            start: torch.Tensor | None, init: str, noise_std: float, seed: int) -> torch.Tensor:
    """(1, 3, H, W) every view as a run with this seed starts, tiled on a canvas of size (w, h) (the
    canvas view outlined). background(size) -> (1, 3, h, w); start: the (1, C, H, W) start image
    framed on the canvas. CPU."""
    cfg = normalize_config(cfg)
    views = cfg["views"]
    torch.manual_seed(seed)
    canvas = view_camera(cfg, views["canvas_view"], size)
    color = start_color(cfg, start, canvas, size, init, noise_std) if channels(cfg) else None
    voxels = Voxels(cfg, "cpu", color)
    material = torch.tensor([int(cfg["color"][i:i + 2], 16) / 255 for i in (1, 3, 5)])
    with torch.no_grad():
        voxels.update()
        return all_views(voxels, size, background(size), material)


def all_views(voxels, size: tuple[int, int], background: torch.Tensor, material: torch.Tensor) -> torch.Tensor:
    """(1, 3, H, W) every camera's unjittered view of what a VolumeRenderer (or Voxels) holds, tiled on a
    canvas of size (w, h) over `background` (1, 3, H, W) (each tile over it shrunk), the canvas view
    outlined."""
    cfg = voxels.cfg
    views = cfg["views"]
    count = views["count"]
    _, _, tile = mesh_texture.sheet_layout(count, size)
    with torch.no_grad():
        back = F.interpolate(background, size=(tile[1], tile[0]), mode="area")[0]
        tiles = []
        for i in range(count):
            camera = view_camera(cfg, i, tile)
            light = light_direction(cfg, camera)
            tiles.append(voxels.render(camera, tile, back, material, light.to(back.device) if light is not None else None))
        return mesh_texture.tile_sheet(torch.stack(tiles), size, background, views["canvas_view"])


# ── Export ─────────────────────────────────────────────────────────────────────

def save_npz(path: Path, volume: torch.Tensor, type_hint: str) -> None:
    """A (1, C, R, R, R) volume as a dvolume field file (VolumeField.save: float16 'data' + metadata)."""
    dv = dvolume()
    field = dv.VolumeField(volume.detach().float().cpu(), type_hint=dv.VolumeFieldTypeHint(type_hint))
    field.save(str(path))


def isosurface(density: torch.Tensor, color: torch.Tensor | None, level: float,
               material: tuple[int, int, int]) -> volume_mesh.Isosurface:
    """The density's isosurface at `level` over the cube, coloured from the colour volume (else the
    material colour)."""
    values = density[0, 0].detach().float().cpu().numpy()
    surface = volume_mesh.marching_cubes(values, level, (-1, -1, -1), (1, 1, 1))
    if color is not None:
        surface.colors = volume_mesh.sample_colors(color[0].detach().float().cpu().numpy(), surface.vertices,
                                                   (-1, -1, -1), (1, 1, 1))
    else:
        surface.colors = np.tile(np.array(material, dtype=np.uint8), (len(surface.vertices), 1))
    return surface


def write_rhino(path: Path, surface: volume_mesh.Isosurface, layer: str, color: tuple[int, int, int],
                notes: dict | None = None, extra=None) -> None:
    """A .3dm with the isosurface as a vertex-coloured mesh on `layer`; extra(model) may add more."""
    import rhino3dm as rhino
    model = rhino.File3dm()
    attributes = rhino.ObjectAttributes()
    attributes.Name = layer
    attributes.LayerIndex = model.Layers.AddLayer(layer, (*color, 255))
    for key, value in (notes or {}).items():
        attributes.SetUserString(key, str(value))
    if len(surface):
        model.Objects.AddMesh(volume_mesh.to_rhino(surface), attributes)
    if extra is not None:
        extra(model)
    if not model.Write(str(path), 8):
        raise RuntimeError(f"Could not write {path}")
