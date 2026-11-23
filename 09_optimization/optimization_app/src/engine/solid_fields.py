"""Solid fields for optimization.py, composited into a voxel density and rendered by dvolume (through
voxel_field.VolumeRenderer, with voxel_field.normalize_render's settings), in the cube -1..1:

  ASSEMBLY        groups of solids, each a .3dm (its meshes) or a .npz / .npy volume and a count of
                  clones; every clone's 3D affine transformation is optimized (translation, rotation
                  about x, y, z, and scale, built with dvolume's Transform3d), and the clones are
                  composited by sum or by max density. A shape is centred in its own bounding cube
                  first (dvolume's unit box): its transformations act on that, not on where it sits in
                  its file.
  SOLIDS_ON_OFF   every mesh of one .3dm, voxelized in one cube that keeps their relative positions;
                  one density multiplier per mesh is optimized, pushed (with weights) towards on / off
                  and towards a share of their volume being on.
  EXTRUSIONS      closed-curve patterns (as SHAPES') on the XY, YZ and ZX planes, each shape extruded
                  through the cube along the plane's normal; one multiplier per shape, in -1..1 (or
                  0..1): a negative one subtracts. The planes' weighted masks are summed, then clamped
                  to 0..1.

Meshes are voxelized by dvolume (VolumeField.fromFile: a signed distance per mesh, sampled by VTK,
cached as an .npz beside the .3dm), and a signed distance becomes a density through a sigmoid
`edge` voxels wide. dvolume samples its grid at the corners of the cube (-1..1 inclusive): its
volumes are resampled with align_corners=True.

Each model's volume() -> (density (1, 1, R, R, R), colour (1, 3, R, R, R) or None), in the graph of
its parameters; export(folder, ...) writes its .3dm (meshes where it has them, and the density's
isosurface) and .json.
"""
import json
import math
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import volume_mesh
import voxel_field
from surface_of_revolution import _choice, _clamped, _filled, _hex

COMPOSITES = ["max", "add"]
PLANES = ["xy", "yz", "zx"]
VOLUME_EXTENSIONS = (".3dm", ".npz", ".npy")
CLONE_BATCH = 16          # clones resampled at once (memory: this x R^3 x 4 floats)
MIN_SCALE, MAX_SCALE = 0.02, 4.0


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


# ── Config ─────────────────────────────────────────────────────────────────────

GROUP_DEFAULT = {"path": "", "count": 4, "color": "#c08850", "scale": 0.3}


def normalize_assembly(cfg: dict | None) -> dict:
    """An ASSEMBLY config: normalize_render's settings and
    {groups: [{path (.3dm / .npz / .npy), count, color, scale (starting, of the scene's half side)}],
     composite ("max" | "add"), shape_resolution (each shape's own voxels per side), edge (voxels of
     the shape's grid: the density's sigmoid width), spread (the clones start uniformly in +- this),
     random_rotation, uniform_scale, optimize_position, optimize_rotation, optimize_scale,
     transform_lr (x the learning rate)}."""
    render = voxel_field.normalize_render(cfg)
    cfg = _filled({"composite": "max", "shape_resolution": 64, "edge": 1.0, "spread": 0.5, "random_rotation": True,
                   "uniform_scale": True, "optimize_position": True, "optimize_rotation": True, "optimize_scale": True,
                   "transform_lr": 1.0}, cfg)
    groups = []
    for given in cfg.get("groups") or [{}]:
        g = _filled(GROUP_DEFAULT, given)
        groups.append({"path": str(g["path"]).strip(), "count": int(_clamped(g["count"], 0, 256)),
                       "color": _hex(g["color"], GROUP_DEFAULT["color"]), "scale": _clamped(g["scale"], MIN_SCALE, MAX_SCALE)})
    return {**render, "groups": groups, "composite": _choice(cfg["composite"], COMPOSITES, "composite"),
            "shape_resolution": int(_clamped(cfg["shape_resolution"], 8, 256)), "edge": _clamped(cfg["edge"], 0.1, 16.0),
            "spread": _clamped(cfg["spread"], 0.0, 1.0), "random_rotation": bool(cfg["random_rotation"]),
            "uniform_scale": bool(cfg["uniform_scale"]), "optimize_position": bool(cfg["optimize_position"]),
            "optimize_rotation": bool(cfg["optimize_rotation"]), "optimize_scale": bool(cfg["optimize_scale"]),
            "transform_lr": _clamped(cfg["transform_lr"], 0.0, 1000.0)}


def normalize_on_off(cfg: dict | None) -> dict:
    """A SOLIDS_ON_OFF config: normalize_render's settings and
    {path (a .3dm), composite ("max" | "add"), edge (voxels), init (every multiplier's start, 0..1),
     init_noise (std in logit space), harden_weight (x mean 4 m (1 - m), ramped in over the run),
     coverage (% of the parts' volume to have on; None = free), coverage_weight}."""
    render = voxel_field.normalize_render(cfg)
    cfg = _filled({"path": "", "composite": "max", "edge": 1.0, "init": 0.5, "init_noise": 0.5,
                   "harden_weight": 1.0, "coverage_weight": 10.0}, cfg)
    coverage = (cfg or {}).get("coverage")
    return {**render, "path": str(cfg["path"]).strip(), "composite": _choice(cfg["composite"], COMPOSITES, "composite"),
            "edge": _clamped(cfg["edge"], 0.1, 16.0), "init": _clamped(cfg["init"], 0.01, 0.99),
            "init_noise": _clamped(cfg["init_noise"], 0.0, 10.0), "harden_weight": _clamped(cfg["harden_weight"], 0.0, 1e4),
            "coverage": None if coverage in (None, "") else _clamped(coverage, 0.0, 100.0),
            "coverage_weight": _clamped(cfg["coverage_weight"], 0.0, 1e4)}


def normalize_extrusions(cfg: dict | None) -> dict:
    """An EXTRUSIONS config: normalize_render's settings and
    {patterns: {xy, yz, zx} (.3dm paths of closed curves; empty = the XY pattern's), planes: {xy, yz, zx}
     (which are used), border (voxels around a pattern), init (every multiplier's start), init_noise
     (std, in the multipliers' raw space), negative (multipliers in -1..1, else 0..1)}."""
    render = voxel_field.normalize_render(cfg)
    given = cfg or {}
    cfg = _filled({"border": 0, "init": 0.3, "init_noise": 0.1, "negative": True}, given)
    patterns = _filled({p: "" for p in PLANES}, given.get("patterns"))
    planes = {p: True for p in PLANES} | {p: bool(v) for p, v in (given.get("planes") or {}).items() if p in PLANES}
    return {**render, "patterns": {p: str(patterns[p]).strip() for p in PLANES}, "planes": planes,
            "border": int(_clamped(cfg["border"], 0, 64)), "init": _clamped(cfg["init"], -0.99, 0.99),
            "init_noise": _clamped(cfg["init_noise"], 0.0, 10.0), "negative": bool(cfg["negative"])}


NORMALIZERS = {"ASSEMBLY": normalize_assembly, "SOLIDS_ON_OFF": normalize_on_off, "EXTRUSIONS": normalize_extrusions}


def schema() -> dict:
    return {"composites": COMPOSITES, "planes": PLANES, "group_default": GROUP_DEFAULT,
            "volume_extensions": list(VOLUME_EXTENSIONS)}


def validate(kind: str, cfg: dict) -> None:
    """ValueError naming a missing file or an empty setup of a normalized config."""
    if kind == "ASSEMBLY":
        groups = [g for g in cfg["groups"] if g["count"] > 0]
        if not groups:
            raise ValueError("Assembly: add a group with a count above 0")
        for g in groups:
            if Path(g["path"]).suffix.lower() not in VOLUME_EXTENSIONS or not Path(g["path"]).is_file():
                raise ValueError(f"Assembly: not a volume file ({', '.join(VOLUME_EXTENSIONS)}): {g['path'] or '(none)'}")
    elif kind == "SOLIDS_ON_OFF":
        if not Path(cfg["path"]).is_file():
            raise ValueError(f"Solids on/off: .3dm not found: {cfg['path'] or '(none)'}")
    elif kind == "EXTRUSIONS":
        used = [p for p in PLANES if cfg["planes"][p]]
        if not used:
            raise ValueError("Extrusions: turn on at least one plane")
        for p in used:
            path = cfg["patterns"][p] or cfg["patterns"]["xy"]
            if not Path(path).is_file():
                raise ValueError(f"Extrusions: {p.upper()} pattern not found: {path or '(none)'}")


# ── Volumes from files ─────────────────────────────────────────────────────────

class SourceMeshes:
    """The Mesh objects of a .3dm in file order (what dvolume voxelizes), and the cube dvolume fits
    them into: its centre and half side (unit box: local = (world - center) / half)."""

    def __init__(self, path: str):
        import rhino3dm as rhino
        model = rhino.File3dm.Read(str(path))
        if model is None:
            raise ValueError(f"Could not read {path}")
        self.meshes: list[tuple[np.ndarray, list[tuple]]] = []  # (vertices (V, 3), faces (a, b, c, d))
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for obj in model.Objects:
            geometry = obj.Geometry
            if not isinstance(geometry, rhino.Mesh):
                continue
            vertices = np.array([(v.X, v.Y, v.Z) for v in (geometry.Vertices[i] for i in range(len(geometry.Vertices)))],
                                dtype=np.float64)
            faces = [tuple(geometry.Faces[i]) for i in range(len(geometry.Faces))]
            self.meshes.append((vertices, faces))
            box = geometry.GetBoundingBox()
            lo = np.minimum(lo, [box.Min.X, box.Min.Y, box.Min.Z])
            hi = np.maximum(hi, [box.Max.X, box.Max.Y, box.Max.Z])
        if not self.meshes:
            raise ValueError(f"No meshes in {Path(path).name} (Breps and extrusions: _Mesh them first)")
        # As dvolume.VolumeField.signedDistanceFromRhinoFile: a cube around the bounding box's centre.
        self.center = (lo + hi) / 2
        self.half = float((hi - lo).max()) / 2 + 0.01

    def to_unit(self, vertices: np.ndarray) -> np.ndarray:
        return (vertices - self.center) / self.half

    def to_world(self, local: np.ndarray) -> np.ndarray:
        return local * self.half + self.center


def rhino_mesh(vertices: np.ndarray, faces: list[tuple], color: tuple[int, int, int] | None = None):
    import rhino3dm as rhino
    mesh = rhino.Mesh()
    for x, y, z in vertices.tolist():
        mesh.Vertices.Add(x, y, z)
    for a, b, c, d in faces:
        if c == d:
            mesh.Faces.AddFace(a, b, c)
        else:
            mesh.Faces.AddFace(a, b, c, d)
    mesh.Normals.ComputeNormals()
    return mesh


_VOLUMES: OrderedDict = OrderedDict()  # (path, mtime, resolution, union) -> (B, 1, n, n, n) sdf / density


def load_volume(path: str, resolution: int, union: bool, edge: float) -> torch.Tensor:
    """(B, 1, n, n, n) densities in [0, 1] of a volume file, CPU: a .3dm's meshes voxelized by dvolume in
    their unit box (union: all in one, B = 1; else one per mesh, in the same cube), its signed distance
    through a sigmoid `edge` voxels wide; a .npz / .npy as dvolume reads it (a signed distance turned
    to a density the same way, a density as it is). dvolume caches a .3dm's voxelization beside it."""
    p = Path(path)
    key = (str(p.resolve()), p.stat().st_mtime_ns, resolution, union)
    if key not in _VOLUMES:
        dv = voxel_field.dvolume()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            field = dv.VolumeField.fromFile(str(p), torch.device("cpu"), resolution, unit_box=True, use_cache=True,
                                            union=union)
        sdf = field.type_hint == dv.VolumeFieldTypeHint.SDF
        _VOLUMES[key] = (field.tensor.float()[:, :1], sdf)
        while len(_VOLUMES) > 6:
            _VOLUMES.popitem(last=False)
    values, sdf = _VOLUMES[key]
    if not sdf:
        return values.clamp(0, 1)
    n = values.shape[-1]
    return torch.sigmoid(-values / (edge * 2 / max(n - 1, 1)))


def centers_grid(resolution: int, device) -> torch.Tensor:
    """(1, R, R, R, 3) grid_sample coordinates (x, y, z) of the scene's voxel centres."""
    return voxel_field.voxel_centers(resolution, device)[None]


def resample(volume: torch.Tensor, resolution: int, batch: int = 8) -> torch.Tensor:
    """(B, C, R, R, R): a corner-sampled volume (dvolume's grids: -1..1 inclusive) at the scene's voxel
    centres -- even at the same resolution, the two grids are half a voxel apart at the faces."""
    grid = centers_grid(resolution, volume.device)
    return torch.cat([F.grid_sample(part, grid.expand(len(part), -1, -1, -1, -1), mode="bilinear",
                                    padding_mode="border", align_corners=True) for part in volume.split(batch)])


def composite(stack: torch.Tensor, mode: str) -> torch.Tensor:
    """(1, 1, R, R, R) from (N, 1, R, R, R) densities: their max, or their sum clamped to 1."""
    return stack.amax(0, keepdim=True) if mode == "max" else stack.sum(0, keepdim=True).clamp(max=1.0)


def _logit(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-3, 1 - 1e-3)
    return torch.log(x / (1 - x))


# ── ASSEMBLY ───────────────────────────────────────────────────────────────────

class Assembly:
    """Clones of the config's groups, each placed by an optimized affine transformation (shape's unit
    box -> scene): M = T(position) @ Rz @ Ry @ Rx (rotation, radians) @ S(scale), built with dvolume's
    Transform3d. A clone's density at a scene voxel is its shape's at M^-1 (voxel), by affine_grid /
    grid_sample. Parameters: position (N, 3), rotation (N, 3), log scale (N, 1 | 3)."""

    def __init__(self, cfg: dict, device):
        self.cfg = cfg
        self.device = torch.device(device)
        self.groups = [g for g in cfg["groups"] if g["count"] > 0]
        self.shapes, self.sources = [], []
        for g in self.groups:
            self.shapes.append(load_volume(g["path"], cfg["shape_resolution"], True, cfg["edge"]).to(self.device))
            self.sources.append(SourceMeshes(g["path"]) if Path(g["path"]).suffix.lower() == ".3dm" else None)
        self.group_of = torch.cat([torch.full((g["count"],), i) for i, g in enumerate(self.groups)])
        n = len(self.group_of)
        spread = cfg["spread"]
        position = (2 * torch.rand(n, 3) - 1) * spread
        rotation = (2 * torch.rand(n, 3) - 1) * math.pi if cfg["random_rotation"] else torch.zeros(n, 3)
        scales = torch.tensor([g["scale"] for g in self.groups])[self.group_of]
        log_scale = torch.log(scales)[:, None].expand(-1, 1 if cfg["uniform_scale"] else 3).clone()
        self.position = position.to(self.device).requires_grad_(cfg["optimize_position"])
        self.rotation = rotation.to(self.device).requires_grad_(cfg["optimize_rotation"])
        self.log_scale = log_scale.to(self.device).requires_grad_(cfg["optimize_scale"])
        self.colors = torch.tensor([[c / 255 for c in _rgb(g["color"])] for g in self.groups],
                                   device=self.device)[self.group_of.to(self.device)]  # (N, 3)

    def param_groups(self, lr: float) -> list[dict]:
        tensors = [t for t in (self.position, self.rotation, self.log_scale) if t.requires_grad]
        return [{"params": tensors, "lr": lr * self.cfg["transform_lr"]}] if tensors else []

    def scales(self) -> torch.Tensor:
        """(N, 3)"""
        return self.log_scale.exp().clamp(MIN_SCALE, MAX_SCALE).expand(-1, 3)

    def matrices(self) -> torch.Tensor:
        """(N, 4, 4) every clone's shape-to-scene transformation (dvolume's Transform3d)."""
        dv = voxel_field.dvolume()
        T = dv.Transform3d
        scales = self.scales()
        out = []
        for i in range(len(self.position)):
            r = self.rotation[i]
            m = (T.makeTranslation(self.position[i]) @ T.makeRotationZ(r[2]) @ T.makeRotationY(r[1])
                 @ T.makeRotationX(r[0]) @ T.makeScale(scales[i]))
            out.append(m.tensor)
        return torch.stack(out)

    def volume(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(density (1, 1, R, R, R), colour (1, 3, R, R, R): each clone's group colour, weighted by its
        density)."""
        r = self.cfg["resolution"]
        theta = torch.linalg.inv(self.matrices())[:, :3]   # scene -> shape unit box
        density, weight, color = None, None, None
        for g, shape in enumerate(self.shapes):
            clones = (self.group_of == g).nonzero().squeeze(1).to(self.device)
            for batch in clones.split(CLONE_BATCH):
                grid = F.affine_grid(theta[batch], [len(batch), 1, r, r, r], align_corners=False)
                d = F.grid_sample(shape.expand(len(batch), -1, -1, -1, -1), grid, mode="bilinear",
                                  padding_mode="zeros", align_corners=True)                # (b, 1, R, R, R)
                part = composite(d, self.cfg["composite"]) if self.cfg["composite"] == "max" else d.sum(0, keepdim=True)
                tint = (d * self.colors[batch].view(-1, 3, 1, 1, 1)).sum(0, keepdim=True)
                if density is None:
                    density, weight, color = part, d.sum(0, keepdim=True), tint
                else:
                    density = torch.maximum(density, part) if self.cfg["composite"] == "max" else density + part
                    weight, color = weight + d.sum(0, keepdim=True), color + tint
        if self.cfg["composite"] == "add":
            density = density.clamp(max=1.0)
        return density, color / weight.clamp_min(1e-4)

    def values(self) -> dict:
        with torch.no_grad():
            scales = self.scales()
            return {"clones": len(self.position), "mean scale": round(scales.mean().item(), 4),
                    "mean |position|": round(self.position.norm(dim=-1).mean().item(), 4)}

    def export(self, folder: Path, density: torch.Tensor, color: torch.Tensor) -> list[str]:
        """assembly.3dm: every clone's meshes (of a .3dm group) transformed into the scene (cube -1..1),
        a layer per group in its colour, and the density's isosurface on "contour"; assembly.json: the
        groups and every clone's 4 x 4 matrix (shape unit box -> scene)."""
        import rhino3dm as rhino
        with torch.no_grad():
            matrices = self.matrices().cpu().double().numpy()
            surface = voxel_field.isosurface(density, color, self.cfg["iso_level"], (180, 180, 180))
        model = rhino.File3dm()
        for g, group in enumerate(self.groups):
            source = self.sources[g]
            if source is None:
                continue
            color = _rgb(group["color"])
            layer = model.Layers.AddLayer(f"{g + 1} {Path(group['path']).stem}", (*color, 255))
            for k, i in enumerate((self.group_of == g).nonzero().squeeze(1).tolist()):
                m = matrices[i]
                for j, (vertices, faces) in enumerate(source.meshes):
                    local = source.to_unit(vertices)
                    placed = local @ m[:3, :3].T + m[:3, 3]
                    attributes = rhino.ObjectAttributes()
                    attributes.LayerIndex = layer
                    attributes.Name = f"{Path(group['path']).stem} {k + 1}" + (f".{j + 1}" if len(source.meshes) > 1 else "")
                    attributes.SetUserString("matrix", json.dumps(np.round(m, 6).tolist()))
                    model.Objects.AddMesh(rhino_mesh(placed, faces), attributes)
        add_contour(model, surface)
        model.Write(str(folder / "assembly.3dm"), 8)
        (folder / "assembly.json").write_text(json.dumps({
            "groups": self.groups, "units": "scene: the cube -1..1; a shape's own unit box is its meshes' "
                                            "bounding cube, centred",
            "clones": [{"group": int(self.group_of[i]), "matrix": np.round(matrices[i], 6).tolist()}
                       for i in range(len(matrices))]}, indent=1))
        return ["assembly.3dm", "assembly.json"]


def add_contour(model, surface: volume_mesh.Isosurface, to_world=None) -> None:
    """The isosurface as a vertex-coloured mesh on a layer "contour" of a rhino3dm File3dm (to_world:
    maps its vertices, e.g. back to model units)."""
    import rhino3dm as rhino
    if not len(surface):
        return
    if to_world is not None:
        surface = volume_mesh.Isosurface(to_world(surface.vertices), surface.normals, surface.triangles, surface.colors)
    attributes = rhino.ObjectAttributes()
    attributes.Name = "contour"
    attributes.LayerIndex = model.Layers.AddLayer("contour", (128, 128, 128, 255))
    model.Objects.AddMesh(volume_mesh.to_rhino(surface), attributes)


# ── SOLIDS_ON_OFF ──────────────────────────────────────────────────────────────

class OnOff:
    """Every mesh of the config's .3dm as a density in one shared cube, times its multiplier
    m = sigmoid(logit) (optimized), composited by max or sum."""

    def __init__(self, cfg: dict, device):
        self.cfg = cfg
        self.device = torch.device(device)
        r = cfg["resolution"]
        self.parts = resample(load_volume(cfg["path"], r, False, cfg["edge"]), r).to(self.device)  # (B, 1, R, R, R)
        self.source = SourceMeshes(cfg["path"])
        b = len(self.parts)
        self.volumes = self.parts.sum((1, 2, 3, 4))                                   # (B,)
        start = _logit(torch.tensor(cfg["init"])) + cfg["init_noise"] * torch.randn(b)
        self.logits = start.to(self.device).requires_grad_(True)
        self.progress = 0.0  # the run's, 0..1: the hardening's ramp

    def param_groups(self, lr: float) -> list[dict]:
        return [{"params": [self.logits], "lr": lr}]

    def multipliers(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def volume(self) -> tuple[torch.Tensor, None]:
        stack = self.parts * self.multipliers().view(-1, 1, 1, 1, 1)
        return composite(stack, self.cfg["composite"]), None

    def penalty(self) -> tuple[torch.Tensor, dict[str, float]]:
        """harden_weight x progress x mean 4 m (1 - m) (0 = all on or off); coverage_weight x (share of
        the parts' volume on - coverage)^2."""
        m = self.multipliers()
        loss = m.new_zeros(())
        hardness = (4 * m * (1 - m)).mean()
        covered = (m * self.volumes).sum() / self.volumes.sum().clamp_min(1e-6)
        stats = {"on": (m > 0.5).float().mean().item(), "undecided": hardness.item(), "coverage %": 100 * covered.item()}
        if self.cfg["harden_weight"]:
            loss = loss + self.cfg["harden_weight"] * self.progress * hardness
        if self.cfg["coverage"] is not None and self.cfg["coverage_weight"]:
            loss = loss + self.cfg["coverage_weight"] * (covered - self.cfg["coverage"] / 100) ** 2
        return loss, stats

    def values(self) -> dict:
        with torch.no_grad():
            m = self.multipliers()
            return {"parts": len(m), "on (> 0.5)": int((m > 0.5).sum()), "mean multiplier": round(m.mean().item(), 4)}

    def export(self, folder: Path, density: torch.Tensor, color) -> list[str]:
        """on_off.3dm in the file's own units: the meshes on layers "on" / "off" (multiplier > / <= 0.5),
        each with its multiplier, and the density's isosurface on "contour"; on_off.json: the
        multipliers in file order."""
        import rhino3dm as rhino
        with torch.no_grad():
            m = self.multipliers().cpu().tolist()
            surface = voxel_field.isosurface(density, None, self.cfg["iso_level"], _rgb(self.cfg["color"]))
        model = rhino.File3dm()
        on = model.Layers.AddLayer("on", (*_rgb(self.cfg["color"]), 255))
        off = model.Layers.AddLayer("off", (220, 220, 220, 255))
        for i, (vertices, faces) in enumerate(self.source.meshes):
            attributes = rhino.ObjectAttributes()
            attributes.LayerIndex = on if m[i] > 0.5 else off
            attributes.Name = f"part {i + 1}"
            attributes.SetUserString("multiplier", f"{m[i]:.4f}")
            model.Objects.AddMesh(rhino_mesh(vertices, faces), attributes)
        add_contour(model, surface, self.source.to_world)
        model.Write(str(folder / "on_off.3dm"), 8)
        (folder / "on_off.json").write_text(json.dumps({"source": self.cfg["path"], "multipliers": m}, indent=1))
        return ["on_off.3dm", "on_off.json"]


# ── EXTRUSIONS ─────────────────────────────────────────────────────────────────

def _pattern_masks(path: str, resolution: int, border: int) -> torch.Tensor:
    """(S, R, R) the closed curves of a .3dm rasterized as SHAPES' (image rows: top = the pattern's +y)."""
    from pattern import Pattern
    pattern = Pattern.from_rhino(path, img_size=(resolution, resolution), img_border=border)
    return torch.from_numpy(np.stack(pattern.masks))


def extrude(plane: str, image: torch.Tensor) -> torch.Tensor:
    """(R, R, R) [z, y, x]: an (R, R) plane image (rows top-down = its vertical axis up) extruded
    through the cube along the plane's normal. xy: across x, y (from above); yz: across y, z (as Rhino's
    Right view); zx: across x, z (as its Front view)."""
    up = image.flip(0)  # rows bottom-up: index = the plane's vertical coordinate
    r = up.shape[0]
    if plane == "xy":
        return up[None].expand(r, r, r)            # [z, y, x] = up[y, x]
    if plane == "yz":
        return up[:, :, None].expand(r, r, r)      # [z, y, x] = up[z, y]
    return up[:, None, :].expand(r, r, r)          # zx: [z, y, x] = up[z, x]


class Extrusions:
    """Each enabled plane's pattern, its shapes' masks x their multipliers (tanh: -1..1, or sigmoid:
    0..1, of the optimized raw values), summed per plane and extruded through the cube; the planes
    summed, clamped to 0..1."""

    def __init__(self, cfg: dict, device):
        self.cfg = cfg
        self.device = torch.device(device)
        r = cfg["resolution"]
        self.planes = [p for p in PLANES if cfg["planes"][p]]
        self.paths = {p: cfg["patterns"][p] or cfg["patterns"]["xy"] for p in self.planes}
        self.masks, self.raw = {}, {}
        for p in self.planes:
            masks = _pattern_masks(self.paths[p], r, cfg["border"]).to(self.device)
            self.masks[p] = masks
            start = torch.full((len(masks),), float(cfg["init"]))
            if not cfg["negative"]:
                start = start.clamp(0.01, 0.99)
            raw = torch.atanh(start.clamp(-0.99, 0.99)) if cfg["negative"] else _logit(start)
            self.raw[p] = (raw + cfg["init_noise"] * torch.randn(len(masks))).to(self.device).requires_grad_(True)

    def param_groups(self, lr: float) -> list[dict]:
        return [{"params": list(self.raw.values()), "lr": lr}]

    def multipliers(self, plane: str) -> torch.Tensor:
        raw = self.raw[plane]
        return torch.tanh(raw) if self.cfg["negative"] else torch.sigmoid(raw)

    def volume(self) -> tuple[torch.Tensor, None]:
        total = 0
        for p in self.planes:
            image = (self.masks[p] * self.multipliers(p).view(-1, 1, 1)).sum(0)   # (R, R)
            total = total + extrude(p, image)
        return total.clamp(0, 1)[None, None], None

    def values(self) -> dict:
        with torch.no_grad():
            out = {}
            for p in self.planes:
                w = self.multipliers(p)
                out[f"{p} shapes"] = len(w)
                out[f"{p} negative"] = int((w < 0).sum())
                out[f"{p} mean"] = round(w.mean().item(), 4)
            return out

    def export(self, folder: Path, density: torch.Tensor, color) -> list[str]:
        """extrusions.3dm: the density's isosurface on "contour"; extrusions.json: every plane's
        pattern and multipliers (in the pattern's curve order)."""
        import rhino3dm as rhino
        with torch.no_grad():
            surface = voxel_field.isosurface(density, None, self.cfg["iso_level"], _rgb(self.cfg["color"]))
            weights = {p: [round(v, 5) for v in self.multipliers(p).cpu().tolist()] for p in self.planes}
        model = rhino.File3dm()
        add_contour(model, surface)
        model.Write(str(folder / "extrusions.3dm"), 8)
        (folder / "extrusions.json").write_text(json.dumps(
            {"patterns": self.paths, "multipliers": weights, "units": "scene: the cube -1..1"}, indent=1))
        return ["extrusions.3dm", "extrusions.json"]


MODELS = {"ASSEMBLY": Assembly, "SOLIDS_ON_OFF": OnOff, "EXTRUSIONS": Extrusions}


def preview(kind: str, cfg: dict, size: tuple[int, int], background, seed: int) -> torch.Tensor:
    """(1, 3, H, W) every camera's view of a model as a run with this seed starts, tiled on a canvas of
    size (w, h) (the canvas view outlined). background(size) -> (1, 3, h, w). CPU."""
    cfg = NORMALIZERS[kind](cfg)
    validate(kind, cfg)
    torch.manual_seed(seed)
    model = MODELS[kind](cfg, "cpu")
    renderer = voxel_field.VolumeRenderer(cfg, "cpu")
    material = torch.tensor([c / 255 for c in _rgb(cfg["color"])])
    with torch.no_grad():
        renderer.set(*model.volume())
        return voxel_field.all_views(renderer, size, background(size), material)
