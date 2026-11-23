"""Parametric surfaces for optimization.py's SURFACE field, rendered differentiably.

A surface is a base (BASES: plane XY, sphere, ellipsoid) over the unit uv square, displaced
along the base normal by a deformation (DEFORMATIONS: a stack of plane waves in the base's
wave coordinates), sampled as a mesh of U x V spans. Cameras orbit the origin (z up) and a
renderer (RENDERERS: Gaussian splats) turns the mesh into images, differentiable w.r.t.
every optimizable parameter (the base's, e.g. the ellipsoid radii, and the deformation's).

Everything is described by one JSON-able config dict (normalize_config fills in and
validates it), so the web page can preview a surface without touching a run. Adding a
base, deformation or renderer = a class with the same interface plus a registry entry;
schema() describes the registries to the page, which builds its controls from it.
"""
import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import gaussian_splats_3d


# ── Parameters ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Param:
    """One parameter of a base or a deformation item, as the config / page see it.
    size: number of values (1 = a scalar); lr: Adam learning rate x this (in the
    parameter's optimization space, see the class using it); random: the default range
    a page randomizes it in (None = not randomizable)."""
    key: str
    label: str
    default: float | tuple
    size: int = 1
    step: float = 0.1
    min: float | None = None
    lr: float = 1.0
    random: tuple[float, float] | None = None

    def describe(self) -> dict:
        return {"key": self.key, "label": self.label, "default": self.default, "size": self.size,
                "step": self.step, "min": self.min, "random": self.random}


def _number(value, default: float, lo: float | None = None, hi: float | None = None) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = float(default)
    if not math.isfinite(x):
        x = float(default)
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def _param_value(p: Param, value):
    """A config value of p, cleaned: a float, or a list of p.size floats."""
    if p.size == 1:
        return _number(value, p.default, p.min)
    values = list(value) if isinstance(value, (list, tuple)) else []
    return [_number(values[i] if i < len(values) else p.default[i], p.default[i], p.min) for i in range(p.size)]


def _leaf(value: float | list, optimize: bool, device) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32, device=device).requires_grad_(optimize)


# ── Bases: uv in [0, 1]^2 -> position, unit normal ─────────────────────────────

class Base:
    """A surface over the unit uv square. closed_u: u wraps around (the mesh's last span
    joins its first column); poles: the v = 0 and v = 1 rows each collapse to one point.
    wave_coords are the 2-D coordinates deformations are defined in, roughly arc length."""
    label = ""
    params: tuple[Param, ...] = ()
    closed_u = False
    poles = False

    def __init__(self, cfg: dict, optimize: dict, device):
        self.tensors: dict[str, torch.Tensor] = {}

    def evaluate(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def wave_coords(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def taper(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor | None:
        """Factor on the deformation (None = 1): closes the poles."""
        return None

    def values(self) -> dict:
        return {}


class PlaneXY(Base):
    """z = 0, x and y from -1 to 1; normal +z. Wave coordinates: (x, y)."""
    label = "plane XY"

    def evaluate(self, u, v):
        p = torch.stack([2 * u - 1, 2 * v - 1, torch.zeros_like(u)], dim=-1)
        return p, torch.zeros_like(p) + p.new_tensor([0.0, 0.0, 1.0])

    def wave_coords(self, u, v):
        return 2 * u - 1, 2 * v - 1


class Sphere(Base):
    """Unit sphere: u = longitude (-180..180 deg, wraps), v = latitude (south to north
    pole). Wave coordinates: (longitude, latitude) in radians -- arc length on the equator.
    Deformations taper off near the poles, which would otherwise tear open."""
    label = "sphere"
    closed_u = True
    poles = True

    def _angles(self, u, v):
        return 2 * math.pi * u - math.pi, math.pi * v - math.pi / 2

    def _unit(self, u, v):
        lon, lat = self._angles(u, v)
        return torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=-1)

    def evaluate(self, u, v):
        p = self._unit(u, v)
        return p, p

    def wave_coords(self, u, v):
        return self._angles(u, v)

    def taper(self, u, v):
        _, lat = self._angles(u, v)
        return 1 - (lat.abs() * (2 / math.pi)) ** 6


class Ellipsoid(Sphere):
    """The sphere scaled by the radii (x, y, z), which are optimizable (in log space, so
    they stay positive and the learning rate is a relative change)."""
    label = "ellipsoid"
    params = (Param("radii", "radii x y z", (1.0, 0.75, 0.5), size=3, step=0.05, min=0.01),)

    def __init__(self, cfg, optimize, device):
        super().__init__(cfg, optimize, device)
        radii = _param_value(self.params[0], cfg.get("radii"))
        self.tensors["radii"] = _leaf([math.log(r) for r in radii], optimize.get("radii", False), device)

    def radii(self) -> torch.Tensor:
        return self.tensors["radii"].exp()

    def evaluate(self, u, v):
        r = self.radii()
        unit = self._unit(u, v)
        return unit * r, F.normalize(unit / r, dim=-1)

    def values(self):
        return {"radii": [round(x, 5) for x in self.radii().tolist()]}


BASES: dict[str, type[Base]] = {"plane": PlaneXY, "sphere": Sphere, "ellipsoid": Ellipsoid}


# ── Deformations: a stack of items, each displacing along the base normal ──────

class Deformation:
    """A list of items (config: cfg["items"], each a dict of `params`) whose
    displacements add up; the result moves each point along the base normal."""
    label = ""
    item_label = "item"
    params: tuple[Param, ...] = ()

    def __init__(self, cfg: dict, optimize: dict, device):
        self.tensors: dict[str, torch.Tensor] = {}

    def displacement(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def values(self) -> dict:
        return {}


class PlaneWaves(Deformation):
    """h(s, t) = sum_k amplitude_k sin(2 pi frequency_k (s cos angle_k + t sin angle_k)
    + phase_k) in the base's wave coordinates (s, t). Frequency is in cycles per unit of
    (s, t), optimized as its log (positive, relative steps); angles and phases in radians."""
    label = "plane waves"
    item_label = "wave"
    params = (
        Param("angle", "angle°", 0.0, step=5, random=(0.0, 360.0)),
        Param("frequency", "freq", 1.0, step=0.1, min=0.01, random=(0.5, 4.0)),
        Param("amplitude", "amp", 0.05, step=0.01, lr=0.1, random=(0.01, 0.08)),
        Param("phase", "phase°", 0.0, step=5, random=(0.0, 360.0)),
    )

    def __init__(self, cfg, optimize, device):
        super().__init__(cfg, optimize, device)
        items = cfg.get("items") or []
        col = lambda key: [float(item[key]) for item in items]
        self.count = len(items)
        self.tensors = {
            "angle": _leaf([math.radians(a) for a in col("angle")], optimize.get("angle", False), device),
            "frequency": _leaf([math.log(f) for f in col("frequency")], optimize.get("frequency", False), device),
            "amplitude": _leaf(col("amplitude"), optimize.get("amplitude", False), device),
            "phase": _leaf([math.radians(p) for p in col("phase")], optimize.get("phase", False), device),
        }

    def displacement(self, s, t):
        if not self.count:
            return torch.zeros_like(s)
        x = self.tensors
        angle, phase = x["angle"].view(-1, 1), x["phase"].view(-1, 1)
        k = 2 * math.pi * x["frequency"].exp().view(-1, 1)
        arg = k * (s.reshape(1, -1) * angle.cos() + t.reshape(1, -1) * angle.sin()) + phase
        return (x["amplitude"].view(-1, 1) * arg.sin()).sum(0).view(s.shape)

    def values(self):
        x = {k: v.detach() for k, v in self.tensors.items()}
        cols = {"angle": torch.rad2deg(x["angle"]) % 360, "frequency": x["frequency"].exp(),
                "amplitude": x["amplitude"], "phase": torch.rad2deg(x["phase"]) % 360}
        cols = {k: v.tolist() for k, v in cols.items()}
        return {"items": [{k: round(cols[k][i], 5) for k in cols} for i in range(self.count)]}


DEFORMATIONS: dict[str, type[Deformation]] = {"plane_waves": PlaneWaves}


# ── Cameras and lights ─────────────────────────────────────────────────────────

CAMERA_PARAMS = (
    Param("distance", "dist", 3.5, step=0.1, min=0.1),
    Param("azimuth", "az°", 0.0, step=5),
    Param("elevation", "el°", 30.0, step=5),
    Param("fov", "fov°", 40.0, step=1, min=1),
    Param("jitter", "jit°", 5.0, step=1, min=0),
)

LIGHT_MODES = ["shared", "camera", "random"]
LIGHT_MODE_LABELS = {"shared": "one light for all views", "camera": "a light per camera (relative to it)",
                     "random": "random light per view"}
SHADINGS = ["smooth", "faceted"]


@dataclass
class View:
    """One camera pose: orbit (distance, azimuth, elevation in degrees) around the origin,
    vertical fov in degrees, looking at the origin with z up; light: unit vector towards
    the light, in world space."""
    distance: float
    azimuth: float
    elevation: float
    fov: float
    light: torch.Tensor

    def frame(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """(eye (3,), rotation (3, 3): rows = right, up, forward) in world space."""
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        eye = self.distance * torch.tensor([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
                                           device=device)
        forward = -eye / eye.norm()
        up = torch.tensor([0.0, 0.0, 1.0], device=device)
        if abs(float(forward @ up)) > 0.999:  # looking straight up / down
            up = torch.tensor([-math.cos(az), -math.sin(az), 0.0], device=device) * (1 if el > 0 else -1)
        right = F.normalize(torch.linalg.cross(forward, up), dim=0)
        return eye, torch.stack([right, torch.linalg.cross(right, forward), forward])


def _direction(azimuth: float, elevation: float, frame: torch.Tensor | None = None) -> torch.Tensor:
    """Unit vector from angles in degrees: in world space (z up), or with a camera frame
    relative to it (0, 0 = towards the camera, azimuth turns right, elevation up)."""
    az, el = math.radians(azimuth), math.radians(elevation)
    if frame is None:
        return torch.tensor([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right, up, forward = frame.cpu()
    return F.normalize(math.cos(el) * (math.cos(az) * -forward + math.sin(az) * right) + math.sin(el) * up, dim=0)


def make_views(cfg: dict, jitter: bool, device, rng: random.Random | None = None) -> list[View]:
    """The cameras of a config (render.views_per_camera each when jittering), with their
    lights by light.mode. jitter: each camera's azimuth / elevation move by up to its
    jitter in degrees, distance and fov by up to jitter %; the light by light.jitter
    degrees (one shared offset per call in "shared" mode, else per view). Without jitter
    (the canvas, previews) random lights come from a fixed seed."""
    rng = rng or (random.Random() if jitter else random.Random(0))
    light = cfg["light"]
    repeat = cfg["render"]["views_per_camera"] if jitter else 1
    j = lambda amount: rng.uniform(-amount, amount) if jitter else 0.0
    shared = (j(light["jitter"]), j(light["jitter"]))
    views = []
    for cam in cfg["cameras"]:
        for _ in range(repeat):
            a = cam["jitter"]
            view = View(cam["distance"] * (1 + j(a) / 100), cam["azimuth"] + j(a),
                        max(-89.9, min(89.9, cam["elevation"] + j(a))), cam["fov"] * (1 + j(a) / 100), None)
            _, frame = view.frame("cpu")
            if light["mode"] == "shared":
                view.light = _direction(light["azimuth"] + shared[0], light["elevation"] + shared[1])
            elif light["mode"] == "camera":
                view.light = _direction(light["azimuth"] + j(light["jitter"]), light["elevation"] + j(light["jitter"]), frame)
            else:  # random: uniform over the hemisphere facing the camera
                d = F.normalize(torch.tensor([rng.gauss(0, 1) for _ in range(3)]), dim=0)
                view.light = d if float(d @ -frame[2]) >= 0 else -d
            view.light = view.light.to(device)
            views.append(view)
    return views


# ── The surface: base + deformation, meshed ────────────────────────────────────

class SurfaceModel:
    """The optimizable surface of a (normalized) config: its base and deformation, meshed
    at the config's spans. Parameters are leaf tensors; those not set to optimize have no
    gradient."""

    def __init__(self, cfg: dict, device):
        self.cfg = cfg
        self.device = device
        optimize = cfg["optimize"]
        self.base = BASES[cfg["base"]["type"]](cfg["base"], optimize, device)
        self.deformation = DEFORMATIONS[cfg["deformation"]["type"]](cfg["deformation"], optimize, device)
        self.spans = tuple(cfg["spans"])
        self.faceted = cfg["shading"] == "faceted"
        nu = self.spans[0] if self.base.closed_u else self.spans[0] + 1
        u = torch.arange(nu, device=device) / self.spans[0]
        v = torch.arange(self.spans[1] + 1, device=device) / self.spans[1]
        self.v, self.u = torch.meshgrid(v, u, indexing="ij")  # (V + 1, nu)
        self._quads = _quad_corners(self.spans, self.base.closed_u, device)

    def param_groups(self) -> list[tuple[torch.Tensor, float]]:
        """(tensor, lr scale) of every optimized parameter."""
        groups = []
        for module in (self.base, self.deformation):
            scales = {p.key: p.lr for p in module.params}
            groups += [(t, scales.get(k, 1.0)) for k, t in module.tensors.items() if t.requires_grad and t.numel()]
        return groups

    def vertices(self) -> torch.Tensor:
        """(V + 1, nu, 3) mesh vertices: base + deformation along the base normal."""
        p, n = self.base.evaluate(self.u, self.v)
        s, t = self.base.wave_coords(self.u, self.v)
        h = self.deformation.displacement(s, t)
        taper = self.base.taper(self.u, self.v)
        if taper is not None:
            h = h * taper
        return p + n * h[..., None]

    def values(self) -> dict:
        """The current parameters, in the config's format (degrees, plain frequency)."""
        with torch.no_grad():
            return {"base": self.base.values(), "deformation": self.deformation.values()}

    def write_obj(self, path) -> None:
        """The mesh as a Wavefront OBJ (quads, uv per vertex)."""
        with torch.no_grad():
            verts = self.vertices().reshape(-1, 3).cpu().tolist()
        uvs = torch.stack([self.u, self.v], -1).reshape(-1, 2).cpu().tolist()
        a, b, c, d = (q.cpu().tolist() for q in self._quads)
        lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
        lines += [f"vt {s:.6f} {t:.6f}" for s, t in uvs]
        lines += [f"f {i + 1}/{i + 1} {j + 1}/{j + 1} {l + 1}/{l + 1} {k + 1}/{k + 1}" for i, j, k, l in zip(a, b, c, d)]
        path.write_text("\n".join(lines) + "\n")


def _quad_corners(spans, closed_u: bool, device) -> tuple[torch.Tensor, ...]:
    """Flat vertex indices (a, b, c, d) of every quad, row-major over (V, U): a = (i, j),
    b = (i, j + 1), c = (i + 1, j), d = (i + 1, j + 1); u wraps when closed."""
    U, V = spans
    nu = U if closed_u else U + 1
    i, j = torch.meshgrid(torch.arange(V, device=device), torch.arange(U, device=device), indexing="ij")
    j1 = (j + 1) % nu
    flat = lambda r, c: (r * nu + c).reshape(-1)
    return flat(i, j), flat(i, j1), flat(i + 1, j), flat(i + 1, j1)


# ── Renderers ──────────────────────────────────────────────────────────────────

def hex_color(value: str, device) -> torch.Tensor:
    value = value.lstrip("#")
    return torch.tensor([int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)], device=device)


class SplatRenderer:
    """Surfels: every quad of the mesh is sampled D x D times (on its two triangles, so
    the splats lie on the faceted mesh), and each sample is a flat 2-D Gaussian in the
    surface's tangent plane, projected to the screen and alpha-composited front to back
    per pixel (like 3D Gaussian splatting, pure PyTorch). Silhouettes are soft over about
    one sample spacing, so they pass gradients too. D adapts to the views: samples about
    render.spacing pixels apart (at most MAX_SPLATS). Normals: per triangle (faceted) or
    interpolated vertex normals (smooth); lit two-sided with Lambert + Blinn-Phong."""
    label = "gaussian splats"
    MAX_SPLATS = 1 << 18
    MAX_RADIUS = 10       # px: the splat window's half size is capped here
    SIGMA = 0.75          # Gaussian std, in sample spacings
    LOW_PASS = 0.3        # px^2 added to every projected covariance (anti-aliasing)
    OPACITY = 0.99
    NEAR = 0.05
    SHININESS = 24.0

    def __init__(self, cfg: dict, device):
        self.cfg = cfg
        self.device = device
        self.spacing = cfg["render"]["spacing"]
        light = cfg["light"]
        self.ambient, self.specular = light["ambient"], light["specular"]
        self.color = hex_color(cfg["material"]["color"], device)
        self.background = hex_color(cfg["material"]["background"], device).requires_grad_(
            cfg["material"]["optimize_background"])
        self._load_image = None     # set_background_image
        self._multiply = False
        self._images: dict[tuple[int, int], torch.Tensor] = {}
        self._samples: dict[int, tuple] = {}
        # The last render's sizes (what its memory scales with): samples per span edge, splats,
        # and per view the visible splats, window half size (px) and covered (splat, pixel) entries.
        self.stats: dict = {}

    def set_background_image(self, load, multiply: bool = False) -> None:
        """Every view's background becomes an image. load(size) -> (1, 4, H, W), its RGB + alpha
        framed for a view of size (width, height), called once per size (see paint_background)."""
        self._load_image, self._multiply = load, multiply
        self._images.clear()

    def background_color(self) -> torch.Tensor:
        """(3,) in [0, 1]. Kept linear, the gradient passing a clamp at the bounds straight through:
        a white or black start would sit where a sigmoid is flat."""
        c = self.background
        return c + (c.clamp(0, 1) - c).detach()

    def backdrop(self, size) -> torch.Tensor:
        """(1, 3, H, W) what shows where a view of size (width, height) has no surface."""
        size = tuple(size)
        if self._load_image is not None and size not in self._images:
            self._images[size] = self._load_image(size).to(self.device)
        return paint_background(self.background_color(), self._images.get(size), size, self._multiply)

    # ── geometry ──

    def _sample_layout(self, model: SurfaceModel, d: int) -> tuple:
        """Per sample (Q * d * d of them): its quad, barycentric weights on (a, b, c, d) and
        which triangle it is on (True: (a, b, d), False: (a, d, c))."""
        if d not in self._samples:
            k = (torch.arange(d, device=self.device) + 0.5) / d
            fv, fu = torch.meshgrid(k, k, indexing="ij")
            fu, fv = fu.reshape(-1), fv.reshape(-1)
            w = torch.stack([1 - torch.maximum(fu, fv), (fu - fv).clamp(min=0),
                             (fv - fu).clamp(min=0), torch.minimum(fu, fv)], dim=-1)  # (d*d, 4)
            quads = len(model._quads[0])
            self._samples[d] = (torch.arange(quads, device=self.device).repeat_interleave(d * d),
                                w.repeat(quads, 1), (fu >= fv).repeat(quads))
        return self._samples[d]

    def _surfels(self, model: SurfaceModel, d: int):
        """(positions, normals, tangent u, tangent v), (N, 3) each; tangents span one
        sample spacing."""
        verts = model.vertices().reshape(-1, 3)
        a, b, c, dd = (torch.index_select(verts, 0, q) for q in model._quads)
        n1 = torch.linalg.cross(b - a, dd - a)   # triangle (a, b, d), area weighted
        n2 = torch.linalg.cross(dd - a, c - a)   # triangle (a, d, c)
        quad, w, upper = self._sample_layout(model, d)
        per_sample = lambda x: torch.index_select(x, 0, quad)    # (Q, ...) -> (N, ...)
        corners = per_sample(torch.stack([a, b, c, dd], dim=1))   # (N, 4, 3)
        pos = (w[..., None] * corners).sum(1)
        if model.faceted:
            normal = torch.where(upper[:, None], per_sample(n1), per_sample(n2))
        else:
            vn = torch.zeros_like(verts)
            for idx, n in ((model._quads[0], n1 + n2), (model._quads[1], n1), (model._quads[3], n1 + n2),
                           (model._quads[2], n2)):
                vn = vn.index_add(0, idx, n)
            if model.base.poles:  # the pole rows are one point each: one shared normal
                vn = vn.view(*model.u.shape, 3)
                vn = torch.cat([vn[:1].sum(1, keepdim=True).expand_as(vn[:1]), vn[1:-1],
                                vn[-1:].sum(1, keepdim=True).expand_as(vn[-1:])])
                vn = vn.reshape(-1, 3)
            vn = F.normalize(vn, dim=-1, eps=1e-12)
            corner_normals = torch.stack([torch.index_select(vn, 0, q) for q in model._quads], dim=1)
            normal = (w[..., None] * per_sample(corner_normals)).sum(1)
        tu = per_sample((b - a) + (dd - c)) / (2 * d)
        tv = per_sample((c - a) + (dd - b)) / (2 * d)
        return pos, F.normalize(normal, dim=-1, eps=1e-12), tu, tv

    def _density(self, model: SurfaceModel, views: list[View], size) -> int:
        """Samples per span edge: the 95th percentile of the projected mesh edges over the
        views, divided by the target spacing."""
        with torch.no_grad():
            verts = model.vertices().reshape(-1, 3)
            a, b, c, _ = model._quads
            lengths = []
            for view in views:
                xy, z = self._project(verts, view, size)[:2]
                ok = (z[a] > self.NEAR) & (z[b] > self.NEAR) & (z[c] > self.NEAR)
                lengths.append(torch.cat([(xy[b] - xy[a]).norm(dim=-1)[ok], (xy[c] - xy[a]).norm(dim=-1)[ok]]))
            lengths = torch.cat(lengths)
            edge = torch.quantile(lengths.float()[:1 << 20], 0.95).item() if len(lengths) else 1.0
        cap = max(1, int(math.sqrt(self.MAX_SPLATS / len(a))))
        return max(1, min(cap, math.ceil(edge / self.spacing)))

    def _project(self, pos: torch.Tensor, view: View, size):
        """Screen positions (N, 2) px, depth (N,), camera-space positions (N, 3), focal px."""
        w, h = size
        eye, rot = view.frame(pos.device)
        cam = (pos - eye) @ rot.T
        focal = (h / 2) / math.tan(math.radians(view.fov) / 2)
        z = cam[:, 2]
        zs = z.clamp(min=self.NEAR)
        xy = torch.stack([w / 2 + focal * cam[:, 0] / zs, h / 2 - focal * cam[:, 1] / zs], dim=-1)
        return xy, z, cam, focal, eye, rot

    # ── rendering ──

    def render(self, model: SurfaceModel, views: list[View], size) -> torch.Tensor:
        """(len(views), 3, H, W) in [0, 1]; size is (width, height)."""
        d = self._density(model, views, size)
        surfels = self._surfels(model, d)
        self.stats = {"density": d, "splats": len(surfels[0]), "views": []}
        return torch.stack([self._render_view(surfels, view, size) for view in views])

    def _render_view(self, surfels, view: View, size) -> torch.Tensor:
        w, h = size
        pos, normal, tu, tv = surfels
        xy, z, cam, focal, eye, rot = self._project(pos, view, size)

        # Projected covariance: J (sigma^2 (tu tu^T + tv tv^T)) J^T + low pass, J = d(screen)/d(camera).
        def screen(t):
            tc = t @ rot.T
            zs = z.clamp(min=self.NEAR)
            return torch.stack([focal * (tc[:, 0] / zs - cam[:, 0] * tc[:, 2] / zs ** 2),
                                -focal * (tc[:, 1] / zs - cam[:, 1] * tc[:, 2] / zs ** 2)], dim=-1)
        a2, b2 = screen(tu), screen(tv)
        s2 = self.SIGMA ** 2
        sxx = s2 * (a2[:, 0] ** 2 + b2[:, 0] ** 2) + self.LOW_PASS
        syy = s2 * (a2[:, 1] ** 2 + b2[:, 1] ** 2) + self.LOW_PASS
        sxy = s2 * (a2[:, 0] * a2[:, 1] + b2[:, 0] * b2[:, 1])
        conic, radius = gaussian_splats_3d.conic_and_radius(sxx, sxy, syy)
        # Shaded only where it can be seen: the culling composite does again, on these.
        with torch.no_grad():
            r = radius.clamp(max=self.MAX_RADIUS)
            visible = ((z > self.NEAR) & (xy[:, 0] + r > 0) & (xy[:, 0] - r < w) & (xy[:, 1] + r > 0)
                       & (xy[:, 1] - r < h)).nonzero().squeeze(1)
        pick = lambda x: torch.index_select(x, 0, visible)
        color = self._shade(pick(pos), pick(normal), eye, view.light)
        img, stats = gaussian_splats_3d.composite(pick(xy), pick(conic), pick(radius), pick(z), color, self.OPACITY,
                                                  size, self.backdrop(size)[0], self.MAX_RADIUS, self.NEAR)
        self.stats["views"].append(stats)
        return img

    def _shade(self, pos, normal, eye, light) -> torch.Tensor:
        """(N, 3) two-sided Lambert + Blinn-Phong."""
        to_eye = F.normalize(eye - pos, dim=-1)
        n = torch.where((normal * to_eye).sum(-1, keepdim=True) < 0, -normal, normal)
        diffuse = (n @ light).clamp(min=0)
        half = F.normalize(to_eye + light, dim=-1)
        spec = (n * half).sum(-1).clamp(min=0) ** self.SHININESS * (diffuse > 0)
        lit = self.color * (self.ambient + (1 - self.ambient) * diffuse)[:, None]
        return lit + self.specular * spec[:, None]


def paint_background(color: torch.Tensor, image: torch.Tensor | None, size, multiply: bool = False) -> torch.Tensor:
    """(1, C, H, W) a canvas' background, size (width, height): the solid colour (C,), or with an
    image ((1, C + 1, H, W): straight colours + alpha) the image over that colour, so its transparent
    parts and letterbox bars show the colour. multiply: the image's colours x the colour as well."""
    c = color.view(1, -1, 1, 1)
    if image is None:
        return c.expand(1, -1, size[1], size[0])
    rgb, alpha = image[:, :-1], image[:, -1:]
    return (rgb * c if multiply else rgb) * alpha + c * (1 - alpha)


RENDERERS: dict[str, type] = {"splats": SplatRenderer}


def make_renderer(cfg: dict, device):
    return RENDERERS[cfg["render"]["type"]](cfg, device)


def sheet_layout(count: int, size) -> tuple[int, int, tuple[int, int]]:
    """(cols, rows, tile size) of a contact sheet of `count` views on a canvas of size
    (w, h): tiles keep the canvas' aspect."""
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))
    return cols, rows, (max(8, size[0] // cols), max(8, size[1] // cols))


def render_sheet(model: SurfaceModel, renderer, views: list[View], size) -> torch.Tensor:
    """(1, 3, H, W): the views tiled on the canvas (row-major), centred vertically, the
    rest the background colour (a background image is behind each view, in its tile)."""
    w, h = size
    cols, rows, (tw, th) = sheet_layout(len(views), size)
    tiles = renderer.render(model, views, (tw, th))
    sheet = renderer.background_color().view(1, 3, 1, 1).expand(1, 3, h, w).clone()
    top = max(0, (h - rows * th) // 2)
    left = (w - cols * tw) // 2
    for i, tile in enumerate(tiles):
        y, x = top + (i // cols) * th, left + (i % cols) * tw
        sheet[0, :, y:y + th, x:x + tw] = tile[:, :min(th, h - y), :min(tw, w - x)]
    return sheet


# ── Config ─────────────────────────────────────────────────────────────────────

def _items(params: tuple[Param, ...], items) -> list[dict]:
    return [{p.key: _param_value(p, (item or {}).get(p.key)) for p in params}
            for item in (items if isinstance(items, list) else [])]


def _random_ranges(params: tuple[Param, ...], given) -> dict:
    given = given if isinstance(given, dict) else {}
    out = {}
    for p in params:
        if p.random is None:
            continue
        r = given.get(p.key)
        r = r if isinstance(r, list) and len(r) == 2 else p.random
        out[p.key] = [_number(r[0], p.random[0]), _number(r[1], p.random[1])]
    return out


def default_config() -> dict:
    return normalize_config({
        "base": {"type": "plane"},
        "deformation": {"type": "plane_waves", "items": [
            {"angle": 20, "frequency": 1.0, "amplitude": 0.06, "phase": 0},
            {"angle": 110, "frequency": 2.2, "amplitude": 0.03, "phase": 90},
            {"angle": 250, "frequency": 3.5, "amplitude": 0.02, "phase": 200},
        ]},
        "cameras": [{"distance": 3.5, "azimuth": az, "elevation": 35, "fov": 40, "jitter": 5} for az in (-60, 30, 120, 210)],
    })


def normalize_config(cfg: dict | None) -> dict:
    """A surface config with every key filled in (defaults) and every value in range.
    Raises ValueError on an unknown base / deformation / renderer / light mode / shading.
    The base keeps the values of every base's parameters (switching back restores them)."""
    cfg = cfg if isinstance(cfg, dict) else {}
    section = lambda key: cfg.get(key) if isinstance(cfg.get(key), dict) else {}

    def choice(value, choices, default, what):
        value = value or default
        if value not in choices:
            raise ValueError(f"Bad surface {what}: {value!r} (one of {list(choices)})")
        return value

    base_in = section("base")
    base = {"type": choice(base_in.get("type"), BASES, "plane", "base")}
    for cls in BASES.values():
        for p in cls.params:
            base.setdefault(p.key, _param_value(p, base_in.get(p.key)))

    deform_in = section("deformation")
    dtype = choice(deform_in.get("type"), DEFORMATIONS, "plane_waves", "deformation")
    dparams = DEFORMATIONS[dtype].params
    items = _items(dparams, deform_in.get("items"))[:256]
    deformation = {"type": dtype, "items": items, "random": _random_ranges(dparams, deform_in.get("random"))}

    optimize_in = section("optimize")
    optimize = {p.key: bool(optimize_in.get(p.key, True))
                for cls in [*BASES.values(), *DEFORMATIONS.values()] for p in cls.params}

    spans = cfg.get("spans") if isinstance(cfg.get("spans"), list) else [64, 64]
    spans = [int(_number(spans[i] if i < len(spans) else 64, 64, 1, 512)) for i in range(2)]
    if BASES[base["type"]].closed_u:
        spans[0] = max(3, spans[0])

    cameras = _items(CAMERA_PARAMS, cfg.get("cameras"))[:32]
    if not cameras:
        cameras = _items(CAMERA_PARAMS, [{}])
    for cam in cameras:
        cam["elevation"] = max(-89.9, min(89.9, cam["elevation"]))
        cam["fov"] = min(170.0, cam["fov"])

    light_in = section("light")
    light = {"mode": choice(light_in.get("mode"), LIGHT_MODES, "shared", "light mode"),
             "azimuth": _number(light_in.get("azimuth"), 45), "elevation": _number(light_in.get("elevation"), 50),
             "jitter": _number(light_in.get("jitter"), 15, 0), "ambient": _number(light_in.get("ambient"), 0.15, 0, 1),
             "specular": _number(light_in.get("specular"), 0.25, 0)}

    def color(value, default):
        value = str(value or default)
        return value.lower() if len(value) == 7 and value[0] == "#" and all(
            ch in "0123456789abcdefABCDEF" for ch in value[1:]) else default
    material_in = section("material")
    material = {"color": color(material_in.get("color"), "#c8c8c8"),
                "background": color(material_in.get("background"), "#ffffff"),
                "optimize_background": bool(material_in.get("optimize_background", False))}

    render_in = section("render")
    render = {"type": choice(render_in.get("type"), RENDERERS, "splats", "renderer"),
              "spacing": _number(render_in.get("spacing"), 1.0, 0.25, 8),
              "views_per_camera": int(_number(render_in.get("views_per_camera"), 1, 1, 16))}

    return {"base": base, "deformation": deformation, "optimize": optimize, "spans": spans,
            "shading": choice(cfg.get("shading"), SHADINGS, "smooth", "shading"),
            "cameras": cameras, "light": light, "material": material, "render": render}


def schema() -> dict:
    """The registries for the page: bases and deformations with their parameters,
    renderers, light modes, shadings, the camera parameters."""
    return {
        "bases": [{"id": k, "label": c.label, "params": [p.describe() for p in c.params]} for k, c in BASES.items()],
        "deformations": [{"id": k, "label": c.label, "item_label": c.item_label,
                          "params": [p.describe() for p in c.params]} for k, c in DEFORMATIONS.items()],
        "renderers": [{"id": k, "label": c.label} for k, c in RENDERERS.items()],
        "light_modes": [{"id": m, "label": LIGHT_MODE_LABELS[m]} for m in LIGHT_MODES],
        "shadings": SHADINGS,
        "camera_params": [p.describe() for p in CAMERA_PARAMS],
    }


def preview(cfg: dict, size, device, load_image=None, multiply: bool = False) -> torch.Tensor:
    """(1, 3, H, W) contact sheet of a config's un-jittered views. load_image, multiply: a
    background image (SplatRenderer.set_background_image)."""
    cfg = normalize_config(cfg)
    with torch.no_grad():
        model = SurfaceModel(cfg, device)
        renderer = make_renderer(cfg, device)
        if load_image is not None:
            renderer.set_background_image(load_image, multiply)
        return render_sheet(model, renderer, make_views(cfg, False, device), size)
