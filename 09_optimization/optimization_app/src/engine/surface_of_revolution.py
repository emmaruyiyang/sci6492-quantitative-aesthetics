"""Surfaces of revolution for optimization.py's REVOLUTION field, without a rendering pipeline.

A half profile gives the radius r(t) along the vertical axis, t = 0 at the bottom and 1 at the
top. The camera is orthographic and looks straight at the axis, so nothing ever occludes anything
and each pixel's surface point has a closed form: at x, y (canvas units, x right, y up, the canvas'
short side = 1) with s = x / r(y),

    mask     soft |x| < r(y), and y within the surface's height
    depth    z = r sqrt(1 - s^2)
    normal   (s, -dr/dy, sqrt(1 - s^2)), normalized
    uv       u = asin(s) / 2 pi + angle / 360 + 1/2, v = t

Turning the surface only moves u: mask, normals and depth are the same from every angle, so one
evaluation per step serves all the views.

The profile is base radius + a radial offset (normalize_config's mode):

    points      n points evenly along t, each with its offset and how it interpolates to the next
    waves       a stack of superimposed waves
    parametric  a formula of t; every other name that isn't a function (FUNCTIONS) or pi is a free
                parameter

write_rhino saves the surface as a mesh with texture coordinates, its texture referenced and embedded.
"""
import ast
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODES = ["points", "waves", "parametric"]
INTERPOLATIONS = ["linear", "nn", "spline"]
WAVES = ["sine", "triangle", "square"]
SHADINGS = ["lit", "flat"]
RADIUS_SHARPNESS = 100.0  # the radius is softplus(sharpness x raw) / sharpness: raw above ~0.05, always > 0
SLOPE_DT = 1e-3           # central difference step, in t
SQUARE_SHARPNESS = 8.0    # square wave = tanh(sharpness * sin): higher is steeper, with less gradient
MAX_POINTS = 256
MAX_FORMULA = 500         # characters
FORMULA_LIMIT = 10.0      # a formula's values are clamped to +-this (canvas short sides)
_EPS = 1e-12

POINT_DEFAULT = {"offset": 0.0, "interpolation": "spline"}
WAVE_DEFAULT = {"wave": "sine", "components": 2, "frequency": 1.0, "optimize_frequency": True, "amplitude": 0.05,
                "phase": 0.0}
PARAMETER_DEFAULT = {"value": 0.1, "optimize": True}


def wave_function(name: str, theta: torch.Tensor) -> torch.Tensor:
    """A wave with period 2 pi and range -1..1. Triangle and square are slightly rounded: their
    phases need a gradient, and the exact shapes' is infinite or zero."""
    sin = torch.sin(theta)
    if name == "triangle":
        return torch.asin(0.999 * sin) / math.asin(0.999)
    if name == "square":
        return torch.tanh(SQUARE_SHARPNESS * sin) / math.tanh(SQUARE_SHARPNESS)
    return sin


# ── Formulas ───────────────────────────────────────────────────────────────────
# Parsed with ast and walked into closures: nothing is ever passed to eval, and only the node types
# and functions below exist. Domains are clamped so that a parameter wandering out of one
# gives a large value, not a NaN.

FUNCTIONS = {
    "sin": (1, torch.sin), "cos": (1, torch.cos), "tan": (1, torch.tan),
    "asin": (1, lambda x: torch.asin(x.clamp(-1 + 1e-6, 1 - 1e-6))),
    "acos": (1, lambda x: torch.acos(x.clamp(-1 + 1e-6, 1 - 1e-6))),
    "atan": (1, torch.atan), "atan2": (2, torch.atan2),
    "sinh": (1, torch.sinh), "cosh": (1, torch.cosh), "tanh": (1, torch.tanh),
    "exp": (1, torch.exp), "log": (1, lambda x: torch.log(x.clamp_min(_EPS))),
    "sqrt": (1, lambda x: torch.sqrt(x.clamp_min(_EPS))), "abs": (1, torch.abs),
    "pow": (2, torch.pow), "min": (2, torch.minimum), "max": (2, torch.maximum),
    "clamp": (3, lambda x, lo, hi: torch.minimum(torch.maximum(x, lo), hi)),
    "sign": (1, torch.sign), "floor": (1, torch.floor), "ceil": (1, torch.ceil), "round": (1, torch.round),
    "sigmoid": (1, torch.sigmoid),
}
RESERVED = ["t", "pi"]  # names that are not free parameters, besides FUNCTIONS
_BINARY = {ast.Add: torch.add, ast.Sub: torch.sub, ast.Mult: torch.mul, ast.Div: torch.div, ast.Pow: torch.pow,
           ast.Mod: torch.remainder}


def _compile(node: ast.AST, names: list[str]):
    """An expression node -> a closure of {name: tensor}; `names` collects its free parameters."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Not a number: {node.value!r}")
        value = float(node.value)
        return lambda env: env["t"].new_tensor(value)
    if isinstance(node, ast.Name):
        name = node.id
        if name in FUNCTIONS:
            raise ValueError(f"{name} is a function: write {name}(...)")
        if name == "pi":
            return lambda env: env["t"].new_tensor(math.pi)
        if name != "t" and name not in names:
            names.append(name)
        return lambda env: env[name]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _compile(node.operand, names)
        return inner if isinstance(node.op, ast.UAdd) else (lambda env: -inner(env))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        op, left, right = _BINARY[type(node.op)], _compile(node.left, names), _compile(node.right, names)
        return lambda env: op(left(env), right(env))
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name not in FUNCTIONS:
            raise ValueError(f"Unknown function: {name or ast.dump(node.func)} (one of {', '.join(FUNCTIONS)})")
        arity, function = FUNCTIONS[name]
        if node.keywords or len(node.args) != arity:
            raise ValueError(f"{name} takes {arity} argument{'s' if arity > 1 else ''}")
        args = [_compile(a, names) for a in node.args]
        return lambda env: function(*(a(env) for a in args))
    raise ValueError(f"Not allowed in a formula: {type(node).__name__}")


@dataclass
class Formula:
    function: object
    parameters: list[str]  # the free parameters, in order of appearance

    def __call__(self, t: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        """(N,) the formula along t (N,); out of range values are clamped, NaNs are 0."""
        out = self.function({**values, "t": t})
        out = torch.nan_to_num(out, nan=0.0, posinf=FORMULA_LIMIT, neginf=-FORMULA_LIMIT)
        return out.clamp(-FORMULA_LIMIT, FORMULA_LIMIT).expand_as(t)


def parse_formula(text: str) -> Formula:
    """A formula of t -> Formula. ValueError names what is wrong with it."""
    text = str(text or "").strip()
    if not text:
        raise ValueError("Empty formula")
    if len(text) > MAX_FORMULA:
        raise ValueError(f"Formula longer than {MAX_FORMULA} characters")
    names: list[str] = []
    try:
        # ^ reads as a power, as in a typed formula. In the text: Python's own ^ binds looser than +.
        function = _compile(ast.parse(text.replace("^", "**"), mode="eval").body, names)
    except SyntaxError as exc:
        raise ValueError(f"Can't read the formula: {exc.msg}") from exc
    except (RecursionError, MemoryError) as exc:
        raise ValueError("Formula too deeply nested") from exc
    return Formula(function, names)


# ── Config ─────────────────────────────────────────────────────────────────────

def default_config() -> dict:
    return normalize_config({})


def _filled(defaults: dict, given: dict | None) -> dict:
    return {**defaults, **{k: v for k, v in (given or {}).items() if v not in (None, "")}}


def _choice(value, choices: list[str], what: str) -> str:
    if value not in choices:
        raise ValueError(f"Bad {what}: {value!r} (one of {choices})")
    return value


def _hex(value, default: str) -> str:
    value = str(value or default)
    ok = len(value) == 7 and value[0] == "#" and all(ch in "0123456789abcdefABCDEF" for ch in value[1:])
    return value.lower() if ok else default


def _clamped(value, lo: float, hi: float) -> float:
    return min(hi, max(lo, float(value)))


def reconcile_parameters(formula: Formula, given: dict | None) -> dict:
    """{name: {value, optimize}} for the formula's free parameters, in order: the given ones keep
    their settings, new ones start at PARAMETER_DEFAULT, those no longer used are dropped."""
    given = given or {}
    out = {}
    for name in formula.parameters:
        p = _filled(PARAMETER_DEFAULT, given.get(name))
        out[name] = {"value": float(p["value"]), "optimize": bool(p["optimize"])}
    return out


def normalize_config(cfg: dict | None) -> dict:
    """A REVOLUTION config, defaults filled in and validated (lengths in canvas short sides):
    {mode, radius > 0, optimize_radius, height > 0, softness (canvas px), shape_lr (x the learning
    rate), optimize_profile, canvas_angle (degrees),
     points: [{offset, interpolation (to the next point; the last one's is unused)}] (at least 2,
       first = the bottom),
     waves: [{wave, components, frequency (cycles along t), optimize_frequency, amplitude, phase}],
     parametric: {formula, parameters: {name: {value, optimize}}},
     texture: {enabled, monochrome, resolution [w, h], levels, lr_decay, views (angles per step)},
     color, optimize_color (the surface, without a texture),
     shading: {mode, azimuth, elevation (degrees; 0, 0 = from the camera), ambient},
     mesh: {rows, segments} (the Rhino export)}.
    Only the current mode's formula has to parse."""
    cfg = _filled({"mode": "points", "radius": 0.25, "optimize_radius": False, "height": 0.9, "softness": 1.0,
                   "shape_lr": 0.5, "optimize_profile": True, "canvas_angle": 0.0, "color": "#bbbbbb",
                   "optimize_color": False}, cfg)
    mode = _choice(cfg["mode"], MODES, "profile mode")
    points = []
    for i, given in enumerate(cfg.get("points") or [{} for _ in range(8)]):
        p = _filled(POINT_DEFAULT, given)
        points.append({"offset": float(p["offset"]),
                       "interpolation": _choice(p["interpolation"], INTERPOLATIONS, f"point {i + 1} interpolation")})
    if not 2 <= len(points) <= MAX_POINTS:
        raise ValueError(f"The profile needs 2 to {MAX_POINTS} points, not {len(points)}")
    waves = []
    for i, given in enumerate(cfg.get("waves") or [{}]):
        w = _filled(WAVE_DEFAULT, given)
        if float(w["frequency"]) <= 0:
            raise ValueError(f"Wave {i + 1}: the frequency must be > 0")
        waves.append({"wave": _choice(w["wave"], WAVES, f"wave {i + 1} shape"),
                      "components": max(1, int(w["components"])), "frequency": float(w["frequency"]),
                      "optimize_frequency": bool(w["optimize_frequency"]),
                      "amplitude": max(0.0, float(w["amplitude"])), "phase": float(w["phase"])})
    given = cfg.get("parametric") or {}
    parametric = {"formula": str(given.get("formula") or "a * sin(2 * pi * f * t + p)"),
                  "parameters": given.get("parameters") or {"a": {"value": 0.05}, "f": {"value": 2.0},
                                                            "p": {"value": 0.0}}}
    try:
        parametric["parameters"] = reconcile_parameters(parse_formula(parametric["formula"]), parametric["parameters"])
    except ValueError:
        if mode == "parametric":
            raise
    texture = _filled({"enabled": False, "monochrome": False, "resolution": [1024, 512], "levels": 4, "lr_decay": 1.0,
                       "views": 4}, cfg.get("texture"))
    w, h = (int(v) for v in texture["resolution"])
    if not (8 <= w <= 8192 and 8 <= h <= 8192):
        raise ValueError(f"Bad texture resolution {w}x{h}")
    shading = _filled({"mode": "lit", "azimuth": -30.0, "elevation": 30.0, "ambient": 0.35}, cfg.get("shading"))
    mesh = _filled({"rows": 256, "segments": 128}, cfg.get("mesh"))
    return {
        "mode": mode, "radius": _clamped(cfg["radius"], 1e-3, 10.0), "optimize_radius": bool(cfg["optimize_radius"]),
        "height": _clamped(cfg["height"], 1e-3, 10.0), "softness": _clamped(cfg["softness"], 0.05, 100.0),
        "shape_lr": max(0.0, float(cfg["shape_lr"])), "optimize_profile": bool(cfg["optimize_profile"]),
        "canvas_angle": float(cfg["canvas_angle"]),
        "points": points, "waves": waves, "parametric": parametric,
        "texture": {"enabled": bool(texture["enabled"]), "monochrome": bool(texture["monochrome"]),
                    "resolution": [w, h], "levels": int(_clamped(texture["levels"], 1, 8)),
                    "lr_decay": _clamped(texture["lr_decay"], 0.05, 1.0), "views": int(_clamped(texture["views"], 1, 32))},
        "color": _hex(cfg["color"], "#bbbbbb"), "optimize_color": bool(cfg["optimize_color"]),
        "shading": {"mode": _choice(shading["mode"], SHADINGS, "shading"), "azimuth": float(shading["azimuth"]),
                    "elevation": float(shading["elevation"]), "ambient": _clamped(shading["ambient"], 0.0, 1.0)},
        "mesh": {"rows": int(_clamped(mesh["rows"], 2, 1024)), "segments": int(_clamped(mesh["segments"], 8, 1024))},
    }


def schema() -> dict:
    """The choices, for the page."""
    return {"modes": MODES, "interpolations": INTERPOLATIONS, "waves": WAVES, "shadings": SHADINGS,
            "functions": list(FUNCTIONS), "reserved": RESERVED, "max_points": MAX_POINTS,
            "parameter_default": PARAMETER_DEFAULT}


# ── Profile ────────────────────────────────────────────────────────────────────

class Profile:
    """The radius along t, per the config's mode. Its optimized tensors are leaves on `device`."""

    def __init__(self, cfg: dict, device):
        self.cfg = cfg
        self.mode = cfg["mode"]
        free = cfg["optimize_profile"]
        self.base = torch.tensor(cfg["radius"], dtype=torch.float32, device=device).requires_grad_(
            free and cfg["optimize_radius"])
        self.tensors: list[torch.Tensor] = []  # what params() optimizes, besides the base radius
        if self.mode == "points":
            self.offsets = torch.tensor([p["offset"] for p in cfg["points"]], dtype=torch.float32,
                                        device=device).requires_grad_(free)
            self.kinds = torch.tensor([INTERPOLATIONS.index(p["interpolation"]) for p in cfg["points"]], device=device)
            self.tensors = [self.offsets]
        elif self.mode == "waves":
            self.waves = []
            for w in cfg["waves"]:
                k = torch.arange(1, w["components"] + 1, dtype=torch.float32, device=device)
                amplitude = (w["amplitude"] / k).requires_grad_(free)
                phase = torch.full_like(k, w["phase"]).requires_grad_(free)
                frequency = (w["frequency"] * k).requires_grad_(free and w["optimize_frequency"])
                self.waves.append((w["wave"], amplitude, phase, frequency))
                self.tensors += [amplitude, phase, frequency]
        else:
            self.formula = parse_formula(cfg["parametric"]["formula"])
            self.parameters = {name: torch.tensor(p["value"], dtype=torch.float32, device=device).requires_grad_(
                                   free and p["optimize"])
                               for name, p in cfg["parametric"]["parameters"].items()}
            self.tensors = list(self.parameters.values())
        self.tensors = [t for t in [self.base, *self.tensors] if t.requires_grad]

    def _spans(self, t: torch.Tensor):
        """Each t's span between points: the four Catmull-Rom points around it, how far along it is,
        its interpolation, and t's scale (spans per unit of t)."""
        v, n = self.offsets, len(self.offsets)
        x = t.clamp(0, 1) * (n - 1)
        i = x.floor().long().clamp(0, n - 2)
        return (v[(i - 1).clamp_min(0)], v[i], v[i + 1], v[(i + 2).clamp_max(n - 1)]), x - i, self.kinds[i], n - 1

    def offset(self, t: torch.Tensor) -> torch.Tensor:
        """(N,) the radial offset along t (N,)."""
        if self.mode == "points":
            (p0, p1, p2, p3), f, kinds, _ = self._spans(t)
            linear = p1 + (p2 - p1) * f
            nearest = torch.where(f < 0.5, p1, p2)
            spline = 0.5 * (2 * p1 + (p2 - p0) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2
                            + (3 * p1 - p0 - 3 * p2 + p3) * f ** 3)
            return torch.where(kinds == 0, linear, torch.where(kinds == 1, nearest, spline))
        if self.mode == "waves":
            total = t.new_zeros(t.shape)
            for name, amplitude, phase, frequency in self.waves:
                theta = 2 * math.pi * frequency[:, None] * t + phase[:, None]
                total = total + (amplitude[:, None] * wave_function(name, theta)).sum(0)
            return total
        return self.formula(t, self.parameters)

    def offset_slope(self, t: torch.Tensor) -> torch.Tensor:
        """(N,) d offset / dt. Exact for points (a nearest-neighbour step is flat, its jump is a ledge
        the camera sees edge on), a central difference otherwise."""
        if self.mode == "points":
            (p0, p1, p2, p3), f, kinds, scale = self._spans(t)
            linear = p2 - p1
            spline = 0.5 * ((p2 - p0) + 2 * (2 * p0 - 5 * p1 + 4 * p2 - p3) * f
                            + 3 * (3 * p1 - p0 - 3 * p2 + p3) * f ** 2)
            return scale * torch.where(kinds == 0, linear, torch.where(kinds == 1, torch.zeros_like(f), spline))
        below, above = (t - SLOPE_DT).clamp(0, 1), (t + SLOPE_DT).clamp(0, 1)
        return (self.offset(above) - self.offset(below)) / (above - below).clamp_min(_EPS)

    def radius(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,), (N,) the radius along t, always > 0, and d radius / dt."""
        raw = RADIUS_SHARPNESS * (self.base + self.offset(t))
        return F.softplus(raw) / RADIUS_SHARPNESS, torch.sigmoid(raw) * self.offset_slope(t)

    def values(self) -> dict:
        """The current parameters, flat: {name: a number or a list}."""
        with torch.no_grad():
            out = {"radius": round(self.base.item(), 5)}
            if self.mode == "points":
                out["offsets"] = [round(v, 5) for v in self.offsets.tolist()]
            elif self.mode == "waves":
                for i, (_, amplitude, phase, frequency) in enumerate(self.waves):
                    out[f"wave {i + 1} amplitude"] = [round(v, 5) for v in amplitude.tolist()]
                    out[f"wave {i + 1} frequency"] = [round(v, 5) for v in frequency.tolist()]
                    out[f"wave {i + 1} phase"] = [round(v, 5) for v in phase.tolist()]
            else:
                out.update({name: round(p.item(), 5) for name, p in self.parameters.items()})
        return out


# ── Rendering ──────────────────────────────────────────────────────────────────

@dataclass
class Geometry:
    """Per pixel of the canvas; the same from every angle but for u."""
    mask: torch.Tensor     # (1, 1, H, W)
    normals: torch.Tensor  # (1, 3, H, W) unit, x right, y up, z to the camera
    depth: torch.Tensor    # (1, 1, H, W) towards the camera from the axis, canvas short sides
    shade: torch.Tensor    # (1, 1, H, W) the lighting's multiplier
    azimuth: torch.Tensor  # (H, W) radians from the front, -pi/2..pi/2
    v: torch.Tensor        # (H, W) t: 0 bottom, 1 top


def light_direction(azimuth: float, elevation: float) -> tuple[float, float, float]:
    """A unit vector in the camera's frame (x right, y up, z to the camera), degrees."""
    a, e = math.radians(azimuth), math.radians(elevation)
    return math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)


class RevolutionModel:
    """A profile on a canvas of `size` (width, height)."""

    def __init__(self, cfg: dict, size: tuple[int, int], device):
        self.cfg = cfg
        self.profile = Profile(cfg, device)
        w, h = size
        short = min(w, h)
        self.softness = cfg["softness"] / short
        self.height = cfg["height"]
        self.x = ((torch.arange(w, device=device) + 0.5 - w / 2) / short)[None]  # (1, W)
        y = (h / 2 - torch.arange(h, device=device) - 0.5) / short               # (H,) up
        self.t = (y / self.height + 0.5).clamp(0, 1)
        self.band = torch.sigmoid((self.height / 2 - y.abs()) / self.softness)[:, None]  # (H, 1)
        self.light = torch.tensor(light_direction(cfg["shading"]["azimuth"], cfg["shading"]["elevation"]),
                                  dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.ambient = cfg["shading"]["ambient"]
        self.lit = cfg["shading"]["mode"] == "lit"

    def geometry(self) -> Geometry:
        r, dr_dt = self.profile.radius(self.t)
        r, dr_dy = r[:, None], (dr_dt / self.height)[:, None]  # (H, 1)
        # Past the silhouette s stops: the mask has hidden those pixels, and asin's slope stays finite.
        s = (self.x / r).clamp(-0.999, 0.999)
        c = torch.sqrt(1 - s ** 2)
        mask = (torch.sigmoid((r - self.x.abs()) / self.softness) * self.band)[None, None]
        normals = F.normalize(torch.stack([s, -dr_dy.expand_as(s), c])[None], dim=1)
        shade = mask.new_ones(mask.shape)
        if self.lit:
            diffuse = (normals * self.light).sum(1, keepdim=True).clamp_min(0)
            shade = self.ambient + (1 - self.ambient) * diffuse
        return Geometry(mask, normals, (r * c)[None, None], shade, torch.asin(s),
                        self.t[:, None].expand_as(s))


def sample_texture(texture: torch.Tensor, geometry: Geometry, angle: float) -> torch.Tensor:
    """(1, C, H, W) the (1, C, TH, TW) texture on the canvas with the surface turned by `angle`
    degrees. u wraps around the seam; image top = t 1."""
    width = texture.shape[-1]
    u = torch.remainder(geometry.azimuth / (2 * math.pi) + angle / 360 + 0.5, 1.0)
    grid = torch.stack([2 * (u * width + 1) / (width + 2) - 1, 1 - 2 * geometry.v], dim=-1)[None]
    wrapped = F.pad(texture, (1, 1, 0, 0), mode="circular")
    return F.grid_sample(wrapped, grid, mode="bilinear", padding_mode="border", align_corners=False)


def compose(geometry: Geometry, albedo: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    """(1, 3, H, W) the shaded surface over a (1, 3, H, W) background."""
    return albedo * geometry.shade * geometry.mask + background * (1 - geometry.mask)


def geometry_maps(geometry: Geometry, radius: float) -> dict[str, torch.Tensor]:
    """(1, C, H, W) images in [0, 1], by name: mask, normals, depth (over `radius`), uv at angle 0
    (red = u, green = v)."""
    mask = geometry.mask
    u = geometry.azimuth / (2 * math.pi) + 0.5
    uv = torch.stack([u, geometry.v, torch.zeros_like(u)])[None]
    return {"mask": mask, "normals": (geometry.normals * 0.5 + 0.5) * mask,
            "depth": (geometry.depth / max(radius, 1e-6)).clamp(0, 1) * mask, "uv": uv * mask}


def preview(cfg: dict | None, size: tuple[int, int], color: torch.Tensor, background: torch.Tensor,
            texture: torch.Tensor | None = None) -> torch.Tensor:
    """(1, 3, H, W) the surface as a run starts: with a texture, at three angles side by side. On the
    CPU, whatever the run's device: it costs next to nothing, and leaves a running run's GPU alone."""
    cfg = normalize_config(cfg)
    model = RevolutionModel(cfg, size, background.device)
    with torch.no_grad():
        geometry = model.geometry()
        if texture is None:
            return compose(geometry, color.view(1, 3, 1, 1), background)
        angles = [cfg["canvas_angle"] + a for a in (0, 120, 240)]
        tiles = [compose(geometry, sample_texture(texture, geometry, a).expand(-1, 3, -1, -1), background)
                 for a in angles]
    return torch.cat(tiles, dim=-1)


# ── Export ─────────────────────────────────────────────────────────────────────

def mesh_arrays(profile: Profile, rows: int, segments: int) -> dict[str, np.ndarray]:
    """The surface as a grid of (rows + 1) x (segments + 1) vertices, the last column repeating the
    first with u = 1 so the texture closes at the seam. Axis Z, centred on the origin, the front
    (u = 1/2) towards -Y, so that Rhino's Front view shows what the canvas shows at angle 0; canvas
    units. -> {vertices (V, 3), normals (V, 3), uvs (V, 2), quads (F, 4), counter-clockwise seen
    from outside}."""
    height = profile.cfg["height"]
    with torch.no_grad():
        t = torch.linspace(0, 1, rows + 1, device=profile.base.device)
        r, dr_dt = profile.radius(t)
    r, slope = r.cpu().numpy().astype(np.float64), dr_dt.cpu().numpy().astype(np.float64) / height
    u = np.linspace(0, 1, segments + 1)
    angle = 2 * np.pi * (u - 0.5)
    sin, cos = np.sin(angle)[None], np.cos(angle)[None]
    vertices = np.stack([r[:, None] * sin, -r[:, None] * cos,
                         np.broadcast_to((t.cpu().numpy()[:, None] - 0.5) * height, (rows + 1, segments + 1))], -1)
    normals = np.stack([sin * np.ones((rows + 1, 1)), -cos * np.ones((rows + 1, 1)),
                        np.broadcast_to(-slope[:, None], (rows + 1, segments + 1))], -1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    uvs = np.stack(np.meshgrid(u, t.cpu().numpy()), -1)
    index = np.arange((rows + 1) * (segments + 1)).reshape(rows + 1, segments + 1)
    quads = np.stack([index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]], -1)
    return {"vertices": vertices.reshape(-1, 3), "normals": normals.reshape(-1, 3), "uvs": uvs.reshape(-1, 2),
            "quads": quads.reshape(-1, 4)}


def write_rhino(path: Path, profile: Profile, color: tuple[int, int, int], texture: Path | None = None,
                notes: dict[str, str] | None = None) -> None:
    """A .3dm with the surface as a mesh on a layer of its own, its texture coordinates, and a material:
    `color`, or the image `texture` both referenced by its path and embedded in the file, so
    the file still shows it after moving. notes: user strings on the mesh."""
    import rhino3dm as rhino
    arrays = mesh_arrays(profile, profile.cfg["mesh"]["rows"], profile.cfg["mesh"]["segments"])
    mesh = rhino.Mesh()
    for x, y, z in arrays["vertices"].tolist():
        mesh.Vertices.Add(x, y, z)
    for x, y, z in arrays["normals"].tolist():
        mesh.Normals.Add(x, y, z)
    for a, b, c, d in arrays["quads"].tolist():
        mesh.Faces.AddFace(a, b, c, d)
    # rhino3dm has no way to add coordinates, only to compute them from a mapping and then replace them.
    # A plane mapping: a cylinder's would split the vertices along its own seam.
    unit = rhino.Interval(0, 1)
    mesh.SetTextureCoordinates(rhino.TextureMapping.CreatePlaneMapping(rhino.Plane.WorldXY(), unit, unit, unit),
                               rhino.Transform.Identity(), False)
    if not len(mesh.TextureCoordinates) == len(mesh.Vertices) == len(arrays["uvs"]):
        raise RuntimeError(f"rhino3dm made {len(mesh.TextureCoordinates)} texture coordinates for "
                           f"{len(mesh.Vertices)} vertices, expected {len(arrays['uvs'])}")
    for i, (u, v) in enumerate(arrays["uvs"].tolist()):
        mesh.TextureCoordinates[i] = rhino.Point2f(u, v)

    model = rhino.File3dm()
    material = rhino.Material()
    material.Name = "revolution"
    material.DiffuseColor = (255, 255, 255, 255) if texture is not None else (*color, 255)
    if texture is not None:
        path_text = str(Path(texture).resolve())
        material.SetBitmapTexture(path_text)
        model.EmbeddedFiles.Add(rhino.EmbeddedFile.Read(path_text))
    attributes = rhino.ObjectAttributes()
    attributes.Name = "revolution"
    attributes.LayerIndex = model.Layers.AddLayer("revolution", (*color, 255))
    attributes.MaterialSource = rhino.ObjectMaterialSource.MaterialFromObject
    attributes.MaterialIndex = model.Materials.Add(material)
    attributes.ColorSource = rhino.ObjectColorSource.ColorFromObject
    attributes.ObjectColor = (*color, 255)
    for key, value in (notes or {}).items():
        attributes.SetUserString(key, str(value))
    model.Objects.AddMesh(mesh, attributes)
    if not model.Write(str(path), 8):
        raise RuntimeError(f"Could not write {path}")
