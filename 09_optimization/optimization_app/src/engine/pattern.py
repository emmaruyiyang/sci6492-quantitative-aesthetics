"""Shape patterns: closed Rhino curves as polygons, rasterized into per-shape masks
and composited into images from [[shape, color], ...] descriptors.

`Pattern` holds everything needed to render (shapes + canvas/colour settings) and
round-trips through a plain dict. `PatternDataset` is a folder written by
pattern_generator.py: dataset.json (pattern config + samples) and embedding caches,
with or without the PNGs.
"""
import json
from collections.abc import Callable
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from shared.models.model_central import ModelCentral, _emb_filename, print_progress
from shared.models.model_types import ModelID

Polygon = list[list[float]]
Pairs = list[list[int]]

_SUBPIXEL_BITS = 4


# ── Rhino curves -> polygons ───────────────────────────────────────────────────

def _curve_points(curve, samples_per_span: int) -> list[tuple[float, float]]:
    """Tessellate a curve to XY points (endpoint included)."""
    import rhino3dm as rhino
    if isinstance(curve, rhino.PolyCurve):
        pts = []
        for i in range(curve.SegmentCount):
            seg = _curve_points(curve.SegmentCurve(i), samples_per_span)
            pts.extend(seg[1:] if pts else seg)
        return pts
    polyline = curve.TryGetPolyline()
    if polyline is not None:
        return [(polyline[i].X, polyline[i].Y) for i in range(polyline.Count)]
    n = max(2, curve.SpanCount * samples_per_span)
    t0, t1 = curve.Domain.T0, curve.Domain.T1
    return [(p.X, p.Y) for p in (curve.PointAt(t0 + (t1 - t0) * i / n) for i in range(n + 1))]


def _read_3dm(path: str | Path):
    import rhino3dm as rhino
    model = rhino.File3dm.Read(str(path))
    if model is None:
        raise RuntimeError(f"Could not read {path}")
    return model


def closed_curve_objects(model, samples_per_span: int = 16) -> list[tuple[object, Polygon]]:
    """The closed curves of a read .3dm that make a shape (3+ distinct points), each with
    its open point list (no repeated closing point), in file order -- the order of a
    Pattern's shapes and masks."""
    import rhino3dm as rhino
    out = []
    for obj in model.Objects:
        geom = obj.Geometry
        if not (isinstance(geom, rhino.Curve) and geom.IsClosed):
            continue
        pts = []
        for p in _curve_points(geom, samples_per_span):
            if not pts or p != pts[-1]:
                pts.append(p)
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts.pop()
        if len(pts) >= 3:
            out.append((obj, [[float(x), float(y)] for x, y in pts]))
    return out


def load_closed_polygons(path: str | Path, samples_per_span: int = 16) -> list[Polygon]:
    """Every closed curve in a .3dm as an open point list (no repeated closing point)."""
    return [polygon for _, polygon in closed_curve_objects(_read_3dm(path), samples_per_span)]


def _cross(a, b, c) -> float:
    """z of (b - a) x (c - b): > 0 where a -> b -> c turns counter-clockwise."""
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])


def triangulate_polygon(points: Polygon) -> list[tuple[int, int, int]]:
    """Ear clipping of a counter-clockwise simple polygon (concave is fine): triangles of
    indices into points, counter-clockwise. A vertex left with no ear (collinear leftovers, a
    self-intersecting outline) is dropped, so this always ends."""
    area = abs(sum(_cross((0.0, 0.0), points[i - 1], points[i]) for i in range(len(points))))
    eps = area * 1e-12
    remaining = list(range(len(points)))
    triangles = []
    while len(remaining) > 2:
        n = len(remaining)
        corners = [(remaining[i - 1], remaining[i], remaining[(i + 1) % n]) for i in range(n)]
        for ia, ib, ic in corners:
            a, b, c = points[ia], points[ib], points[ic]
            if _cross(a, b, c) > eps and not any(
                    min(_cross(a, b, points[j]), _cross(b, c, points[j]), _cross(c, a, points[j])) >= -eps
                    for j in remaining if j not in (ia, ib, ic) and points[j] not in (a, b, c)):
                triangles.append((ia, ib, ic))
                remaining.remove(ib)
                break
        else:
            flattest = min(corners, key=lambda t: abs(_cross(points[t[0]], points[t[1]], points[t[2]])))
            if _cross(points[flattest[0]], points[flattest[1]], points[flattest[2]]) > eps:
                triangles.append(flattest)
            remaining.remove(flattest[1])
    return triangles


def extrusion_mesh(polygon: Polygon, z: float, height: float, crease_degrees: float = 30.0):
    """A rhino3dm Mesh: the polygon at z extruded up by height, closed by two caps with their
    own vertices, so the edge between a cap and the sides stays sharp. The sides share vertices
    around the outline (a sampled curve shades smooth) except at corners turning by more than
    crease_degrees, which stay sharp too."""
    import rhino3dm as rhino
    points = [tuple(p) for p in polygon]
    if sum(_cross((0.0, 0.0), points[i - 1], points[i]) for i in range(len(points))) < 0:
        points.reverse()  # counter-clockwise: every face's normal points out
    mesh = rhino.Mesh()

    def add_ring(level: float) -> list[int]:
        first = len(mesh.Vertices)
        for x, y in points:
            mesh.Vertices.Add(x, y, level)
        return list(range(first, first + len(points)))

    triangles = triangulate_polygon(points)
    bottom, top = add_ring(z), add_ring(z + height)
    for a, b, c in triangles:
        mesh.Faces.AddFace(bottom[c], bottom[b], bottom[a])
        mesh.Faces.AddFace(top[a], top[b], top[c])

    # Each side vertex i: `ending` for the side arriving at i, `starting` for the side leaving it.
    n = len(points)
    crease = np.cos(np.deg2rad(crease_degrees))
    side_bottom, side_top = add_ring(z), add_ring(z + height)
    ending_bottom, ending_top = list(side_bottom), list(side_top)
    for i in range(n):
        p, q, r = (np.asarray(points[j], dtype=np.float64) for j in (i - 1, i, (i + 1) % n))
        before, after = q - p, r - q
        if before @ after < crease * np.linalg.norm(before) * np.linalg.norm(after):
            x, y = points[i]
            ending_bottom[i], ending_top[i] = mesh.Vertices.Add(x, y, z), mesh.Vertices.Add(x, y, z + height)
    for i in range(n):
        j = (i + 1) % n
        mesh.Faces.AddFace(side_bottom[i], ending_bottom[j], ending_top[j], side_top[i])
    mesh.Normals.ComputeNormals()
    return mesh


def save_colored_rhino(src: str | Path, dst: str | Path, colors, samples_per_span: int = 16,
                       extrusion_heights=None, extrusion_layer: str = "extrusions") -> None:
    """A copy of the .3dm at src with every shape's curve (Pattern order) in its colour:
    colors[i] is the RGB (0..255) of shape i, set as the object colour (and as the user
    string "rgb"). extrusion_heights (model units per shape): also a new layer with each curve
    extruded up from its own z, as a capped mesh (extrusion_mesh) in the same colour, by object
    and by vertex, so it shows in every display mode. Curves are taken as flat in XY."""
    import rhino3dm as rhino
    model = _read_3dm(src)
    shapes = closed_curve_objects(model, samples_per_span)
    if len(shapes) != len(colors):
        raise ValueError(f"{len(colors)} colours for {len(shapes)} shapes in {src}")
    if extrusion_heights is not None and len(shapes) != len(extrusion_heights):
        raise ValueError(f"{len(extrusion_heights)} heights for {len(shapes)} shapes in {src}")
    layer = None
    if extrusion_heights is not None:
        layer = model.Layers.AddLayer(extrusion_layer, (128, 128, 128, 255))
    for i, ((obj, polygon), rgb) in enumerate(zip(shapes, colors)):
        r, g, b = (int(round(v)) for v in rgb)
        attributes = obj.Attributes
        attributes.ColorSource = rhino.ObjectColorSource.ColorFromObject
        attributes.ObjectColor = (r, g, b, 255)
        attributes.SetUserString("rgb", f"{r},{g},{b}")
        if extrusion_heights is not None:
            height = float(extrusion_heights[i])
            mesh = extrusion_mesh(polygon, obj.Geometry.PointAtStart.Z, height)
            for _ in range(len(mesh.Vertices)):
                mesh.VertexColors.Add(r, g, b)
            mesh_attributes = rhino.ObjectAttributes()
            mesh_attributes.LayerIndex = layer
            mesh_attributes.ColorSource = rhino.ObjectColorSource.ColorFromObject
            mesh_attributes.ObjectColor = (r, g, b, 255)
            mesh_attributes.SetUserString("rgb", f"{r},{g},{b}")
            mesh_attributes.SetUserString("height", f"{height:.6g}")
            model.Objects.AddMesh(mesh, mesh_attributes)
    if not model.Write(str(dst), 7):
        raise RuntimeError(f"Could not write {dst}")


def _bbox(polygons: list[Polygon]) -> list[float]:
    pts = np.concatenate([np.asarray(p) for p in polygons])
    (xmin, ymin), (xmax, ymax) = pts.min(axis=0), pts.max(axis=0)
    return [float(xmin), float(ymin), float(xmax), float(ymax)]


# ── Pattern ────────────────────────────────────────────────────────────────────

class Pattern:
    """Shapes plus render settings. Colours are RGB; img_size is (width, height)."""

    CONFIG_KEYS = ("img_size", "img_border", "antialias", "bg_color", "fg_colors", "bbox", "shapes")

    def __init__(self, shapes: list[Polygon], img_size=(256, 256), img_border: int = 20,
                 antialias: bool = True, bg_color=(255, 255, 255), fg_colors=((0, 0, 0),),
                 bbox: list[float] | None = None):
        if not shapes:
            raise ValueError("Pattern needs at least one shape")
        self.shapes = shapes
        self.img_size = tuple(img_size)
        self.img_border = img_border
        self.antialias = antialias
        self.bg_color = tuple(bg_color)
        self.fg_colors = [tuple(c) for c in fg_colors]
        self.bbox = list(bbox) if bbox is not None else _bbox(shapes)

    @classmethod
    def from_rhino(cls, path: str | Path, samples_per_span: int = 16, **render_settings) -> 'Pattern':
        shapes = load_closed_polygons(path, samples_per_span)
        if not shapes:
            raise RuntimeError(f"No closed curves in {path}")
        return cls(shapes, **render_settings)

    @classmethod
    def from_config(cls, config: dict) -> 'Pattern':
        return cls(**{k: config[k] for k in cls.CONFIG_KEYS})

    def to_config(self) -> dict:
        return {
            "img_size": list(self.img_size),
            "img_border": self.img_border,
            "antialias": self.antialias,
            "bg_color": list(self.bg_color),
            "fg_colors": [list(c) for c in self.fg_colors],
            "bbox": self.bbox,
            "shapes": self.shapes,
        }

    @property
    def pixels_per_unit(self) -> float:
        """Canvas pixels per model unit: the total bbox letterboxed into the canvas inside img_border."""
        w, h = self.img_size
        xmin, ymin, xmax, ymax = self.bbox
        return min((w - 2 * self.img_border) / max(xmax - xmin, 1e-12),
                   (h - 2 * self.img_border) / max(ymax - ymin, 1e-12))

    def pixel_polygons(self) -> list[np.ndarray]:
        """Shapes as float64 (K, 2) image-space points: the total bbox letterboxed
        into the canvas inside img_border, Rhino's Y-up flipped to image Y-down."""
        w, h = self.img_size
        xmin, ymin, xmax, ymax = self.bbox
        scale = self.pixels_per_unit
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        polys = []
        for poly in self.shapes:
            pts = np.asarray(poly, dtype=np.float64)
            polys.append(np.stack([(pts[:, 0] - cx) * scale + w / 2,
                                   h / 2 - (pts[:, 1] - cy) * scale], axis=1))
        return polys

    @cached_property
    def masks(self) -> list[np.ndarray]:
        """Per-shape float32 alpha masks (H, W) in [0, 1], see pixel_polygons."""
        w, h = self.img_size
        line_type = cv2.LINE_AA if self.antialias else cv2.LINE_8

        masks = []
        for pts in self.pixel_polygons():
            fixed = np.round(pts * (1 << _SUBPIXEL_BITS)).astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [fixed], 255, lineType=line_type, shift=_SUBPIXEL_BITS)
            masks.append(mask.astype(np.float32) / 255.0)
        return masks

    def render(self, pairs: Pairs) -> np.ndarray:
        """RGB uint8 (H, W, 3) image for one [[shape, color], ...] descriptor, drawn in order."""
        w, h = self.img_size
        img = np.empty((h, w, 3), dtype=np.float32)
        img[:] = self.bg_color
        for shape, color in pairs:
            alpha = self.masks[shape][..., None]
            img = img * (1 - alpha) + np.asarray(self.fg_colors[color], dtype=np.float32) * alpha
        return np.clip(np.round(img), 0, 255).astype(np.uint8)

    def render_pil(self, pairs: Pairs) -> Image.Image:
        return Image.fromarray(self.render(pairs))

    def render_outlines(self, thickness: int = 1) -> np.ndarray:
        """RGB uint8 (H, W, 3) of every shape's outline in fg_colors[0] over bg_color."""
        w, h = self.img_size
        img = np.empty((h, w, 3), dtype=np.uint8)
        img[:] = self.bg_color
        fixed = [np.round(p * (1 << _SUBPIXEL_BITS)).astype(np.int32) for p in self.pixel_polygons()]
        cv2.polylines(img, fixed, True, tuple(int(c) for c in self.fg_colors[0]), thickness,
                      lineType=cv2.LINE_AA, shift=_SUBPIXEL_BITS)
        return img


# ── Generated dataset folder ───────────────────────────────────────────────────

class PatternDataset:
    """A folder holding dataset.json (pattern config + `samples`) and embedding caches.
    PNGs are optional; anything missing is rendered from the config."""

    CONFIG_FILE = "dataset.json"
    EMBED_BATCH = 32

    def __init__(self, folder: str | Path, config: dict):
        self.folder = Path(folder)
        self.config = config
        self.pattern = Pattern.from_config(config)
        self.samples: list[Pairs] = config["samples"]

    @classmethod
    def is_dataset(cls, folder: str | Path) -> bool:
        return (Path(folder) / cls.CONFIG_FILE).is_file()

    @classmethod
    def load(cls, folder: str | Path) -> 'PatternDataset':
        folder = Path(folder)
        return cls(folder, json.loads((folder / cls.CONFIG_FILE).read_text()))

    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / self.CONFIG_FILE).write_text(json.dumps(self.config, indent=1))

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def image_names(self) -> list[str] | None:
        """Saved PNG names in sample order, or None if images weren't saved."""
        return self.config.get("images")

    def sample_name(self, index: int) -> str:
        names = self.image_names
        return names[index] if names else f"sample_{index:06d}"

    def render(self, index: int) -> Image.Image:
        return self.pattern.render_pil(self.samples[index])

    # ── Signatures / growth ──────────────────────────────────────────────────

    @staticmethod
    def signature(pairs: Pairs) -> str:
        """Canonical id of a shape->colour assignment: order-independent, so the
        same drawing never gets stored twice. Shapes are composited in shape order."""
        return ",".join(f"{s}:{c}" for s, c in sorted((int(s), int(c)) for s, c in pairs))

    @cached_property
    def signatures(self) -> dict[str, int]:
        """signature -> sample index."""
        return {self.signature(p): i for i, p in enumerate(self.samples)}

    def add_samples(self, candidates: list[Pairs]) -> list[int]:
        """Append the candidates whose signature is new; returns their indices.
        Saves dataset.json (and the PNGs, when this dataset keeps them)."""
        added = []
        for pairs in candidates:
            pairs = sorted([int(s), int(c)] for s, c in pairs)
            sig = self.signature(pairs)
            if not pairs or sig in self.signatures:
                continue
            self.signatures[sig] = len(self.samples)
            added.append(len(self.samples))
            self.samples.append(pairs)
        if not added:
            return added

        names = self.image_names
        if names is not None:
            for i in added:
                names.append(f"{i:06d}.png")
                self.render(i).save(self.folder / names[i])
            (self.folder / "image_order.json").write_text(json.dumps(names))
        self.save()
        return added

    def ensure_embeddings(self, central: ModelCentral, model_id: ModelID,
                          progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        """(N, D) embeddings in sample order: loaded from the folder cache, or computed
        by rendering samples in memory and saved there. A cache shorter than the
        dataset (samples were added) is extended with just the missing rows.
        `progress(done, total)` is called after each batch, counting new rows only."""
        cache_file = self.folder / _emb_filename(model_id)
        emb = None
        if cache_file.exists():
            emb = np.load(str(cache_file))
            if len(emb) == len(self):
                return emb
            if len(emb) > len(self):
                print(f"{cache_file.name} has {len(emb)} rows for {len(self)} samples; rebuilding")
                emb = None

        first = 0 if emb is None else len(emb)
        todo = len(self) - first
        print(f"Embedding {todo} sample(s) with {model_id} ...", flush=True)
        parts = [] if emb is None else [emb]
        for start in range(first, len(self), self.EMBED_BATCH):
            stop = min(start + self.EMBED_BATCH, len(self))
            parts.append(central.embed_pil_images([self.render(i) for i in range(start, stop)], model_id))
            print_progress(stop - first, todo)
            if progress is not None:
                progress(stop - first, todo)
        emb = np.concatenate(parts, axis=0)
        np.save(str(cache_file), emb)
        print(f"Saved -> {cache_file.name}")
        return emb
