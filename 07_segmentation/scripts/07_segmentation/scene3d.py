"""Rhino files for the Segmentation Explorer's 3D model mode.

read_3dm()       the meshes of a .3dm file as plain arrays, for three.js in the
                 page. Meshes as they are; Breps and extrusions only if the
                 file has their render meshes (saved from Rhino after they were
                 shown shaded), rhino3dm can't mesh them itself. Everything is
                 kept in Rhino's coordinates (Z up): the page renders with Z up
                 too, so the cameras it records are in the model's coordinates.

frustum_3dm()    one rendered frame as Rhino geometry: the camera's frustum as
                 lines, the view vector, and a mask as a mesh with a color per
                 vertex on a plane across the frustum (and the frame itself on
                 a hidden layer, on the same plane). The camera is also saved
                 as a named view, so Rhino can look through it.

A camera is {position, target, up: [x, y, z], fov: vertical, in degrees,
aspect: width / height, near, far} as three.js's PerspectiveCamera has it.
"""

import os
import tempfile

import numpy as np
import rhino3dm

MAX_VERTICES = 3_000_000        # all meshes of a file together


def mesh_arrays(mesh: rhino3dm.Mesh) -> tuple[list, list, list | None]:
    """-> (vertices x, y, z, ..., triangles a, b, c, ..., colors r, g, b, ... 0..1 or None)"""
    vertices = [c for v in mesh.Vertices for c in (v.X, v.Y, v.Z)]
    triangles = []
    for i in range(len(mesh.Faces)):
        a, b, c, d = mesh.Faces[i]
        triangles += [a, b, c] if c == d else [a, b, c, a, c, d]    # quads in two triangles
    colors = None
    if len(mesh.VertexColors) == len(mesh.Vertices):
        colors = [round(x / 255, 4) for color in mesh.VertexColors for x in color[:3]]
    return vertices, triangles, colors


def render_meshes(geometry) -> list[rhino3dm.Mesh]:
    if isinstance(geometry, rhino3dm.Mesh):
        return [geometry]
    if isinstance(geometry, rhino3dm.Brep):
        meshes = [face.GetMesh(rhino3dm.MeshType.Any) for face in geometry.Faces]
        return [m for m in meshes if m is not None]
    if isinstance(geometry, rhino3dm.Extrusion):
        mesh = geometry.GetMesh(rhino3dm.MeshType.Any)
        return [mesh] if mesh is not None else []
    return []


def read_3dm(data: bytes) -> dict:
    """-> {"meshes": [{name, color [r, g, b] 0..1, vertices, triangles, colors}],
    "bounds": [[x, y, z] min, max], "notes": [...]}"""
    model = rhino3dm.File3dm.FromByteArray(data)
    if model is None:
        raise ValueError("not a Rhino file")
    layers = list(model.Layers)
    meshes, skipped, without_mesh, count = [], {}, 0, 0
    for obj in model.Objects:
        attributes, geometry = obj.Attributes, obj.Geometry
        layer = layers[attributes.LayerIndex] if 0 <= attributes.LayerIndex < len(layers) else None
        if (layer is not None and not layer.Visible) or not attributes.Visible:
            continue
        parts = render_meshes(geometry)
        if not parts:
            if isinstance(geometry, (rhino3dm.Brep, rhino3dm.Extrusion)):
                without_mesh += 1
            elif not isinstance(geometry, (rhino3dm.Curve, rhino3dm.Point, rhino3dm.TextDot)):
                kind = type(geometry).__name__
                skipped[kind] = skipped.get(kind, 0) + 1
            continue
        from_layer = attributes.ColorSource == rhino3dm.ObjectColorSource.ColorFromLayer
        color = layer.Color if from_layer and layer is not None else attributes.ObjectColor
        for mesh in parts:
            vertices, triangles, colors = mesh_arrays(mesh)
            count += len(vertices) // 3
            if count > MAX_VERTICES:
                raise ValueError(f"more than {MAX_VERTICES} vertices: too big to show")
            meshes.append({"name": attributes.Name or type(geometry).__name__,
                           "color": [round(c / 255, 4) for c in color[:3]],
                           "vertices": vertices, "triangles": triangles, "colors": colors})
    if not meshes:
        raise ValueError("no meshes in this file (Breps need their render meshes saved: "
                         "shade them in Rhino before saving)")
    points = np.concatenate([np.reshape(m["vertices"], (-1, 3)) for m in meshes])
    notes = [f"{len(meshes)} meshes, {count} vertices"]
    if without_mesh:
        notes.append(f"{without_mesh} Breps without render meshes left out")
    notes += [f"{n} {kind} left out" for kind, n in skipped.items()]
    return {"meshes": meshes, "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
            "notes": notes}


# ---------------------------------------------------------------------------
# Frustum
# ---------------------------------------------------------------------------

def camera_frame(camera: dict) -> tuple:
    """-> (position, forward, right, up) unit vectors, numpy"""
    position = np.array(camera["position"], float)
    forward = np.array(camera["target"], float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array(camera["up"], float))
    right /= np.linalg.norm(right)
    return position, forward, right, np.cross(right, forward)


def corners(camera: dict, distance: float) -> list[np.ndarray]:
    """The frustum's corners at a distance along the view: top left, top
    right, bottom right, bottom left"""
    position, forward, right, up = camera_frame(camera)
    h = distance * np.tan(np.radians(camera["fov"]) / 2)
    w = h * camera["aspect"]
    center = position + distance * forward
    return [center - w * right + h * up, center + w * right + h * up,
            center + w * right - h * up, center - w * right - h * up]


def grid_mesh(camera: dict, distance: float, colors: np.ndarray) -> rhino3dm.Mesh:
    """A planar grid across the frustum at a distance, one vertex per color:
    colors [rows, columns, 3] 0..255, row 0 at the top of the picture"""
    rows, columns = colors.shape[:2]
    top_left, top_right, _, bottom_left = corners(camera, distance)
    mesh = rhino3dm.Mesh()
    for r in range(rows):
        for c in range(columns):
            p = (top_left + (top_right - top_left) * c / max(columns - 1, 1)
                 + (bottom_left - top_left) * r / max(rows - 1, 1))
            mesh.Vertices.Add(*map(float, p))
    for r in range(rows - 1):
        for c in range(columns - 1):
            k = r * columns + c
            # counterclockwise seen from the camera: the normals face it
            mesh.Faces.AddFace(k, k + columns, k + columns + 1, k + 1)
    for r, g, b in colors.reshape(-1, 3).tolist():
        mesh.VertexColors.Add(int(r), int(g), int(b))
    mesh.Normals.ComputeNormals()
    return mesh


def add_layer(model: rhino3dm.File3dm, name: str, color: tuple, visible: bool = True) -> int:
    layer = rhino3dm.Layer()
    layer.Name = name
    layer.Color = (*color, 255)
    layer.Visible = visible
    return model.Layers.Add(layer)


def point(p) -> rhino3dm.Point3d:
    return rhino3dm.Point3d(*map(float, p))


def attributes(layer: int, name: str, **user_strings) -> rhino3dm.ObjectAttributes:
    a = rhino3dm.ObjectAttributes()
    a.LayerIndex = layer
    a.Name = name
    for key, value in user_strings.items():
        a.SetUserString(key, str(value))
    return a


def frustum_layers(model: rhino3dm.File3dm) -> dict:
    return {"lines": add_layer(model, "frustum", (40, 40, 40)),
            "view": add_layer(model, "view vector", (220, 40, 40)),
            "masks": add_layer(model, "mask", (42, 120, 214)),
            "frames": add_layer(model, "frame", (128, 128, 128), visible=False)}


def add_frustum(model: rhino3dm.File3dm, layers: dict, camera: dict, mask: np.ndarray,
                picture: np.ndarray | None, name: str, plane: float = 0.5, depth: float | None = None,
                **user_strings) -> None:
    """Into a model: frustum lines, view vector, mask mesh at `plane` (0 = at
    the camera, 1 = at the target) of the way to the target, the frame (on a
    hidden layer), the camera as a named view. depth: how far the frustum's
    lines reach (default: twice the distance to the target). user_strings:
    stored on every object (Rhino: Properties > Attribute User Text)."""
    position, forward, _, up = camera_frame(camera)
    target = np.array(camera["target"], float)
    distance = float(np.linalg.norm(target - position))
    depth = depth or 2 * distance
    at = max(plane, 0.01) * distance
    a = lambda layer, what: attributes(layers[layer], f"{name} {what}", **user_strings)

    far = corners(camera, depth)
    for i, c in enumerate(far):
        model.Objects.AddLine(point(position), point(c), a("lines", f"edge {i + 1}"))
        model.Objects.AddLine(point(c), point(far[(i + 1) % 4]), a("lines", "far end"))
    middle = corners(camera, at)
    for i, c in enumerate(middle):
        model.Objects.AddLine(point(c), point(middle[(i + 1) % 4]), a("lines", "mask plane"))
    model.Objects.AddLine(point(position), point(target), a("view", "view vector"))
    model.Objects.AddPoint(point(position), a("view", "camera"))
    model.Objects.AddPoint(point(target), a("view", "target"))
    model.Objects.AddMesh(grid_mesh(camera, at, mask), a("masks", "mask"))
    if picture is not None:
        model.Objects.AddMesh(grid_mesh(camera, at, picture), a("frames", "frame"))

    # the camera as a named view
    view_info = rhino3dm.ViewInfo()
    view_info.Name = name
    viewport = view_info.Viewport
    viewport.ChangeToPerspectiveProjection(distance, True, 50)
    viewport.SetCameraLocation(point(position))
    viewport.SetCameraDirection(rhino3dm.Vector3d(*map(float, forward)))
    viewport.SetCameraUp(rhino3dm.Vector3d(*map(float, up)))
    viewport.TargetPoint = point(target)
    near = max(camera.get("near", distance / 100), 1e-6)
    h = near * np.tan(np.radians(camera["fov"]) / 2)
    w = h * camera["aspect"]
    viewport.SetFrustum(-w, w, -h, h, near, max(camera.get("far", depth), near * 2))
    model.NamedViews.Add(view_info)


def frustum_3dm(camera: dict, mask: np.ndarray, picture: np.ndarray | None, name: str,
                plane: float = 0.5) -> bytes:
    """A .3dm file with one frame's frustum (add_frustum)"""
    model = rhino3dm.File3dm()
    add_frustum(model, frustum_layers(model), camera, mask, picture, name, plane)
    return to_bytes(model)


def frustums_3dm(frames: list[dict], plane: float = 0.5) -> bytes:
    """A .3dm file with the frustums of many frames: [{camera, mask, picture,
    name, value}]. value (the map's value in that frame, as in the page's
    graph) is stored as user text on its objects."""
    model = rhino3dm.File3dm()
    layers = frustum_layers(model)
    for f in frames:
        extra = {"value": f["value"]} if f.get("value") is not None else {}
        add_frustum(model, layers, f["camera"], f["mask"], f.get("picture"), f["name"], plane, **extra)
    return to_bytes(model)


def colored_model_3dm(meshes: list[dict], colors: list[np.ndarray], name: str) -> bytes:
    """A .3dm file with the meshes of read_3dm(), one color per vertex:
    colors [vertices, 3] 0..255 for every mesh"""
    model = rhino3dm.File3dm()
    layer = add_layer(model, name, (0, 0, 0))
    for m, c in zip(meshes, colors):
        mesh = rhino3dm.Mesh()
        for x, y, z in np.reshape(m["vertices"], (-1, 3)).tolist():
            mesh.Vertices.Add(x, y, z)
        for a, b, d in np.reshape(m["triangles"], (-1, 3)).tolist():
            mesh.Faces.AddFace(a, b, d)
        for r, g, b in np.asarray(c, int).reshape(-1, 3).tolist():
            mesh.VertexColors.Add(r, g, b)
        mesh.Normals.ComputeNormals()
        model.Objects.AddMesh(mesh, attributes(layer, m["name"]))
    return to_bytes(model)


def to_bytes(model: rhino3dm.File3dm) -> bytes:
    """rhino3dm writes files only: through a temporary one"""
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "model.3dm")
    try:
        if not model.Write(path, 8):
            raise RuntimeError("rhino3dm couldn't write the file")
        with open(path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(path):
            os.remove(path)
        os.rmdir(folder)
