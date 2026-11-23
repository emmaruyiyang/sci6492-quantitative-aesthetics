"""Isosurfaces of scalar volumes, for the 3D fields' exports (VOXELS' density, GAUSSIANS_3D's
splats summed on a grid): VTK's marching cubes, the result as numpy arrays and Rhino meshes with
vertex colours.

A volume is a (D, H, W) array of values at voxel centres -- z, y, x, x fastest -- spanning the box
lo..hi (world (x, y, z), the voxels' outer faces). It is padded by one voxel below the level all
round first, so a surface that reaches the box's sides is closed there.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Isosurface:
    vertices: np.ndarray        # (V, 3) world
    normals: np.ndarray         # (V, 3) unit, outwards (towards lower values)
    triangles: np.ndarray       # (F, 3)
    colors: np.ndarray | None = None  # (V, 3) uint8

    def __len__(self) -> int:
        return len(self.triangles)


def marching_cubes(values: np.ndarray, level: float, lo, hi) -> Isosurface:
    """The isosurface at `level` of a (D, H, W) volume over the box lo..hi (see the module)."""
    import vtk
    from vtkmodules.util import numpy_support

    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d, h, w = values.shape
    spacing = (hi - lo) / np.array([w, h, d], dtype=np.float64)
    below = min(float(values.min()), level) - 1.0
    padded = np.pad(values, 1, constant_values=below)
    image = vtk.vtkImageData()
    image.SetDimensions(w + 2, h + 2, d + 2)
    image.SetSpacing(*spacing)
    image.SetOrigin(*(lo + spacing / 2 - spacing))  # voxel centres, one padding voxel out
    scalars = numpy_support.numpy_to_vtk(padded.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    scalars.SetName("values")
    image.GetPointData().SetScalars(scalars)

    cubes = vtk.vtkMarchingCubes()
    cubes.SetInputData(image)
    cubes.SetValue(0, level)
    cubes.ComputeNormalsOn()
    cubes.Update()
    out = cubes.GetOutput()
    if out.GetNumberOfPoints() == 0:
        return Isosurface(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    vertices = numpy_support.vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
    triangles = numpy_support.vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
    normals = numpy_support.vtk_to_numpy(out.GetPointData().GetNormals()).astype(np.float64)
    return Isosurface(vertices, normals, triangles)


def sample_colors(colors: np.ndarray, points: np.ndarray, lo, hi) -> np.ndarray:
    """(N, 3) uint8: a (3, D, H, W) colour volume in [0, 1] over the box lo..hi, trilinearly at world
    points (N, 3)."""
    import torch
    import torch.nn.functional as F
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    grid = torch.from_numpy(2 * (points - lo) / (hi - lo) - 1).float().view(1, 1, 1, -1, 3)
    volume = torch.from_numpy(np.ascontiguousarray(colors, dtype=np.float32))[None]
    sampled = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return (sampled[0, :, 0, 0].T.clamp(0, 1) * 255).round().byte().numpy()


def to_rhino(surface: Isosurface):
    """A rhino3dm Mesh of an isosurface, with its normals and vertex colours."""
    import rhino3dm as rhino
    mesh = rhino.Mesh()
    for x, y, z in surface.vertices.tolist():
        mesh.Vertices.Add(x, y, z)
    for x, y, z in surface.normals.tolist():
        mesh.Normals.Add(x, y, z)
    for a, b, c in surface.triangles.tolist():
        mesh.Faces.AddFace(a, b, c)
    if surface.colors is not None:
        for r, g, b in surface.colors.tolist():
            mesh.VertexColors.Add(r, g, b)
    return mesh
