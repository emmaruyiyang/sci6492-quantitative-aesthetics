import torch

import numpy as np
import vtk
import vtkmodules.util.numpy_support as vtk_np
import os
import json


from .vol import *
from .vol_types import *
from .vol_utils import *

from enum import Enum

class RESAMPLEPADDING(Enum):
    ZEROS = 'zeros'
    BORDER = 'border'
    REFLECTION = 'reflection'

class VolumeFieldTypeHint(Enum):
    SDF = 'sdf'
    DENSITY = 'density'
    VEC3 = 'vec3'
    COLOR = 'color'

class VolSimpleMesh:
    def __init__ (self, verts=None, faces=None):
        self.verts = np.array(verts, dtype=np.float32) if verts is not None else np.array([], dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32) if faces is not None else np.array([], dtype=np.int32)

    def to_rhino_mesh(self):
        import rhino3dm as r3d
        mesh = r3d.Mesh()
        for v in self.verts:
            mesh.Vertices.Add(v[0], v[1], v[2])
        for f in self.faces:
            mesh.Faces.AddFace(f[0], f[1], f[2])
        return mesh

class VolumeField:
    @staticmethod
    def fromRGB(r : float, g : float, b : float, resolution : Idx3, device : torch.device = Vol.device):
        color = torch.zeros(3, resolution[0], resolution[1], resolution[2], dtype=torch.float32, device=device)
        color[0, ::] = r
        color[1, ::] = g
        color[2, ::] = b
        return VolumeField(color, type_hint=VolumeFieldTypeHint.COLOR)
    
    @staticmethod
    def _fix_resolution(resolution: int | Idx3) -> Idx3:
        if isinstance(resolution, int):
            res = [resolution]*3
        else:
            res = resolution
        return res
    
    @staticmethod    
    def signedDistanceFromRhinoMesh(mesh, resolution: Idx3|int, device: torch.device, bounds_min : np.ndarray, bounds_max : np.ndarray) -> 'VolumeField':
        import rhino3dm as r3d
        if not isinstance(mesh, r3d.Mesh):
            raise ValueError("Input mesh must be a rhino3dm Mesh")
        
        resolution = VolumeField._fix_resolution(resolution)

        mesh.Faces.ConvertQuadsToTriangles()

        bounds_min = np.array(bounds_min, dtype=np.float32)
        bounds_max = np.array(bounds_max, dtype=np.float32)
        resolution = np.array(resolution, dtype=int)

        # Create a uniform grid of the same spatial extent
        grid = vtk.vtkImageData()
        grid.SetDimensions(resolution)
        grid.SetSpacing((bounds_max - bounds_min) / (resolution - 1.0))
        grid.SetOrigin(bounds_min)

        grid_shape = resolution
        field_shape = np.flip(grid_shape)


        surf : vtk.vtkPolyData = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        for v in mesh.Vertices:
            points.InsertNextPoint(v.X, v.Y, v.Z)
        surf.SetPoints(points)
        polys = vtk.vtkCellArray()
        for f in mesh.Faces:
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, f[0])
            tri.GetPointIds().SetId(1, f[1])
            tri.GetPointIds().SetId(2, f[2])
            polys.InsertNextCell(tri)

        surf.SetPolys(polys)


        implicitPolyDataDistance = vtk.vtkImplicitPolyDataDistance()
        implicitPolyDataDistance.SetInput(surf)

        sampler = vtk.vtkSampleImplicitFunctionFilter()
        sampler.SetInputData(grid)
        sampler.ComputeGradientsOff()
        sampler.SetScalarArrayName('signed_distance')
        sampler.SetImplicitFunction(implicitPolyDataDistance)

        sampler.Update()

        singed_distance : vtk.vtkImageData = sampler.GetOutput()
        signed_distance = vtk_np.vtk_to_numpy(singed_distance.GetPointData().GetArray('signed_distance'))

        signed_distance = signed_distance.reshape(field_shape)

        with torch.no_grad():
            distance_tensor = torch.tensor(signed_distance, dtype=torch.float32, device=device)
            distance_tensor = distance_tensor.unsqueeze(0).unsqueeze(0)

        return VolumeField(distance_tensor, type_hint=VolumeFieldTypeHint.SDF)
    
    @staticmethod
    def _get_cache_name(
        path:str,
        resolution:int|Idx3,
        unit_box:bool
        ) -> str:
        fname = os.path.splitext(path)[0]
        if isinstance(resolution, int):
            res_str = f"{resolution}"
        elif (resolution[0] == resolution[1]) and (resolution[1] == resolution[2]):
            res_str = f"{resolution[0]}"
        else:
            res_str = f"[{resolution[0]}x{resolution[1]}x{resolution[2]}]"

        if unit_box:
            cache_name = f"{fname}_sdf_{res_str}_unitbox.npz"
        else:
            cache_name = f"{fname}_sdf_{res_str}.npz"

        return cache_name
    
    @staticmethod
    def _try_get_cached(
        path:str,
        resolution:Idx3|int,
        unit_box:bool,
        device: torch.device
        ) -> tuple[str, 'VolumeField']:
        cache_name = VolumeField._get_cache_name(path, resolution, unit_box)
        if os.path.exists(cache_name):
            with torch.no_grad():
                print(f"Loading cached SDF from: {cache_name}")
                field = VolumeField.fromNpzFile(cache_name, device)
                return cache_name, field
        return cache_name, None 
            
    @staticmethod
    def _split_tensor_to_fields(
        sdf_tensor: torch.Tensor,
        type_hint: VolumeFieldTypeHint
        ) -> list['VolumeField']:
        with torch.no_grad():
            fields = []
            for i in range(sdf_tensor.shape[0]):
                fields.append(VolumeField(sdf_tensor[i].unsqueeze(0), type_hint=type_hint))
            return fields
        

    @staticmethod
    def fromFile(
            path:str,
            device: torch.device,
            resolution: int | Idx3, #this is only used for mesh files
            unit_box:bool = True,
            use_cache: bool = True,
            union: bool = False
        ) -> 'VolumeField':

        ext = path.split('.')[-1].lower()
        
        if ext == 'npy':
            return VolumeField.fromNpyFile(path, device, type_hint=VolumeFieldTypeHint.SDF, union=union)
        elif ext == 'npz':
            return VolumeField.fromNpzFile(path, device, union=union)        
        elif ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
            return VolumeField.fromImage(path, device)
        elif ext == 'obj':
            return VolumeField.signedDistanceFromOBJFile(path, resolution, device, unit_box, use_cache)
        elif ext == '3dm':
            field = VolumeField.signedDistanceFromRhinoFile(
                path, 
                device, 
                resolution, 
                unit_box, 
                use_cache, 
                union
                )
            return field
        else:
            raise ValueError(f"Unsupported file extension: {ext}")


    @staticmethod    
    def signedDistanceFromRhinoFile(
        path:str, 
        device: torch.device, 
        resolution:int | Idx3 = 32, 
        unit_box:bool = True, 
        use_cache: bool = True, 
        union: bool = False
        ) -> 'VolumeField':

        
        if use_cache:
            cache_name, sdf_field = VolumeField._try_get_cached(path, resolution, unit_box, device)

            if sdf_field is not None:
                with torch.no_grad():
                    if union:                    
                        sdf_field = sdf_field.union_batch()
                    return sdf_field


        import rhino3dm as r3d
        file3dm = r3d.File3dm.Read(path)        

        meshes = []
        box_union = None

        for obj in file3dm.Objects:
            if isinstance(obj.Geometry, r3d.Mesh):
                mesh : r3d.Mesh = obj.Geometry
                box = mesh.GetBoundingBox()
                if box_union is None:
                    box_union = box
                else:
                    box_union = r3d.BoundingBox.Union(box_union, box)
                meshes.append(mesh)
        
        max_dim = max(box_union.Diagonal.X, box_union.Diagonal.Y, box_union.Diagonal.Z)
        center = box_union.Center
        half = (max_dim / 2.0) + 0.01
        bounds_min = np.array([center.X - half, center.Y - half, center.Z - half], dtype=np.float32)
        bounds_max = np.array([center.X + half, center.Y + half, center.Z + half], dtype=np.float32)
        res = VolumeField._fix_resolution(resolution)
        
        scale_sdf = 1.0
        if unit_box:
            scale_sdf = 1.0 / half
        
        with torch.no_grad():
            sdf_tensors = []
            for i, mesh in enumerate(meshes):
                print(f"Processing mesh {i+1}/{len(meshes)}")
                sdf_field = VolumeField.signedDistanceFromRhinoMesh(
                    mesh, 
                    res, 
                    device, 
                    bounds_min, 
                    bounds_max
                    )
                
                sdf_tensors.append(sdf_field.tensor)   
        
            if len(sdf_tensors) == 0:
                raise ValueError("No meshes found in the Rhino file.")
            
            sdf_stack = torch.cat(sdf_tensors, dim=0) * scale_sdf
            field = VolumeField(sdf_stack, type_hint=VolumeFieldTypeHint.SDF)

            if use_cache:        
                field.save(cache_name)     

            if union:
                field = field.union_batch()

            return field

    @staticmethod
    def signedDistanceFromOBJFile(
        filename : str, 
        resolution : int | Idx3, 
        device : torch.device = Vol.device,
        unit_box:bool = True, 
        use_cache: bool = True, 
        ) -> 'VolumeField':

        if use_cache:
            cache_name, sdf_field = VolumeField._try_get_cached(filename, resolution, unit_box, device)

            if sdf_field is not None:
                return sdf_field
            
        resolution = VolumeField._fix_resolution(resolution)

        bounds_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        bounds_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        resolution = np.array(resolution, dtype=int)

        # Create a uniform grid of the same spatial extent
        grid = vtk.vtkImageData()
        grid.SetDimensions(resolution)
        grid.SetSpacing((bounds_max - bounds_min) / (resolution - 1.0))
        grid.SetOrigin(bounds_min)

        grid_shape = resolution
        field_shape = np.flip(grid_shape)

        reader  = vtk.vtkOBJReader()
        reader.SetFileName(filename)
        reader.Update()

        surf : vtk.vtkPolyData = reader.GetOutput()
        sbounds = surf.bounds

        implicitPolyDataDistance = vtk.vtkImplicitPolyDataDistance()
        implicitPolyDataDistance.SetInput(surf)

        sampler = vtk.vtkSampleImplicitFunctionFilter()
        sampler.SetInputData(grid)
        sampler.ComputeGradientsOff()
        sampler.SetScalarArrayName('signed_distance')
        sampler.SetImplicitFunction(implicitPolyDataDistance)

        sampler.Update()

        singed_distance : vtk.vtkImageData = sampler.GetOutput()
        signed_distance = vtk_np.vtk_to_numpy(singed_distance.GetPointData().GetArray('signed_distance'))

        signed_distance = signed_distance.reshape(field_shape)

        with torch.no_grad():
            distance_tensor = torch.tensor(signed_distance, dtype=torch.float32, device=device)
            distance_tensor = distance_tensor.unsqueeze(0).unsqueeze(0)

        field = VolumeField(distance_tensor, type_hint=VolumeFieldTypeHint.SDF)

        if use_cache:
            field.save(cache_name)

        return field
        
    def linear_density(self, peak_dist:float, fade_dist:float) -> 'VolumeField':
        self.type_hint = VolumeFieldTypeHint.DENSITY
        return self.remap((fade_dist, peak_dist))
    
    def gaussian_density(self, peak_dist : float, fade_dist : float) -> 'VolumeField':
        self.type_hint = VolumeFieldTypeHint.DENSITY
        self.remap((peak_dist, fade_dist), clamp=False).clamp_min(0.0)
        x = self.tensor**2
        mul = -1.0/(0.25*fade_dist*fade_dist)
        self.tensor = torch.exp(x*mul)
        return self

    def remap(self, inp_range:tuple[float, float], out_range:tuple[float, float] = None, clamp:bool = True) -> 'VolumeField':
        self.tensor = (self.tensor - inp_range[0]) / (inp_range[1] - inp_range[0])
        if out_range is not None:
            self.tensor = self.tensor * (out_range[1] - out_range[0]) + out_range[0]

        if clamp:
            if out_range is None:
                self.clamp(0.0, 1.0)
            else:
                self.clamp(out_range[0], out_range[1])

        return self
    
    def clamp(self, min_val:float, max_val:float) -> 'VolumeField':
        self.tensor.clamp_(min_val, max_val)
        return self
    
    def clamp_min(self, min_val:float) -> 'VolumeField':
        self.tensor.clamp_min_(min_val)
        return self
    
    def clamp_max(self, max_val:float) -> 'VolumeField':
        self.tensor.clamp_max_(max_val)
        return self
        
 
        
    @staticmethod
    def fromNumpyArray(np_array : np.ndarray, device : torch.device = Vol.device, type_hint: VolumeFieldTypeHint = VolumeFieldTypeHint.DENSITY):
        return VolumeField(torch.tensor(np_array, dtype=torch.float32, device=device), type_hint=type_hint)
    
    @staticmethod
    def fromNpyFile(filename : str, device : torch.device = Vol.device, type_hint: VolumeFieldTypeHint = VolumeFieldTypeHint.SDF, union: bool = False) -> 'VolumeField':
        arr = np.load(filename)
        field = VolumeField.fromNumpyArray(arr, device, type_hint)
        if union:
            field = field.union_batch()
        return field
    
    @staticmethod
    def fromNpzFile(filename : str, device : torch.device = Vol.device, union: bool = False) -> tuple['VolumeField', dict]:
        npz = np.load(filename, allow_pickle=True)
        arr = npz['data']
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        tensor = torch.tensor(arr, dtype=torch.float32, device=device)

        metadata = {}
        if 'metadata' in npz:
            meta_json = npz['metadata'].item()
            metadata = json.loads(meta_json)

        type_hint_name = metadata.get('type_hint', None)
        type_hint = VolumeFieldTypeHint.DENSITY
        if type_hint_name is not None:
            type_hint = VolumeFieldTypeHint(type_hint_name)

        field = VolumeField(tensor, type_hint=type_hint, metadata=metadata)
        if union:
            field = field.union_batch()
        return field
    
    @staticmethod
    def fromImage(filename : str, device : torch.device = Vol.device):
        
        img = Image.open(filename).convert('L')
        np_img = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.tensor(np_img, dtype=torch.float32, device=device)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return VolumeField(tensor, type_hint=VolumeFieldTypeHint.DENSITY)

    @staticmethod
    def makeSolidDensity(d: float, resolution : Idx3, device : torch.device = Vol.device, type_hint: VolumeFieldTypeHint = VolumeFieldTypeHint.DENSITY) -> 'VolumeField':
        return VolumeField(torch.ones(1, 1, resolution[0], resolution[1], resolution[2], dtype=torch.float32, device=device)*d, type_hint=type_hint)



    def __init__(self, tensor: torch.Tensor, type_hint: VolumeFieldTypeHint = VolumeFieldTypeHint.DENSITY, metadata: dict = {}):
        if len(tensor.shape) == 3:
            self.tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif len(tensor.shape) == 4:
            self.tensor = tensor.unsqueeze(0)
        elif len(tensor.shape) == 5:
            self.tensor = tensor
        else:
            raise ValueError('Invalid tensor shape for volume field. Must be 3, 4 or 5 dimensional.')
        
        self.type_hint = type_hint
        self.metadata = metadata


    @property
    def device(self):
        return self.tensor.device
    
    @property
    def shape(self):
        return self.tensor.shape
    
    @property
    def nx(self):
        return self.tensor.shape[4]
    
    @property
    def ny(self):
        return self.tensor.shape[3]
    
    @property
    def nz(self):
        return self.tensor.shape[2]
    
    @property
    def channels(self):
        return self.tensor.shape[1]
    
    @property
    def batchCount(self):
        return self.tensor.shape[0]
    
    def union_batch(self) -> 'VolumeField':
        if self.batchCount == 1:
            return self
        united_tensor, _ = torch.min(self.tensor, dim=0, keepdim=True)
        return VolumeField(united_tensor, type_hint=self.type_hint)
        
    def fill(self, value: float):
        self.tensor.fill_(value)

    def set_boundary_voxels(self, value: float):
        self.tensor[:, :, 0, :, :].fill_(value)
        self.tensor[:, :, -1, :, :].fill_(value)
        self.tensor[:, :, :, 0, :].fill_(value)
        self.tensor[:, :, :, -1, :].fill_(value)
        self.tensor[:, :, :, :, 0].fill_(value)
        self.tensor[:, :, :, :, -1].fill_(value)

    def splitBatch(self) -> list['VolumeField']:
        if self.batchCount == 1:
            return [self]
        fields = []
        for i in range(self.batchCount):
            fields.append(VolumeField(self.tensor[i].unsqueeze(0), type_hint=self.type_hint))
        return fields

    def getGradient(self, normalized : bool = True) -> 'VolumeField':
        return VolumeField(Vol.computeGradient(self.tensor, normalized), type_hint=VolumeFieldTypeHint.VEC3)
    
    def getLaplacian(self) -> 'VolumeField':
        return VolumeField(Vol.computeLaplacian(self.tensor), type_hint=VolumeFieldTypeHint.DENSITY)
    
    def smooth(self, kernel_size=5, strength:float = 1.0) -> 'VolumeField':
        channel_count = self.tensor.shape[1]
        kernel = torch.ones(1, channel_count, kernel_size, kernel_size, kernel_size, dtype=torch.float32, device=self.tensor.device)
        kernel = kernel / kernel.sum()
        blured = F.conv3d(self.tensor, kernel, padding=kernel_size//2)

        if strength != 1.0:
            self.tensor.data = self.tensor + (blured - self.tensor)*strength
        else:
            self.tensor.data = blured

        return self
    
    def saveAsImage(self, filename : str):
        slice_stack = self.tensor.reshape(self.channels, self.batchCount*self.nz*self.ny, self.nx)
        save_tensor_image(slice_stack, filename)

    def save(self, filename : str, type_hint_override: VolumeFieldTypeHint = None, use_16_bit: bool = True, extra_data : dict = {}):

        meta = self.metadata.copy()
        meta['extra_data'] = extra_data

        saved_type_hint = type_hint_override if type_hint_override is not None else self.type_hint

        meta['type_hint'] = saved_type_hint.value
 
        meta_json = json.dumps(meta)

        arr = self.tensor.detach().cpu().numpy()
        if use_16_bit:
            arr = arr.astype(np.float16)

        np.savez_compressed(filename, data=arr, metadata=meta_json)
        

    def resample(self, grid: torch.Tensor, padding_mode : RESAMPLEPADDING = RESAMPLEPADDING.ZEROS) -> 'VolumeField':
        return VolumeField(F.grid_sample(self.tensor, grid, padding_mode=padding_mode.value), type_hint=self.type_hint) 
    
    def resized(self, size: Idx3, mode='trilinear') -> 'VolumeField':
        return VolumeField(F.interpolate(self.tensor, size=size, mode=mode), type_hint=self.type_hint)
    
    def clone(self) -> 'VolumeField':
        return VolumeField(self.tensor.clone(), type_hint=self.type_hint, metadata=self.metadata.copy())
    
    def get_isosurfaces(self, contours : list[float] = [0.5]) -> list[VolSimpleMesh]:
        
        #create vtk field from density tensor then save mesh
        vtk_density = self.tensor.squeeze(0).squeeze(0).permute(2, 1, 0).cpu().numpy()
        vtk_density = vtk_np.numpy_to_vtk(vtk_density.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_density.SetName('Scalar')

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(self.nz, self.ny, self.nx)
        vtk_image.GetPointData().AddArray(vtk_density)
        vtk_image.GetPointData().SetActiveScalars('Scalar')

        #use vtkMarchingCubes to create a mesh from the density field
        mc = vtk.vtkImageMarchingCubes()
        mc.SetInputData(vtk_image)

        meshes = []
        for contour in contours:
            mc.SetValue(0, contour)
            mc.Update()

            surf : vtk.vtkPolyData = mc.GetOutput()
            points = surf.GetPoints()
            polys = surf.GetPolys()


            num_verts = points.GetNumberOfPoints()
            verts = np.ndarray((num_verts, 3), dtype=np.float32)

            sx = 2.0/self.nx
            sy = 2.0/self.ny
            sz = 2.0/self.nz

            for i in range(num_verts):
                v = points.GetPoint(i)
                verts[i] = [v[2]*sz - 1.0, v[1]*sy - 1.0, v[0]*sx - 1.0]
            
            
            faces = []
            polys.InitTraversal()
            idList = vtk.vtkIdList()
            while polys.GetNextCell(idList):
                if idList.GetNumberOfIds() == 3:
                    faces.append([idList.GetId(2), idList.GetId(1), idList.GetId(0)])
                elif idList.GetNumberOfIds() == 4:
                    #convert quad to two triangles
                    faces.append([idList.GetId(2), idList.GetId(1), idList.GetId(0)])
                    faces.append([idList.GetId(3), idList.GetId(2), idList.GetId(0)])
                else:
                    #ignore non-triangle faces
                    pass

            meshes.append(VolSimpleMesh(verts, faces))

        return meshes
    

class VolumeComponent:
    def __init__(self):
        pass


    


