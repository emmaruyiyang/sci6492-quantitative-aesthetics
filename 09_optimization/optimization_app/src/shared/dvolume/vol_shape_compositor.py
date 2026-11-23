import torch

from typing import Protocol
from abc import abstractmethod

from .vol import *
from .vol_types import *
from .vol_field import *
from .vol_coords import *
from .vol_camera import *
from .vol_bounds import *
from .vol_transform3d import *
from .vol_lighting import *
from .vol_viewport import *
from .vol_model import *

ParamDef = Union[float, list[float], tuple[float, float, float], torch.Tensor]
Param = Union[float, torch.Tensor]


class ShapeParam:
    @staticmethod
    def make(p : ParamDef, device : torch.device) -> Param:
        if isinstance(p, tuple):
            t = torch.tensor(p[0], device=device)
        elif isinstance(p, list):
            t = torch.tensor(p[0], device=device)
        elif isinstance(p, torch.Tensor):
            return p
        else:
            return torch.tensor(p, device=device)
        
        t.param_range = (p[1], p[2])
        return t
        
    @staticmethod
    def isParam(p : Param) -> bool:
        return isinstance(p, torch.Tensor) and hasattr(p, 'param_range')
    
    @staticmethod
    def clamp(p : Param):
        if ShapeParam.isParam(p):
            p.clamp_(p.param_range[0], p.param_range[1])

class ShapeTransform:
    def __init__(self, 
                 dx: ParamDef, dy : ParamDef, dz : ParamDef,
                 rx: ParamDef, ry : ParamDef, rz : ParamDef,
                 sx: ParamDef, sy : ParamDef, sz : ParamDef,
                 device:torch.device):
        self.device = device
        self.dx = ShapeParam.make(dx, device)
        self.dy = ShapeParam.make(dy, device)
        self.dz = ShapeParam.make(dz, device)

        self.rx = ShapeParam.make(rx, device)
        self.ry = ShapeParam.make(ry, device)
        self.rz = ShapeParam.make(rz, device)

        self.sx = ShapeParam.make(sx, device)
        self.sy = ShapeParam.make(sy, device)
        self.sz = ShapeParam.make(sz, device)

        self.post_transform_matrix : torch.Tensor = None

        self.params = [self.dx, self.dy, self.dz, self.rx, self.ry, self.rz, self.sx, self.sy, self.sz]

    def createSamplingGrid(self, coords_grid : VolumeCoords) -> torch.Tensor:
        dx = Transform3d.makeTranslation3(self.dx, self.dy, self.dz, self.device)
    
        sx = Transform3d.makeScale3(self.sx, self.sy, self.sz, self.device)
        
        rx = Transform3d.makeRotationX(self.rx)
        ry = Transform3d.makeRotationY(self.ry)
        rz = Transform3d.makeRotationZ(self.rz)

   
        if self.post_transform_matrix is not None:
            xform = dx @ self.post_transform_matrix @  rx @ ry @ rz @ sx
        else:
            xform = dx @ rx @ ry @ rz @ sx

        unproject_matrix = xform.getInverse()

        #transform grid to volume space using inverse perspective matrix
        grid_volume = torch.matmul(coords_grid.xyzw, unproject_matrix.tensor.t())
        grid_volume = grid_volume / grid_volume[:, :, :, 3].unsqueeze(3)
        grid_volume = grid_volume[:, :, :, 0:3]
        sampling_grid = grid_volume.unsqueeze(0)

        return sampling_grid

    def clampParams(self):
        for p in self.params:
            ShapeParam.clamp(p)

    def collectParams(self) -> list[torch.Tensor]:
        return [p for p in self.params if ShapeParam.isParam(p)]
            


class ShapeProto(Protocol):
    @abstractmethod
    def transformToGrid(self, coords_grid : VolumeCoords) -> VolumeField: raise NotImplementedError
    def clampParams(self): raise NotImplementedError
    def collectParams(self) -> list[torch.Tensor]: raise NotImplementedError


class Shape(ShapeProto):
    def __init__(self, field:VolumeField, transform:ShapeTransform, strength:ParamDef):
        self.device = field.device
        self.field = field
        self.transform = transform
        self.strength = ShapeParam.make(strength, self.device)
        
    def transformToGrid(self, coords_grid : VolumeCoords) -> VolumeField:
        sampling_grid = self.transform.createSamplingGrid(coords_grid)
        transformed_field = self.field.resample(sampling_grid)
        return transformed_field

    def clampParams(self):
        ShapeParam.clamp(self.strength)
        self.transform.clampParams()

    def collectParams(self) -> list[torch.Tensor]:
        params = self.transform.collectParams()
        if ShapeParam.isParam(self.strength):
            params.append(self.strength)
        return params

class ShapeCompositor:
    def __init__(self, device:torch.device):
        self.device = device
        self.shapes : list[Shape] = []
        
    def append(self, shape:Shape):
        self.shapes.append(shape)

    def createCompositeForGrid(self, coords_grid : VolumeCoords) -> torch.Tensor:
        transformed_shape_tensors = []
        for s in self.shapes:
            sg = s.transformToGrid(coords_grid).tensor * s.strength
            transformed_shape_tensors.append(sg)

        stack = torch.stack(transformed_shape_tensors, dim=0)
        return torch.sum(stack, dim=0)

    def clampParams(self):
        for s in self.shapes:
            s.clampParams()

    def collectParams(self) -> list[torch.Tensor]:
        params = []
        for s in self.shapes:
            params.extend(s.collectParams())
        return params
    