import torch

from .vol_types import *
from .vol import *

class VolumeCoords:
    def __init__(self, 
                 resolution : Idx3, 
                 min : Vec3 = [-1.0, -1.0, -1.0], 
                 max : Vec3 = [1.0, 1.0, 1.0],
                 device : torch.device = Vol.device
                 ):
        self.device = device
        self.resolution = resolution
        #x has shape {resolution[2]} ~ {width}
        self.x = torch.linspace(min[0], max[0], resolution[2], dtype=torch.float32, device=device)
        #y has shape {resolution[1]} ~ {height}
        self.y = torch.linspace(min[1], max[1], resolution[1], dtype=torch.float32, device=device)
        #z has shape {resolution[0]} ~ {depth}
        self.z = torch.linspace(min[2], max[2], resolution[0], dtype=torch.float32, device=device)
        #now they have shape {resolution[0], resolution[1], resolution[2]} ~ {depth, height, width}
        self.z, self.y, self.x = torch.meshgrid(self.z, self.y, self.x, indexing='ij')

        #add batch and channel dimensions to the grid coords
        #now shape is {batch=1, channel=1, depth, height, width}
        self.x = self.x.unsqueeze(0).unsqueeze(0)
        self.y = self.y.unsqueeze(0).unsqueeze(0)
        self.z = self.z.unsqueeze(0).unsqueeze(0)  

        #creating transformable tensor for grid
        #xs, ys, zs, ws have shapes {channel=1, depth, height, width}
        xs = self.x.squeeze(0)
        ys = self.y.squeeze(0)
        zs = self.z.squeeze(0)
        ws = torch.ones_like(xs)

        #concatenate xs, ys, zs, ws along channel dimension
        #grid has shape {4, depth, height, width}
        self.xyzw : torch.Tensor = torch.cat([xs, ys, zs, ws], dim=0)
        #set order to {depth, height, width, 4} where the last dim contains {x,y,z,w} points
        self.xyzw = self.xyzw.permute(1, 2, 3, 0)

        #xyz is {3, depth, height, width} used in sdf calculations
        self.xyz = torch.cat([self.x, self.y, self.z], dim=1).squeeze(0)
  
    
    @property
    def shape(self) -> list[int]:
        return self.x.shape
    
    @property
    def nx(self) -> int:
        return self.x.shape[4]
    
    @property
    def ny(self) -> int:
        return self.x.shape[3]
    
    @property
    def nz(self) -> int:
        return self.x.shape[2]