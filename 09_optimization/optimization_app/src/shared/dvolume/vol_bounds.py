import torch
from .vol_types import *
from .vol import *

class Bounds3d:
    @staticmethod
    def makeUnitCube():
        return Bounds3d([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    
    @staticmethod
    def makeUnitSymmetricCube():
        return Bounds3d([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    
    @staticmethod
    def makeFromPointsTensor(points : torch.Tensor) -> 'Bounds3d':
        min = torch.min(points, dim=0).values
        max = torch.max(points, dim=0).values
        return Bounds3d(min, max)
    
    def __init__(self, min : Vec3, max : Vec3):
        self.min = ensureNumpy(min)
        self.max = ensureNumpy(max)

    @property
    def min_x(self):
        return self.min[0].item()
    
    @property
    def min_y(self):
        return self.min[1].item()
    
    @property
    def min_z(self):
        return self.min[2].item()
    
    @property
    def max_x(self):
        return self.max[0].item()
    
    @property
    def max_y(self):
        return self.max[1].item()
    
    @property
    def max_z(self):
        return self.max[2].item()
    
    @property
    def dx(self):
        return self.max_x-self.min_x
    
    @property
    def dy(self): 
        return self.max_y-self.min_y
    
    @property
    def dz(self):
        return self.max_z-self.min_z

    @property
    def size(self):
        return self.max - self.min

    @property
    def center(self):
        return (self.min + self.max)*0.5

    def contains(self, point : Vec3) -> bool:
        return  point[0] >= self.min[0] and point[0] <= self.max[0] and point[1] >= self.min[1] and point[1] <= self.max[1] and point[2] >= self.min[2] and point[2] <= self.max[2]

    def containsBounds(self, bounds : 'Bounds3d') -> bool:
        return self.contains(bounds.min) and self.contains(bounds.max)

    def inflate(self, amount : float):
        self.min -= amount
        self.max += amount

    def getCornersTensor(self, device : torch.device) -> torch.Tensor:
        return torch.tensor([
            [self.min[0], self.min[1], self.min[2]],
            [self.max[0], self.min[1], self.min[2]],
            [self.max[0], self.max[1], self.min[2]],
            [self.min[0], self.max[1], self.min[2]],
            [self.min[0], self.min[1], self.max[2]],
            [self.max[0], self.min[1], self.max[2]],
            [self.max[0], self.max[1], self.max[2]],
            [self.min[0], self.max[1], self.max[2]]
        ], dtype=torch.float32, device = device)
        