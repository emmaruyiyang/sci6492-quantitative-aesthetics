import torch
import numpy as np

from .vol import *
from .vol_types import *

class Transform3d:
    @staticmethod
    def makeIdentity(device : torch.device = Vol.device):
        return Transform3d(torch.eye(4, dtype=torch.float32, device=device))
    
    @staticmethod
    def makeTranslation(translation : Vec3, device : torch.device = Vol.device):
        if isinstance(translation, torch.Tensor):
            if len(translation.shape) != 1 or translation.shape[0] != 3:
                raise ValueError('Invalid shape for 3d translation. Must be 3.')
            tensor = torch.eye(4, dtype=torch.float32, device=translation.device)
            tensor[0:3, 3] = translation
            return Transform3d(tensor)
        else:
            return Transform3d(torch.tensor([
                [1.0, 0.0, 0.0, translation[0]],
                [0.0, 1.0, 0.0, translation[1]],
                [0.0, 0.0, 1.0, translation[2]],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=torch.float32, device=device))
        
    @staticmethod
    def makeTranslation3(
        dx : Union[float , torch.Tensor], 
        dy : Union[float , torch.Tensor], 
        dz : Union[float , torch.Tensor], 
        device : torch.device):

        tensor = torch.eye(4, dtype=torch.float32, device=device)
        tensor[0, 3] = dx
        tensor[1, 3] = dy
        tensor[2, 3] = dz
        return Transform3d(tensor)
    
    @staticmethod
    def makeScale3(
        sx : Union[float, torch.Tensor], 
        sy : Union[float, torch.Tensor], 
        sz : Union[float, torch.Tensor], 
        device : torch.device):

        tensor = torch.eye(4, dtype=torch.float32, device=device)
        tensor[0, 0] = sx
        tensor[1, 1] = sy
        tensor[2, 2] = sz
        return Transform3d(tensor)
        
    @staticmethod
    def isScalar(tensor : torch.Tensor):
        return len(tensor.shape) == 0 or (len(tensor.shape) == 1 and tensor.shape[0] == 1)
    
    @staticmethod
    def makeRotationX(angle_rad : float, device : torch.device = Vol.device):
        if isinstance(angle_rad, torch.Tensor):
            if not Transform3d.isScalar(angle_rad):
                raise ValueError('Invalid shape for 3d rotation angle. Must be scalar.')
            tensor = torch.eye(4, dtype=torch.float32, device=angle_rad.device)
            c = torch.cos(angle_rad)
            s = torch.sin(angle_rad)
            tensor[1,1] = c
            tensor[1,2] = -s
            tensor[2,1] = s
            tensor[2,2] = c
            return Transform3d(tensor) 
        else: 
            c = np.cos(angle_rad)
            s = np.sin(angle_rad)
            return Transform3d(torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, -s, 0.0],
                [0.0, s, c, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=torch.float32, device=device))
    
    @staticmethod
    def makeRotationY(angle_rad : float, device : torch.device = Vol.device):
        if isinstance(angle_rad, torch.Tensor):
            if not Transform3d.isScalar(angle_rad):
                raise ValueError('Invalid shape for 3d rotation angle. Must be scalar.')
            tensor = torch.eye(4, dtype=torch.float32, device=angle_rad.device)
            c = torch.cos(angle_rad)
            s = torch.sin(angle_rad)
            tensor[0,0] = c
            tensor[0,2] = s
            tensor[2,0] = -s
            tensor[2,2] = c
            return Transform3d(tensor)
        else:
            c = np.cos(angle_rad)
            s = np.sin(angle_rad)
            return Transform3d(torch.tensor([
                [c, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=torch.float32, device=device))
    
    @staticmethod
    def makeRotationZ(angle_rad : float, device : torch.device = Vol.device):
        if isinstance(angle_rad, torch.Tensor):
            if not Transform3d.isScalar(angle_rad):
                raise ValueError('Invalid shape for 3d rotation angle. Must be scalar.')
            tensor = torch.eye(4, dtype=torch.float32, device=angle_rad.device)
            c = torch.cos(angle_rad)
            s = torch.sin(angle_rad)
            tensor[0,0] = c
            tensor[0,1] = -s
            tensor[1,0] = s
            tensor[1,1] = c
            return Transform3d(tensor)
        else:
            c = np.cos(angle_rad)
            s = np.sin(angle_rad)
            return Transform3d(torch.tensor([
                [c, -s, 0.0, 0.0],
                [s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=torch.float32, device=device))
    
    @staticmethod
    def makeScale(scale : Vec3, device : torch.device = Vol.device):
        if isinstance(scale, torch.Tensor):
            if len(scale.shape) != 1 or scale.shape[0] != 3:
                raise ValueError('Invalid shape for 3d scale. Must be 3.')
            tensor = torch.eye(4, dtype=torch.float32, device=scale.device)
            tensor[0:3, 0:3] = torch.diag(scale)
            return Transform3d(tensor)
        else:
            return Transform3d(torch.tensor([
                [scale[0], 0.0, 0.0, 0.0],
                [0.0, scale[1], 0.0, 0.0],
                [0.0, 0.0, scale[2], 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=torch.float32, device=device))
    
    @staticmethod
    def makePerspective(
            fov_deg : float, 
            aspect : float, 
            znear : float, zfar : float, 
            flip_y : bool = True, 
            device : torch.device = Vol.device
        ):
        fov_rad = fov_deg * np.pi / 180.0
        top = znear * np.tan(fov_rad / 2.0)
        bottom = -top
        right = top * aspect
        left = -right
        y_scale = -2.0 if flip_y else 2.0
        return Transform3d(torch.tensor([
            [2.0 * znear / (right - left), 0.0, (right + left) / (right - left), 0.0],
            [0.0, y_scale * znear / (top - bottom), (top + bottom) / (top - bottom), 0.0],
            [0.0, 0.0, -(zfar + znear) / (zfar - znear), -2.0 * zfar * znear / (zfar - znear)],
            [0.0, 0.0, -1.0, 0.0]
        ], dtype=torch.float32, device=device))

    @staticmethod
    def makeOrthographic(
        left : float,
        right : float,
        bottom : float,
        top : float,
        znear : float,
        zfar : float,
        device : torch.device = Vol.device
    ):
        return Transform3d(torch.tensor([
            [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
            [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
            [0.0, 0.0, -2.0 / (zfar - znear), -(zfar + znear) / (zfar - znear)],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device))    

    @staticmethod
    def makeLookAt(
        view_point : Vec3, 
        target_point : Vec3, 
        up_vector : Vec3, 
        device : torch.device = Vol.device
    ):
        view_point = ensureTensor(view_point, device)
        target_point = ensureTensor(target_point, device)
        up_vector = ensureTensor(up_vector, device)

        zaxis = view_point - target_point
        xaxis = torch.cross(up_vector, zaxis, -1)
        yaxis = torch.cross(zaxis, xaxis, -1)
        xaxis = xaxis / torch.norm(xaxis)
        yaxis = yaxis / torch.norm(yaxis)
        zaxis = zaxis / torch.norm(zaxis)
        return Transform3d(torch.tensor([
            [xaxis[0], xaxis[1], xaxis[2], -torch.dot(xaxis, view_point)],
            [yaxis[0], yaxis[1], yaxis[2], -torch.dot(yaxis, view_point)],
            [zaxis[0], zaxis[1], zaxis[2], -torch.dot(zaxis, view_point)],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device))

    
    def __init__(self, tensor : torch.Tensor):
        self.tensor = tensor

    def __matmul__(self, other : 'Transform3d') -> 'Transform3d':
        return Transform3d(torch.matmul(self.tensor, other.tensor))
    
    def __invert__(self) -> 'Transform3d':
        return Transform3d(torch.inverse(self.tensor))
    
    @property
    def device(self):
        return self.tensor.device
    
    @property
    def tensor3x3(self):
        return self.tensor[0:3, 0:3]
    
    def getInverse(self) -> 'Transform3d':
        return Transform3d(torch.inverse(self.tensor))
    
    def transformPoints(self, points : torch.Tensor, divideHomogeneous : bool = True) -> torch.Tensor:
        if len(points.shape) != 2:
            raise ValueError('Invalid shape for 3d points. Must be Nx3 or Nx4.')
        
        n = points.shape[0]
        c = points.shape[1]

        if c == 3:
            points = torch.cat([points, torch.ones(n, 1, dtype=torch.float32, device = self.device)], dim=1)
        elif c != 4:
            raise ValueError('Invalid shape for 3d points. Must be Nx3 or Nx4.')
        
        points = torch.matmul(points, self.tensor.t())
        if divideHomogeneous:
            points = points / points[:, 3].unsqueeze(1)
            return points[:, 0:3]
        else:
            return points
        
    def transformVector(self, vector : Vec3) -> torch.Tensor:
        vector = ensureTensor(vector, self.device)
        if len(vector.shape) != 1 or vector.shape[0] != 3:
            raise ValueError('Invalid shape for 3d vector. Must be 3.')
        return torch.matmul(self.tensor3x3, vector)
    
    def transformVectors(self, vectors : list[Vec3]) -> torch.Tensor:
        vectors = ensureTensor(vectors, self.device)
        if len(vectors.shape) != 2 or vectors.shape[1] != 3:
            raise ValueError('Invalid shape for 3d vectors. Must be Nx3.')
        return torch.matmul(vectors, self.tensor3x3.t())
    
