#check https://iquilezles.org/articles/distfunctions/
import torch

from .vol import *
from .vol_types import *
from .vol_field import *
from .vol_coords import *
from .vol_transform3d import *
from enum import Enum

import math

class PLANE(Enum):
    XY = 0
    YZ = 1
    ZX = 2

def __dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum(a*b, dim=0)

def __cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cross(a, b, dim=0)

def __norm(a: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(a, dim=0)

def __normalize(a: torch.Tensor) -> torch.Tensor:
    return a / __norm(a)

def __pointTensor(point: Vec3, device: torch.device) -> torch.Tensor:
    return ensureTensor(point, device=device).reshape(3, 1, 1, 1)

def __max(a: torch.Tensor, b: Union[torch.Tensor ,float]) -> torch.Tensor:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.max(a, b)
    elif isinstance(a, torch.Tensor): #clamp(tensor, min) is equivalent to max(tensor, min)
        return torch.clamp(a,  min = b)
    elif isinstance(b, torch.Tensor):
        return torch.clamp(b,  min = a)
    else:
        return torch.max(a, b)

def makeSphere(coords: VolumeCoords, center : Vec3, radius : Scalar) -> VolumeField:    
    center = __pointTensor(center, coords.device)#{3, 1, 1, 1}    
    p = coords.xyz #{3, depth, height, width}
   
    dp = p - center #{3, depth, height, width}
    l :torch.Tensor = __norm(dp)#{depth, height, width}

    sdf = l - radius #{depth, height, width}

    return VolumeField(sdf)

def makeBox(coords: VolumeCoords, center : Vec3, half_size : Vec3) -> VolumeField:    
    center = __pointTensor(center, coords.device)#{3, 1, 1, 1}    
    half_size = __pointTensor(half_size, coords.device)#{3, 1, 1, 1}    
    p = coords.xyz #{3, depth, height, width}

    dp = torch.abs(p - center) #{3, depth, height, width}

    #clamp(tensor, min) is equivalent to max(tensor, min)
    l = __max(dp - half_size, 0.0)#{3, depth, height, width}

    sdf = __norm(l) #{depth, height, width}

    return VolumeField(sdf)

def makeBoxRounded(coords: VolumeCoords, center : Vec3, half_size : Vec3, radius : Scalar) -> VolumeField:  
    half_size : torch.Tensor = ensureTensor(half_size, coords.device)#{3}    
    sdf = makeBox(coords, center, half_size - radius).tensor #{depth, height, width}

    sdf = sdf - radius #{depth, height, width}

    return VolumeField(sdf)

def makeTorus(coords: VolumeCoords, center : Vec3, radius : Scalar, tube_radius : Scalar, plane: PLANE) -> VolumeField:    
    center : torch.Tensor = __pointTensor(center, coords.device)#{3, 1, 1, 1}    

    p = coords.xyz #{3, depth, height, width}
   
    dp = p - center #{3, depth, height, width}

    iu = plane.value
    iv = (iu + 1) % 3
    inormal = (iu + 2) % 3    

    a = torch.sqrt(dp[iu]**2 + dp[iv]**2) - radius #{depth, height, width}
    b = dp[inormal] #{depth, height, width}

    sdf = torch.sqrt(a**2 + b**2) - tube_radius #{depth, height, width}

    return VolumeField(sdf)

def makeInfiniteCylinder(coords: VolumeCoords, center : Vec3, radius : Scalar, plane: PLANE) -> VolumeField:    
    center : torch.Tensor = __pointTensor(center, coords.device)#{3, 1, 1, 1}    

    p = coords.xyz #{3, depth, height, width}
   
    dp = p - center #{3, depth, height, width}

    iu = plane.value
    iv = (iu + 1) % 3

    sdf = torch.sqrt(dp[iu]**2 + dp[iv]**2) - radius #{depth, height, width}

    return VolumeField(sdf)

#positive height expands the cone along the positive normal direction
def makeCone(coords: VolumeCoords, tip_point : Vec3, radius : Scalar, height : Scalar, plane: PLANE) -> VolumeField:    
    tip_point : torch.Tensor = __pointTensor(tip_point, coords.device)#{3, 1, 1, 1}    

    p = coords.xyz #{3, depth, height, width}
   
    dp = p - tip_point #{3, depth, height, width}

    iu = plane.value
    iv = (iu + 1) % 3
    inormal = (iu + 2) % 3    

    qx = radius
    qy = -ensureTensor(height, coords.device)

    wx = torch.sqrt(dp[iu]**2 + dp[iv]**2)
    wy = dp[inormal]

    wq = wx*qx + wy*qy
    qq = qx*qx + qy*qy
    wqq = torch.clamp(wq/qq, 0.0, 1.0)

    ax = wx - qx*wqq
    ay = wy - qy*wqq

    bx = wx - qx * torch.clamp(wx/qx, 0.0, 1.0)
    by = wy - qy

    k = torch.sign(qy)

    aa = ax**2 + ay**2
    bb = bx**2 + by**2

    d = torch.min(aa, bb)

    s0 = k * (wx*qy - wy*qx)
    s1 = k * (wy - qy)

    s = torch.max(s0, s1)

    sdf = torch.sqrt(d) * torch.sign(s) #{depth, height, width}

    return VolumeField(sdf)

def makePlane(coords: VolumeCoords, origin : Vec3, normal : Vec3) -> VolumeField:    
    origin : torch.Tensor = __pointTensor(origin, coords.device)#{3, 1, 1, 1}    
    normal : torch.Tensor = __pointTensor(normal, coords.device)#{3, 1, 1, 1}
    normal = __normalize(normal)#{3, 1, 1, 1}

    p = coords.xyz #{3, depth, height, width}

    dp = p - origin #{3, depth, height, width}

    l = __dot(dp, normal) #{depth, height, width}

    return VolumeField(l)


def makeCapsule(coords: VolumeCoords, p1 : Vec3, p2 : Vec3, radius : Scalar) -> VolumeField:
    p1 : torch.Tensor = __pointTensor(p1, coords.device)#{3, 1, 1, 1}    
    p2 : torch.Tensor = __pointTensor(p2, coords.device)#{3, 1, 1, 1}    

    p = coords.xyz #{3, depth, height, width}

    pa = p - p1 #{3, depth, height, width}
    ba = p2 - p1 #{3, 1, 1, 1}

    h = __dot(pa, ba) / __dot(ba, ba) #{depth, height, width}
    h = torch.clamp(h, 0.0, 1.0) #{depth, height, width}

    d = pa - ba*h #{3, depth, height, width}

    sdf = __norm(d) - radius #{depth, height, width}

    return VolumeField(sdf)

def makeCylinder(coords: VolumeCoords, p1 : Vec3, p2 : Vec3, radius : Scalar) -> VolumeField:
    p1 : torch.Tensor = __pointTensor(p1, coords.device)#{3, 1, 1, 1}    
    p2 : torch.Tensor = __pointTensor(p2, coords.device)#{3, 1, 1, 1}    

    p = coords.xyz #{3, depth, height, width}
    
    ba = p2 - p1 #{3, 1, 1, 1}
    pa = p - p1 #{3, depth, height, width}

    baba = __dot(ba, ba) #{depth, height, width}
    paba = __dot(pa, ba) #{depth, height, width}

    x = __norm(pa*baba - ba*paba) - radius*baba
    y = torch.abs(paba - baba*0.5) - baba*0.5

    x2 = x**2
    y2 = y**2*baba

    xx2 = torch.zeros_like(x2)
    yy2 = torch.zeros_like(y2)

    mask_x2 = x>0.0
    mask_y2 = y>0.0

    xx2[mask_x2] = x2[mask_x2]
    yy2[mask_y2] = y2[mask_y2]

    d = xx2 + yy2

    mask_d = torch.max(x, y) < 0.0
    d[mask_d] = -torch.min(x2, y2)[mask_d]

    sdf = torch.sign(d)*torch.sqrt(torch.abs(d)) / baba

    return VolumeField(sdf)


def makeRoundCone(coords: VolumeCoords, p1 : Vec3, p2 : Vec3, radius1 : Scalar, radius2 : Scalar) -> VolumeField:
    p1 : torch.Tensor = __pointTensor(p1, coords.device)#{3, 1, 1, 1}    
    p2 : torch.Tensor = __pointTensor(p2, coords.device)#{3, 1, 1, 1}    

    p = coords.xyz #{3, depth, height, width}
    
    ba = p2 - p1 #{3, 1, 1, 1}
    l2 = __dot(ba, ba) 
    rr = ensureTensor(radius1 - radius2, coords.device)
    a2 = l2 - rr*rr
    il2 = 1.0 / l2

    pa = p - p1 #{3, depth, height, width}
    y = __dot(pa, ba)
    z = y - l2

    x2 = pa*l2 - ba*y
    x2 = __dot(x2, x2)

    y2 = y*y*l2
    z2 = z*z*a2

    k = torch.sign(rr)*rr*rr*x2
    sdf = (torch.sqrt(x2*a2*il2) + y*rr)*il2 - radius1

    mask1 = torch.sign(y)*a2*y2<k
    sdf[mask1] = torch.sqrt(x2[mask1] + y2[mask1])*il2 - radius1

    mask2 = torch.sign(z)*a2*z2>k
    sdf[mask2] = torch.sqrt(x2[mask2] + z2[mask2])*il2 - radius2

    return VolumeField(sdf)


def sdfToDensityLinear(sdf: Union[VolumeField , torch.Tensor], r1 : Scalar, r2 : Scalar) -> Union[VolumeField , torch.Tensor]:
    x = sdf.tensor if isinstance(sdf, VolumeField) else sdf
    x = (1.0 - (x - r1)/(r2 - r1)).clamp(0.0, 1.0)
    return VolumeField(x) if isinstance(sdf, VolumeField) else x


def sdfToDensityCubic(sdf: Union[VolumeField , torch.Tensor], r1 : Scalar, r2 : Scalar) -> Union[VolumeField , torch.Tensor]:
    x = sdf.tensor if isinstance(sdf, VolumeField) else sdf
    x = (1.0 - (x - r1)/(r2 - r1)).clamp(0.0, 1.0)
    x = 3.0*x**2 - 2.0*x**3
    return VolumeField(x) if isinstance(sdf, VolumeField) else x
    
def sdfToDensityGaussian(sdf: Union[VolumeField , torch.Tensor], r1 : Scalar, r2 : Scalar, decay : Scalar = 5.0) -> Union[VolumeField , torch.Tensor]:
    x = sdf.tensor if isinstance(sdf, VolumeField) else sdf
    x = torch.clamp((x - r1)/(r2 - r1), min = 0.0)
    x = torch.exp(-(x**2)/(2.0*decay**2))
    return VolumeField(x) if isinstance(sdf, VolumeField) else x