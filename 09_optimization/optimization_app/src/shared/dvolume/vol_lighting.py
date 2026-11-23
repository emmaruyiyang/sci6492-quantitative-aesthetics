import torch
from .vol_types import *
from typing import Optional

class VolLighting:
    def __init__(self, 
                 light_dir : torch.Tensor,              #{3}
                 ambient_color : Optional[torch.Tensor] = None,   #{3}
                 isPointLight : bool = False,
                 specular_power : float = 16.0,
                 device : Optional[torch.device] = None   
                 ):
        
        if device is None and not isinstance(light_dir, torch.Tensor):
            raise ValueError("device must be specified for lighting")
        
        self.device = light_dir.device or device

        self.isPointLight =  isPointLight
        
        self.light_dir = ensureTensor(light_dir, self.device)
        self.specular_power = specular_power

        if not self.isPointLight:
            self.light_dir = self.light_dir / torch.norm(self.light_dir)
        self.ambient_color = ensureTensor(ambient_color, self.device) if ambient_color is not None else None

    def setDirection(self, x : float, y : float, z : float, isPointLight : bool = None):
        if isPointLight is not None:
            self.isPointLight = isPointLight

        self.light_dir = torch.tensor([x, y, z], dtype=torch.float32, device=self.device)

        if not self.isPointLight:
            self.light_dir = self.light_dir / torch.norm(self.light_dir)