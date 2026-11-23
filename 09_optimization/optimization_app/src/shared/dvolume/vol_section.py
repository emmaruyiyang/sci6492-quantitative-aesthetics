from .vol_field import VolumeField
from enum import Enum
import torch
from torchvision import transforms

class SectionPlane(Enum):
    XY = 'xy'
    YZ = 'yz'
    ZX = 'zx'

class SectionDefinition:
    def __init__(self, plane: SectionPlane, positions: list[float], losses: list, contrast: float, bg_color :tuple[float,float,float], fg_color :tuple[float,float,float], name: str):
        self.plane = plane
        self.positions = positions
        self.losses = losses
        self.contrast = contrast
        self.bg_color = bg_color
        self.fg_color = fg_color    
        self.name = name

    def get_section_mask_tensor(self, v: VolumeField, size: tuple[int,int] = None, flip_y: bool = True) -> torch.Tensor:
        
        slices = []
        for pos in self.positions:
     
            if self.plane == SectionPlane.XY:
                z = pos * (v.nz - 1)
                slice = v.tensor[0,0,int(z), :, :]
            elif self.plane == SectionPlane.ZX:
                y = pos * (v.ny - 1)
                slice = v.tensor[0,0,:, int(y), :]
            elif self.plane == SectionPlane.YZ:
                x = pos * (v.nx - 1)
                slice = v.tensor[0,0,:, :, int(x)]

            if slice is None:
                continue
            
            if size is not None:
                slice = transforms.Resize(size)(slice.unsqueeze(0)).squeeze(0)

            if self.contrast > 0.0:
                slice = torch.sigmoid((slice-0.5)*self.contrast)

            if flip_y:
                slice = torch.flip(slice, [0])
        
            slices.append(slice)

        if len(slices) == 0:
            return None
        
        slice = torch.stack(slices, dim=0).mean(dim=0)

        return slice


    def get_section_image_tensor(self, v: VolumeField, size: tuple[int,int] = None) -> torch.Tensor:
  
        sbg = torch.tensor(self.bg_color, device=v.device).view(3, 1, 1)
        sfg = torch.tensor(self.fg_color, device=v.device).view(3, 1, 1)

        slice = self.get_section_mask_tensor(v, size)

        if slice is None:
            return None

        slice = slice.unsqueeze(0).repeat(3, 1, 1)    
        slice_rgb = slice * sfg + (1.0 - slice) * sbg

        return slice_rgb