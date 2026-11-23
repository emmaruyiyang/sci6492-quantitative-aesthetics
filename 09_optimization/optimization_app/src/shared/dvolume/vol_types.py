from typing import Union
import torch
import numpy as np

Vec3 = Union[torch.Tensor, list[float], tuple[float, float, float]]
Idx3 = Union[torch.Tensor, list[int], tuple[int, int, int]]
Scalar = Union[torch.Tensor, float]

def ensureTensor(thing, device : torch.device) -> torch.Tensor:  
    if isinstance(thing, torch.Tensor):
        if thing.device != device:
            return thing.to(device)
        return thing
    
    if isinstance(thing, np.ndarray):
        return torch.tensor(thing, dtype=torch.float32, device=device)
    
    if isinstance(thing, list):
        return torch.tensor(thing, dtype=torch.float32, device=device)
    
    if isinstance(thing, tuple):
        return torch.tensor(thing, dtype=torch.float32, device=device)
    
    if isinstance(thing, float):
        return torch.tensor([thing], dtype=torch.float32, device=device)
    
    
    raise ValueError('Invalid type for tensor conversion.' + str(type(thing)))

def ensureNumpy(thing):
    if isinstance(thing, np.ndarray):
        return thing
    
    if isinstance(thing, list):
        return np.array(thing, dtype=np.float32)
    
    if isinstance(thing, tuple):
        return np.array(thing, dtype=np.float32)
    
    if isinstance(thing, torch.Tensor):
        return thing.detach().cpu().numpy()
    
    raise ValueError('Invalid type for numpy conversion.' + str(type(thing)))

def ensureList(thing):
    if isinstance(thing, list):
        return thing
    
    if isinstance(thing, tuple):
        return list(thing)
    
    if isinstance(thing, np.ndarray):
        return thing.tolist()
    
    if isinstance(thing, torch.Tensor):
        return thing.detach().cpu().numpy().tolist()
    
    raise ValueError('Invalid type for list conversion.' + str(type(thing)))