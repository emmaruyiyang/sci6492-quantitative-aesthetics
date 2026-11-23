#this software is provided as is, and is not guaranteed to work or be suitable for any particular purpose
#for use only in the context of the "Machine Aesthetics" course at GSD. Do not share or distribute
#copyright 2023-2024 Panagiotis Michalatos : pan.michalatos@gmail.com

import torch
import numpy as np
import torch.nn.functional as F


class Vol:
    device : torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    gradient_kernel_tensor : torch.Tensor = None
        
    @staticmethod
    def initialize(device : torch.device = None):
        if device is not None:
            Vol.device = device


        with torch.no_grad():
            sobel_k_y = np.array([
                            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                            [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
                        ], dtype=np.float32)

            sobel_k_z = sobel_k_y.transpose((1, 0, 2))
            sobel_k_x = sobel_k_y.transpose((0, 2, 1))

            gradient_kernel = np.array([sobel_k_x, sobel_k_y, sobel_k_z]).reshape((3, 1, 3, 3, 3))

            Vol.gradient_kernel_tensor = torch.from_numpy(gradient_kernel).to(Vol.device)

    def computeGradient(tensor : torch.Tensor, normalized : bool = True) -> torch.Tensor:
        if Vol.gradient_kernel_tensor is None:
            Vol.initialize(tensor.device)
            
        kernel =  Vol.gradient_kernel_tensor
        if kernel.device != tensor.device:
            kernel = kernel.to(tensor.device)
            
        gradient = F.conv3d(tensor, kernel, stride=1, padding=1) 

        if normalized:            
            grad_norm = torch.norm(gradient, dim=1, keepdim=True)
            grad_norm = grad_norm + 1e-5
            gradient = gradient / grad_norm

        return gradient
    
    def computeLaplacian(tensor : torch.Tensor) -> torch.Tensor:
        laplacian_kernel = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]], 
                                         [[0, 1, 0], [1, -6, 1], [0, 1, 0]], 
                                         [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32, device=tensor.device)
        
        laplacian_kernel = laplacian_kernel.reshape((1, 1, 3, 3, 3))
        laplacian = F.conv3d(tensor, laplacian_kernel, stride=1, padding=1)

        return laplacian
    
    def computeGradient2d(tensor : torch.Tensor, normalized : bool = True) -> torch.Tensor:
        sobel_k_x = torch.tensor([
            [2.0, 0.0, -2.0], 
            [4.0, 0.0, -4.0], 
            [2.0, 0.0, -2.0]], 
            dtype=torch.float32, 
            device=tensor.device)
        
        sobel_k_y = torch.tensor([
            [2.0, 4.0, 2.0], 
            [0.0, 0.0, 0.0], 
            [-2.0, -4.0, -2.0]], 
            dtype=torch.float32, 
            device=tensor.device)
        
        sobel_k = torch.cat([sobel_k_x.unsqueeze(0), sobel_k_y.unsqueeze(0)], 0)
        sobel_k = sobel_k.unsqueeze(1)
        sobel_k.requires_grad = False
            
        gradient = F.conv2d(tensor, sobel_k, stride=1, padding=1) 

        if normalized:            
            grad_norm = torch.norm(gradient, dim=1, keepdim=True)
            grad_norm = grad_norm + 1e-5
            gradient = gradient / grad_norm

        return gradient
    
    def computeImageNormalMap2d(tensor : torch.Tensor, relief_strength: float, normalized : bool = True) -> torch.Tensor:
        gradient = Vol.computeGradient2d(tensor, False)
        gradient = torch.cat([gradient * relief_strength, torch.ones_like(tensor)], dim=1)
        
        if normalized:
            grad_norm = torch.norm(gradient, dim=1, keepdim=True)
            grad_norm = grad_norm + 1e-5
            gradient = gradient / grad_norm

        return gradient
