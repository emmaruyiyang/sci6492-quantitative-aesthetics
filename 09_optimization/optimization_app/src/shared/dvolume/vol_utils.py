import torch
import numpy as np
from PIL import Image


def save_tensor_image(tensor : torch.Tensor, filename):
    shape = tensor.shape
    tensor_np = tensor.detach().cpu().numpy()
    if len(shape) == 4:
        tensor_np = tensor_np[0, :, :, :]

    channels = tensor_np.shape[0]

    if channels == 1:
        tensor_np = tensor_np[0, :, :]
        tensor_np = np.clip(tensor_np, 0.0, 1.0)
        tensor_np = tensor_np*255.0
        tensor_np = tensor_np.astype(np.uint8)
        tensor_pil = Image.fromarray(tensor_np, mode = 'L')
    else:
        tensor_np = np.transpose(tensor_np, (1, 2, 0))
        tensor_np = np.clip(tensor_np, 0.0, 1.0)
        tensor_np = tensor_np*255.0
        tensor_np = tensor_np.astype(np.uint8)
        tensor_pil = Image.fromarray(tensor_np)

    tensor_pil.save(filename)
