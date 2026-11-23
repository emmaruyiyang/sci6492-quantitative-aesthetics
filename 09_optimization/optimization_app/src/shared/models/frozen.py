"""What the frozen, differentiable image models share (custom_models.differentiable,
models.vlm): an image batch (B, 3, H, W) in 0..1 made into a model's input with
differentiable tensor operations, the precisions a model loads in, and a cache of the
loaded models, kept until released.
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

PRECISIONS = {"fp32": torch.float32, "bf16": torch.bfloat16}
GPU_MEMORY_FRACTION = 0.9   # cap_gpu_memory's default share of each gpu's memory


def cap_gpu_memory(fraction: float = GPU_MEMORY_FRACTION) -> None:
    """Cap this process's PyTorch allocations at `fraction` of every cuda gpu's memory: past
    it an allocation raises CUDA out of memory, rather than growing to the card's limit
    (on Windows the driver may then spill into system RAM and freeze the machine), and the
    rest is left to the desktop and other programs. Call it at startup, before the models load."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(fraction, i)


def fit(x: torch.Tensor, size: int) -> torch.Tensor:
    """Short side to size (antialiased bilinear), center square crop: (B, 3, size, size)"""
    h, w = x.shape[-2:]
    scale = size / min(h, w)
    nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
    if (nh, nw) != (h, w):
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", antialias=nh < h, align_corners=False)
    top, left = (nh - size) // 2, (nw - size) // 2
    return x[..., top:top + size, left:left + size]


def normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    return (x - x.new_tensor(mean).view(1, 3, 1, 1)) / x.new_tensor(std).view(1, 3, 1, 1)


def frozen(module: nn.Module) -> nn.Module:
    """No gradient for its weights (only for its input), eval mode."""
    return module.eval().requires_grad_(False)


def precision_dtype(precision: str) -> torch.dtype:
    if precision not in PRECISIONS:
        raise ValueError(f"bad precision {precision!r}: one of {list(PRECISIONS)}")
    return PRECISIONS[precision]


class ModelCache:
    """Loaded models by (id, device, precision), kept until release()."""

    def __init__(self):
        self._models: dict[tuple[str, str, str], nn.Module] = {}

    @staticmethod
    def key(model_id: str, device, precision: str = "fp32") -> tuple[str, str, str]:
        return (model_id, str(device), precision)

    def get(self, key: tuple[str, str, str], make: Callable[[], nn.Module]) -> nn.Module:
        if key not in self._models:
            self._models[key] = make()
        return self._models[key]

    def release(self, keep=()) -> None:
        """Drop the loaded models but those of `keep` (key()s): frees their gpu memory.
        Call it before loading what a run needs, so the old and the new don't share the gpu."""
        for key in [k for k in self._models if k not in keep]:
            del self._models[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
