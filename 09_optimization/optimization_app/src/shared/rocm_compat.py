"""ROCm workarounds, applied on import of `shared`. A no-op on CUDA, MPS and CPU.

Fused attention with head_dim > 256: on ROCm (seen on torch 2.14.1+rocm7.2,
gfx1151) the flash and memory-efficient scaled_dot_product_attention kernels
return garbage for head_dim 512, silently, in fp32 and bf16. head_dim <= 256
is exact. The FLUX.2 / SD VAEs' mid-block attention is one 512-wide head; the
CLIP / DINOv2 / ViT encoders use 64 and keep the fused kernels. Calls above
the limit go to the exact math backend instead.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

MAX_FUSED_HEAD_DIM = 256


def _install() -> None:
    if not torch.version.hip or getattr(F.scaled_dot_product_attention, "_rocm_safe", False):
        return
    fused = F.scaled_dot_product_attention

    def scaled_dot_product_attention(query, key, value, *args, **kwargs):
        if query.is_cuda and max(query.shape[-1], value.shape[-1]) > MAX_FUSED_HEAD_DIM:
            with sdpa_kernel(SDPBackend.MATH):
                return fused(query, key, value, *args, **kwargs)
        return fused(query, key, value, *args, **kwargs)

    scaled_dot_product_attention._rocm_safe = True
    scaled_dot_product_attention.__doc__ = fused.__doc__
    F.scaled_dot_product_attention = scaled_dot_product_attention


_install()
