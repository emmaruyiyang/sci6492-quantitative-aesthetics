"""Lightweight model-identity types: no torch/torchvision/transformers/diffusers
import, so this module is safe to pull into anything that needs to name a model
without paying for the ability to actually load one — notably
``diff_draw.optimizer_config``, which registers the CLIP/DINO arch+pretrained
and SDS-model defaults from these enums directly (see its ``default_clip_model_id``
/ ``default_dino_model_id`` and the ``sds/model`` option). ``model_central.py``
and ``model_diffusion_central.py`` re-export these names (via
``from .model_types import ...``) so existing call sites are unaffected.
"""

from dataclasses import dataclass
from enum import Enum

# ── ModelID ────────────────────────────────────────────────────────────────────

_MODEL_ID_SEP = '~'


class DINOV2Arch(Enum):
    VITS14 = 'dinov2_vits14'
    VITB14 = 'dinov2_vitb14'
    VITL14 = 'dinov2_vitl14'
    VITG14 = 'dinov2_vitg14'

    def __str__(self) -> str:
        return self.value


class CLIPArch(Enum):
    VITB32 = 'ViT-B-32'
    VITL14 = 'ViT-L-14'

    def __str__(self) -> str:
        return self.value


# open_clip's full `pretrained` list for reference (most left commented out —
# uncomment as needed, they just aren't wired up as CLIPPretrained members yet):
# ['openai', 'laion400m_e31', 'laion400m_e32', 'laion2b_e16', 'laion2b_s34b_b79k',
#  'datacomp_xl_s13b_b90k', 'datacomp_m_s128m_b4k', 'commonpool_m_clip_s128m_b4k',
#  'commonpool_m_laion_s128m_b4k', 'commonpool_m_image_s128m_b4k',
#  'commonpool_m_text_s128m_b4k', 'commonpool_m_basic_s128m_b4k',
#  'commonpool_m_s128m_b4k', 'datacomp_s_s13m_b4k', 'commonpool_s_clip_s13m_b4k',
#  'commonpool_s_laion_s13m_b4k', 'commonpool_s_image_s13m_b4k',
#  'commonpool_s_text_s13m_b4k', 'commonpool_s_basic_s13m_b4k',
#  'commonpool_s_s13m_b4k', 'metaclip_400m', 'metaclip_fullcc']
class CLIPPretrained(Enum):
    OPENAI = 'openai'

    #LAION400M_E31 = 'laion400m_e31'
    #LAION400M_E32 = 'laion400m_e32'
    #LAION2B_E16 = 'laion2b_e16'

    LAION2B_S34B_B79K = 'laion2b_s34b_b79k'
    DATACOMP_XL_S13B_B90K = 'datacomp_xl_s13b_b90k'
    #DATACOMP_M_S128M_B4K = 'datacomp_m_s128m_b4k'
    #DATACOMP_S_S13M_B4K = 'datacomp_s_s13m_b4k'

    #Common pool
    # COMMONPOOL_M_CLIP_S128M_B4K = 'commonpool_m_clip_s128m_b4k'
    # COMMONPOOL_M_LAION_S128M_B4K = 'commonpool_m_laion_s128m_b4k'
    # COMMONPOOL_M_IMAGE_S128M_B4K = 'commonpool_m_image_s128m_b4k'
    # COMMONPOOL_M_TEXT_S128M_B4K = 'commonpool_m_text_s128m_b4k'
    # COMMONPOOL_M_BASIC_S128M_B4K = 'commonpool_m_basic_s128m_b4k'
    # COMMONPOOL_M_S128M_B4K = 'commonpool_m_s128m_b4k'
    # COMMONPOOL_S_CLIP_S13M_B4K = 'commonpool_s_clip_s13m_b4k'
    # COMMONPOOL_S_LAION_S13M_B4K = 'commonpool_s_laion_s13m_b4k'
    # COMMONPOOL_S_IMAGE_S13M_B4K = 'commonpool_s_image_s13m_b4k'
    # COMMONPOOL_S_TEXT_S13M_B4K = 'commonpool_s_text_s13m_b4k'
    # COMMONPOOL_S_BASIC_S13M_B4K = 'commonpool_s_basic_s13m_b4k'
    # COMMONPOOL_S_S13M_B4K = 'commonpool_s_s13m_b4k'

    #METACLIP_400M = 'metaclip_400m'
    METACLIP_FULLCC = 'metaclip_fullcc'

    def __str__(self) -> str:
        return self.value


# torchvision ImageNet-1k classifiers (all take 224x224 input). The weights tag
# (e.g. 'IMAGENET1K_V2') goes in ModelID.pretrained, so caches made with different
# weights of the same architecture never collide.
class ImageNetArch(Enum):
    ALEXNET = 'alexnet'
    VGG16 = 'vgg16'
    RESNET18 = 'resnet18'
    RESNET50 = 'resnet50'
    RESNET101 = 'resnet101'
    GOOGLENET = 'googlenet'
    MOBILENET_V3_LARGE = 'mobilenet_v3_large'
    EFFICIENTNET_B0 = 'efficientnet_b0'
    CONVNEXT_TINY = 'convnext_tiny'
    CONVNEXT_BASE = 'convnext_base'
    SWIN_T = 'swin_t'
    VIT_B_32 = 'vit_b_32'
    VIT_B_16 = 'vit_b_16'

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ModelID:
    """Frozen, hashable model identifier usable as a dict key and filename stem.

    String format (4 fields, '~'-separated):
      {family}~{arch}~{pretrained}~{extra}

    Examples:
      clip~ViT-B-32~openai
      dino~dinov2_vitb14
      imagenet~resnet50~IMAGENET1K_V2
    """
    family: str      # 'clip' | 'dino' | 'imagenet' | 'custom'
    arch: str        # e.g. 'ViT-B-32', 'dinov2_vitb14', 'resnet50'; custom: the model's id
    pretrained: str  # e.g. 'openai', '' for dino, torchvision weights tag for imagenet
    extra: str = ''  # reserved for future use

    def __str__(self) -> str:
        base = f"{self.family}{_MODEL_ID_SEP}{self.arch}"
        if self.pretrained or self.extra: #always include pretrained if extra is present, to preserve field order
            base += f"{_MODEL_ID_SEP}{self.pretrained}"

        if self.extra:
            base += f"{_MODEL_ID_SEP}{self.extra}"

        return base

    def filename_stem(self) -> str:
        """Filesystem-safe version of __str__() with '/' replaced by '_'."""
        return str(self).replace('/', '_')

    @staticmethod
    def from_string(s: str) -> 'ModelID':
        parts = s.split(_MODEL_ID_SEP, 3)  # limit to 4 so extra may contain '~'
        if not (2 <= len(parts) <= 4):
            raise ValueError(f"Cannot parse ModelID from {s!r} — expected 2–4 '~'-separated fields")
        while len(parts) < 4:
            parts.append('')
        return ModelID(*parts)

    @staticmethod
    def clip(arch: str | CLIPArch, pretrained: str | CLIPPretrained, extra: str = '') -> 'ModelID':
        return ModelID('clip', str(arch), str(pretrained), extra)

    @staticmethod
    def dino(arch: str | DINOV2Arch, extra: str = '') -> 'ModelID':
        return ModelID('dino', str(arch), '', extra)

    @staticmethod
    def imagenet(arch: str | ImageNetArch, weights: str, extra: str = '') -> 'ModelID':
        """torchvision classifier; `weights` is the enum member name, e.g. 'IMAGENET1K_V1'.
        Its "embedding" is the raw 1000-class logit vector."""
        return ModelID('imagenet', str(arch), weights, extra)

    @staticmethod
    def custom(model_id: str) -> 'ModelID':
        """A custom model (shared.custom_models: a classifier, autoencoder, RankNet or
        aesthetic score), by its id in custom_models.differentiable.list_models(),
        e.g. 'folder/my_classifier', 'aesthetic/laion_ava'."""
        return ModelID('custom', model_id, '')

    def is_custom(self) -> bool:
        return self.family == 'custom'

    def is_clip(self) -> bool:
        return self.family == 'clip'

    def is_dino(self) -> bool:
        return self.family == 'dino'

    def is_imagenet(self) -> bool:
        return self.family == 'imagenet'



def default_clip_model_id() -> ModelID:
    return ModelID.clip(CLIPArch.VITB32, CLIPPretrained.OPENAI)


def default_dino_model_id() -> ModelID:
    return ModelID.dino(DINOV2Arch.VITB14)

# ── Differential diffusion model identity ─────────────────────────────────────
# See model_diffusion_central.py's DifferentialDiffusionModel for the actual
# implementation ("Giving Each Pixel Its Strength", Levin & Fried) — these are
# just the plain enum/set naming which model, so optimizer_config.py can build
# its 'sds/model' option straight from DifferentialDiffusionModelType's members
# rather than a hand-copied string list.

class DifferentialDiffusionModelType(Enum):
    #SD_15_TINY = "segmind/tiny-sd"
    SD_15_SMALL = "segmind/small-sd"
    SD_15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # full SD-1.5 (required for IP-Adapter)
    # SD-Turbo (SD-2.1-base architecture, same 4-ch VAE + eps UNet). ADD-distilled,
    # so it produces good scores at LOW guidance — for SDS this sidesteps the
    # oversaturation you get from cranking guidance_scale on base SD-1.5. Run
    # sds_loss on it with guidance_scale ~= 1.0-3.0. Not IP-Adapter compatible
    # (the SD-1.5 adapter weights don't fit the SD-2.1 UNet).
    SD_TURBO = "stabilityai/sd-turbo"
    # SD-1.5 fine-tuned by Justin Pinkney at native 256x256 (vs. SD-1.5's 512x512).
    # Same full UNet + 4-ch VAE as SD_15 (just re-tuned weights), so it is IP-Adapter
    # compatible and uses the same SDS/DDPM conventions. Pair with a smaller
    # loss_frame_size (~256) — running it at 512 forces the UNet outside the
    # resolution it was fine-tuned for.
    # The repo ships a single original-format .ckpt (no diffusers model_index.json),
    # so this is a direct file URL loaded via from_single_file rather than a repo id
    # (see the from_single_file branch in DifferentialDiffusionModel.__init__).
    MINI_SD = "https://huggingface.co/justinpinkney/miniSD/blob/main/miniSD.ckpt"


# IP-Adapter weights are trained for the full SD-1.5 UNet. The distilled segmind models
# prune attention blocks, so the adapter's cross-attention weights don't fit them.
IP_ADAPTER_COMPATIBLE = {DifferentialDiffusionModelType.SD_15, DifferentialDiffusionModelType.MINI_SD}
