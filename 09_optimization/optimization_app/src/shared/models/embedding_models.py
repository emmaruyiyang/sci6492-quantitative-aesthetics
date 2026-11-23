"""Pretrained image encoders (CLIP, DINOv2, torchvision ImageNet classifiers):
how to load each one and the preprocessing it expects, keyed by ModelID.

Shared by ModelCentral (frozen oracles for the losses) and custom_models
(classifier backbones, the pretrained embedders of model_central_app.py), so
both build the same networks the same way.

  load_model(model_id)          the whole network, as its library returns it
                                (open_clip model with encode_image/encode_text,
                                DINOv2 hub model, torchvision classifier)
  load_feature_extractor(...)   a module that maps normalized images -> feature
                                vectors, and the feature size. For ImageNet
                                classifiers the last 1000-class layer is
                                replaced by Identity.
  image_normalization(...)      (mean, std) of the images it was trained on
"""

import torch
import torch.nn as nn

from .model_types import CLIPArch, CLIPPretrained, DINOV2Arch, HUB_CLIP_WEIGHTS, ImageNetArch, ModelID, is_siglip

INPUT_SIZE = 224        # every encoder here takes 224x224 (DINOv2: a multiple of its 14px patches)

# Mean and standard deviation of the pixel colors (R, G, B) of the images each
# encoder was trained on. Inputs must be normalized the same way, otherwise the
# network sees colors shifted from what it learned.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)       # also DINOv2's
IMAGENET_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)                # SigLIP: pixels to -1..1
SIGLIP_STD = (0.5, 0.5, 0.5)


def image_normalization(model_id: ModelID) -> tuple[tuple, tuple]:
    if model_id.is_clip():
        return (SIGLIP_MEAN, SIGLIP_STD) if is_siglip(model_id.arch) else (CLIP_MEAN, CLIP_STD)
    return IMAGENET_MEAN, IMAGENET_STD


def default_imagenet_weights(arch: str | ImageNetArch) -> str:
    """Name of the best torchvision weights of an architecture, e.g. 'IMAGENET1K_V2'."""
    import torchvision.models
    return torchvision.models.get_model_weights(str(arch)).DEFAULT.name


# ── Loading ────────────────────────────────────────────────────────────────────

def load_clip(arch: str | CLIPArch, pretrained: str | CLIPPretrained) -> nn.Module:
    import open_clip    # imported here so the other encoders work without it

    arch, pretrained = str(arch), str(pretrained)
    if (arch, pretrained) in HUB_CLIP_WEIGHTS:
        return open_clip.create_model(HUB_CLIP_WEIGHTS[arch, pretrained])
    available = [tag for name, tag in open_clip.list_pretrained() if name == arch]
    if pretrained not in available:
        raise ValueError(f"CLIP {arch} has no '{pretrained}' weights. Available: {available}")
    # Weights trained with QuickGELU (the 'openai' ones, among others) under a plain arch name, whose
    # config says standard GELU: open_clip only warns, and would run them with the wrong activation.
    quick_gelu = open_clip.get_pretrained_cfg(arch, pretrained).get("quick_gelu", False)
    return open_clip.create_model(arch, pretrained=pretrained, force_quick_gelu=quick_gelu)


def load_dino(arch: str | DINOV2Arch) -> nn.Module:
    # Downloaded on first use into the torch hub cache (~/.cache/torch/hub).
    # Calling the model returns the embedding of the [CLS] token.
    return torch.hub.load("facebookresearch/dinov2", str(arch))


def load_imagenet(arch: str | ImageNetArch, weights: str = "") -> nn.Module:
    import torchvision.models
    return torchvision.models.get_model(str(arch), weights=weights or default_imagenet_weights(arch))


def load_model(model_id: ModelID) -> nn.Module:
    if model_id.is_clip():
        return load_clip(model_id.arch, model_id.pretrained)
    if model_id.is_dino():
        return load_dino(model_id.arch)
    if model_id.is_imagenet():
        return load_imagenet(model_id.arch, model_id.pretrained)
    raise ValueError(f"unknown model family {model_id.family}")


# ── Feature extractors ────────────────────────────────────────────────────────

class CLIPImageEncoder(nn.Module):
    """CLIP is two networks, one for images and one for text. Only the image one."""

    def __init__(self, visual: nn.Module):
        super().__init__()
        self.visual = visual

    def forward(self, x):
        return self.visual(x)


def clip_feature_size(visual: nn.Module) -> int:
    """The size of an open_clip image tower's vectors. Its own vision transformers
    say so; a timm tower (SigLIP) doesn't, it has no projection: its trunk's width."""
    size = getattr(visual, "output_dim", None)
    return visual.trunk.num_features if size is None else size


# Where each torchvision architecture keeps its final 1000-class layer:
# (attribute path, index in it or None)
_IMAGENET_HEADS = {
    ImageNetArch.ALEXNET: ("classifier", 6),                # 4096 features
    ImageNetArch.VGG16: ("classifier", 6),                  # 4096
    ImageNetArch.RESNET18: ("fc", None),                    # 512
    ImageNetArch.RESNET50: ("fc", None),                    # 2048
    ImageNetArch.RESNET101: ("fc", None),                   # 2048
    ImageNetArch.GOOGLENET: ("fc", None),                   # 1024
    ImageNetArch.MOBILENET_V3_LARGE: ("classifier", 3),     # 1280
    ImageNetArch.EFFICIENTNET_B0: ("classifier", 1),        # 1280
    ImageNetArch.CONVNEXT_TINY: ("classifier", 2),          # 768
    ImageNetArch.CONVNEXT_BASE: ("classifier", 2),          # 1024
    ImageNetArch.SWIN_T: ("head", None),                    # 768
    ImageNetArch.VIT_B_32: ("heads", "head"),               # 768
    ImageNetArch.VIT_B_16: ("heads", "head"),               # 768
}


def strip_imagenet_head(model: nn.Module, arch: str | ImageNetArch) -> tuple[nn.Module, int]:
    """Replace the final 1000-class layer with Identity, so the model returns the
    features that were feeding into it. -> (model, feature size)"""
    arch = ImageNetArch(str(arch))
    if arch not in _IMAGENET_HEADS:
        raise ValueError(f"don't know where the last layer of {arch} is")
    name, index = _IMAGENET_HEADS[arch]
    parent = getattr(model, name)
    if index is None:
        feature_size = parent.in_features
        setattr(model, name, nn.Identity())
    elif isinstance(index, str):
        feature_size = getattr(parent, index).in_features
        setattr(parent, index, nn.Identity())
    else:
        feature_size = parent[index].in_features
        parent[index] = nn.Identity()
    return model, feature_size


def load_feature_extractor(model_id: ModelID) -> tuple[nn.Module, int]:
    """-> (module that maps normalized images -> feature vectors, feature size)"""
    if model_id.is_clip():
        model = load_clip(model_id.arch, model_id.pretrained)
        return CLIPImageEncoder(model.visual), clip_feature_size(model.visual)
    if model_id.is_dino():
        model = load_dino(model_id.arch)
        return model, model.embed_dim
    if model_id.is_imagenet():
        return strip_imagenet_head(load_imagenet(model_id.arch, model_id.pretrained), model_id.arch)
    raise ValueError(f"unknown model family {model_id.family}")
