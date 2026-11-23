"""What the Segmentation Explorer (app_segmentation_server.py) asks of a model,
behind one interface. The counterpart of embedders.py, which only needs one
vector per image.

Families of models:

  classifier   a saved Classifier (classifier_model.py): backbone + head
  autoencoder  a saved Autoencoder: encoder (+ decoder)
  dino         pretrained DINOv2 image encoder
  clip         pretrained CLIP, image and text encoders
  segformer    SegFormer (huggingface transformers): a model trained for
               semantic segmentation, one score per class per pixel
  clipseg      CLIPSeg (transformers): CLIP + a small decoder trained to
               turn a prompt (text or an example image) into a mask

The first four are wrapped in a Backbone:

  embed(x)      images [B, 3, S, S] -> one vector per image [B, D]
                (differentiable: gradients can flow back to the pixels)
  dense(x)      images [B, 3, H, W] -> one vector per patch [B, h, w, D]
                (H, W multiples of `patch`), for the backbones that have it
  attention(x)  -> [B, heads, h, w]: how much the [CLS] token of the last
                layer looks at every patch, one map per attention head (ViTs)

Dense vectors per patch:

  DINOv2     the patch tokens of the last layer. DINOv2 takes any input size
             that is a multiple of 14 (it interpolates its position codes).
  CLIP       CLIP was only trained to match its [CLS] summary with text, its
             patch tokens are not in the text space. ClearCLIP (Lan et al.
             2024) gets them there by changing the last layer: attention of
             every patch to the patches whose query looks like its own query
             (q·q instead of q·k), no residual connection, no MLP. Then the
             usual final norm and projection put every patch in the space of
             the text vectors. The position codes are interpolated for bigger
             inputs.
  ResNet,    the feature map before the global average: one 2048 / 768 / 1024
  ConvNeXt   numbers vector per 32 x 32 pixels. Their average is (about) what
             the classifier head sees.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from autoencoder_model import Autoencoder, read_config
from classifier_model import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, CLIPImageEncoder,
                              Classifier, ClassifierBackbone, ImageNetArch)
from embedders import PRETRAINED
from embedders import find_models as find_embedders

# Segmentation models from huggingface (downloaded on first use):
# id -> (label, huggingface name)
SEGFORMERS = {
    "segformer/b0_ade": ("SegFormer B0, ADE20K 150 classes", "nvidia/segformer-b0-finetuned-ade-512-512"),
    "segformer/b5_ade": ("SegFormer B5, ADE20K 150 classes", "nvidia/segformer-b5-finetuned-ade-640-640"),
}
CLIPSEGS = {
    "clipseg/rd64_refined": ("CLIPSeg (text or image prompt)", "CIDAS/clipseg-rd64-refined"),
}

DENSE_IMAGENET = {ImageNetArch.RESNET50, ImageNetArch.RESNET101,
                  ImageNetArch.CONVNEXT_TINY, ImageNetArch.CONVNEXT_BASE}


# ---------------------------------------------------------------------------
# Images -> tensors
# ---------------------------------------------------------------------------

def to_tensor(images: list[Image.Image], size: tuple[int, int], mean, std, device) -> torch.Tensor:
    """PIL images -> [B, 3, h, w], each squashed to size (w, h) and normalized"""
    w, h = size
    x = torch.stack([TF.to_tensor(img.convert("RGB").resize((w, h), Image.BICUBIC)) for img in images])
    return TF.normalize(x, mean, std).to(device)


def dense_size(width: int, height: int, long_side: int, multiple: int) -> tuple[int, int]:
    """(w, h) with about long_side pixels along the longer side, the image's
    proportions, both multiples of `multiple`"""
    scale = long_side / max(width, height)
    return (max(multiple, round(width * scale / multiple) * multiple),
            max(multiple, round(height * scale / multiple) * multiple))


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

class Backbone:
    size = 224                  # side of the square input of embed()
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    patch = None                # dense() needs input sides that are multiples of this, None = no dense()
    heads = 0                   # attention() maps, 0 = no attention()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def dense(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} has no vectors per patch")

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} has no attention")


def cls_attention(q: torch.Tensor, k: torch.Tensor, scale: float, first: int, h: int, w: int) -> torch.Tensor:
    """q, k [B, heads, tokens, d] -> how much the [CLS] token (token 0) looks
    at every patch (tokens first...), [B, heads, h, w]"""
    a = ((q[:, :, :1] * scale) @ k.transpose(-2, -1)).softmax(-1)[:, :, 0]
    return a[:, :, first:].reshape(len(q), q.shape[1], h, w)


class DinoBackbone(Backbone):

    def __init__(self, model: nn.Module):
        self.model = model
        self.patch = model.patch_size
        self.last = model.blocks[-1]
        while isinstance(self.last, nn.ModuleList):      # blocks in chunks: the last one of the last chunk
            self.last = self.last[-1]
        self.heads = self.last.attn.num_heads
        self.first = 1 + getattr(model, "num_register_tokens", 0)  # [CLS], registers, then the patches

    def embed(self, x):
        return self.model(x)

    def dense(self, x):
        h, w = x.shape[2] // self.patch, x.shape[3] // self.patch
        return self.model.forward_features(x)["x_norm_patchtokens"].reshape(len(x), h, w, -1)

    def attention(self, x):
        # the input of the last attention layer, caught on the way through
        caught = {}
        hook = self.last.attn.register_forward_pre_hook(lambda module, args: caught.update(x=args[0]))
        try:
            self.model(x)
        finally:
            hook.remove()
        t, attn = caught["x"], self.last.attn
        B, N, C = t.shape
        q, k, _ = attn.qkv(t).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        return cls_attention(q, k, attn.scale, self.first, x.shape[2] // self.patch, x.shape[3] // self.patch)


class ClipBackbone(Backbone):
    mean, std = CLIP_MEAN, CLIP_STD

    def __init__(self, visual: nn.Module):
        self.visual = visual
        self.patch = visual.conv1.kernel_size[0]
        self.size = visual.conv1.kernel_size[0] * int(math.sqrt(len(visual.positional_embedding) - 1))
        self.last = visual.transformer.resblocks[-1]
        self.heads = self.last.attn.num_heads

    def embed(self, x):
        return self.visual(x)

    def positions(self, h: int, w: int) -> torch.Tensor:
        """The position codes for an h x w grid of patches, interpolated from
        the g x g grid CLIP was trained on"""
        pos = self.visual.positional_embedding
        g = int(math.sqrt(len(pos) - 1))
        if (h, w) == (g, g):
            return pos
        grid = pos[1:].reshape(1, g, g, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)
        return torch.cat([pos[:1], grid.permute(0, 2, 3, 1).reshape(h * w, -1)])

    def before_last(self, x: torch.Tensor) -> torch.Tensor:
        """Tokens [B, 1 + h * w, width] going into the last layer"""
        v = self.visual
        t = v.conv1(x).flatten(2).transpose(1, 2)
        cls = v.class_embedding.view(1, 1, -1).expand(len(t), -1, -1).to(t.dtype)
        t = torch.cat([cls, t], dim=1)
        t = v.ln_pre(t + self.positions(x.shape[2] // self.patch, x.shape[3] // self.patch).to(t.dtype))
        for block in v.transformer.resblocks[:-1]:
            t = block(t)
        return t

    def qkv(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Queries, keys, values of the last layer, [B, heads, tokens, d] each"""
        attn = self.last.attn
        B, N, C = t.shape
        qkv = F.linear(self.last.ln_1(t), attn.in_proj_weight, attn.in_proj_bias)
        return tuple(z.reshape(B, N, attn.num_heads, C // attn.num_heads).transpose(1, 2)
                     for z in qkv.chunk(3, dim=-1))

    def dense(self, x):
        """ClearCLIP: see the top of this file"""
        h, w = x.shape[2] // self.patch, x.shape[3] // self.patch
        t = self.before_last(x)
        q, _, v = self.qkv(t)
        out = F.scaled_dot_product_attention(q, q, v)     # softmax(q q^T / sqrt(d)) v
        out = self.last.attn.out_proj(out.transpose(1, 2).reshape(t.shape))
        out = self.visual.ln_post(out)
        if self.visual.proj is not None:
            out = out @ self.visual.proj
        return out[:, 1:].reshape(len(x), h, w, -1)

    def attention(self, x):
        q, k, _ = self.qkv(self.before_last(x))
        return cls_attention(q, k, q.shape[-1] ** -0.5, 1, x.shape[2] // self.patch, x.shape[3] // self.patch)


class TorchvisionBackbone(Backbone):
    """ImageNet models of torchvision, their last layer replaced by Identity
    (classifier_model.load_imagenet_backbone)"""

    def __init__(self, model: nn.Module, arch: ImageNetArch):
        self.model = model
        self.arch = arch
        if arch in DENSE_IMAGENET:
            self.patch = 32         # both halve the image 5 times

    def embed(self, x):
        return self.model(x)

    def dense(self, x):
        m = self.model
        if self.arch in (ImageNetArch.RESNET50, ImageNetArch.RESNET101):
            f = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            f = m.layer4(m.layer3(m.layer2(m.layer1(f))))
        elif self.arch in (ImageNetArch.CONVNEXT_TINY, ImageNetArch.CONVNEXT_BASE):
            f = m.classifier[0](m.features(x))      # its LayerNorm, at every position
        else:
            return super().dense(x)
        return f.permute(0, 2, 3, 1)


class AutoencoderBackbone(Backbone):
    """The encoder: images 0..1, no normalization"""
    mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    def __init__(self, model: Autoencoder):
        self.model = model
        self.size = model.image_size

    def embed(self, x):
        return self.model.encode(x)

    def reconstruct(self, x):
        return self.model.decode(self.model.encode(x))


def classifier_backbone(classifier: Classifier) -> Backbone:
    b, family = classifier.backbone, classifier.classifier_id.backbone
    if family == ClassifierBackbone.DINOV2:
        backbone = DinoBackbone(b)
    elif family == ClassifierBackbone.CLIP:
        backbone = ClipBackbone(b.visual if isinstance(b, CLIPImageEncoder) else b)
    else:
        backbone = TorchvisionBackbone(b, classifier.classifier_id.arch)
    backbone.mean, backbone.std = classifier.image_normalization()
    backbone.size = classifier.image_size()[0]
    return backbone


# ---------------------------------------------------------------------------
# Loaded models
# ---------------------------------------------------------------------------

class Model:
    """A loaded model.

    family       see the top of this file
    backbone     a Backbone (classifier, autoencoder, dino, clip), else None
    class_names  classifier, segformer: one name per output
    """
    backbone: Backbone | None = None
    class_names: list[str] | None = None
    description = ""

    def __init__(self, family: str, device: torch.device):
        self.family = family
        self.device = device

    def tensor(self, images: list[Image.Image], size: tuple[int, int] | None = None) -> torch.Tensor:
        b = self.backbone
        return to_tensor(images, size or (b.size, b.size), b.mean, b.std, self.device)

    def image_vectors(self, images: list[Image.Image]) -> torch.Tensor:
        """embed() of whole images (squashed to the square input), [B, D]"""
        with torch.inference_mode():
            return self.backbone.embed(self.tensor(images)).float()


class ClassifierModel(Model):

    def __init__(self, folder: Path, device):
        super().__init__("classifier", device)
        self.classifier = Classifier.from_folder(folder).to(device).eval()
        self.class_names = self.classifier.class_names
        self.backbone = classifier_backbone(self.classifier)
        self.head = self.classifier.head
        self.description = str(self.classifier)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, S, S] -> [B, classes], differentiable"""
        return self.head(self.backbone.embed(x))


class AutoencoderModel(Model):

    def __init__(self, folder: Path, device):
        super().__init__("autoencoder", device)
        self.autoencoder = Autoencoder.from_folder(folder).to(device).eval()
        self.backbone = AutoencoderBackbone(self.autoencoder)
        self.description = str(self.autoencoder)


class DinoModel(Model):

    def __init__(self, model_id: str, device):
        super().__init__("dino", device)
        arch = PRETRAINED[model_id][1].arch
        self.backbone = DinoBackbone(torch.hub.load("facebookresearch/dinov2", arch.value).to(device).eval())
        self.description = f"DINOv2 {arch}"


class ClipModel(Model):

    def __init__(self, model_id: str, device):
        import open_clip
        super().__init__("clip", device)
        clip_id = PRETRAINED[model_id][1]
        self.clip = open_clip.create_model(clip_id.arch.value, pretrained=clip_id.pretrained.value).to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(clip_id.arch.value)
        self.backbone = ClipBackbone(self.clip.visual)
        self.description = f"CLIP {clip_id.arch} {clip_id.pretrained}"

    def text_vectors(self, texts: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            return self.clip.encode_text(self.tokenizer(texts).to(self.device)).float()


class SegformerModel(Model):

    def __init__(self, model_id: str, device):
        from transformers import SegformerForSemanticSegmentation
        super().__init__("segformer", device)
        self.name = SEGFORMERS[model_id][1]
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.name).to(device).eval()
        labels = self.model.config.id2label
        self.class_names = [labels[i] for i in range(len(labels))]
        self.description = f"{self.name}, {len(self.class_names)} classes"

    @torch.inference_mode()
    def probabilities(self, image: Image.Image, long_side: int) -> torch.Tensor:
        """-> [classes, h, w] softmax over the classes, h x w = a quarter of
        the input, which is the image at about long_side pixels"""
        x = to_tensor([image], dense_size(*image.size, long_side, 32), IMAGENET_MEAN, IMAGENET_STD, self.device)
        return self.model(pixel_values=x).logits[0].float().softmax(0)


class ClipsegModel(Model):
    SIZE = 352      # CLIPSeg's input, square

    def __init__(self, model_id: str, device):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        super().__init__("clipseg", device)
        self.name = CLIPSEGS[model_id][1]
        self.processor = CLIPSegProcessor.from_pretrained(self.name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(self.name).to(device).eval()
        self.description = self.name

    def pixels(self, images: list[Image.Image]) -> torch.Tensor:
        return self.processor(images=[i.convert("RGB") for i in images],
                              return_tensors="pt")["pixel_values"].to(self.device)

    @torch.inference_mode()
    def masks(self, image: Image.Image, texts: list[str], references: list[Image.Image]) -> torch.Tensor:
        """-> [prompts, 352, 352] the logits of "this pixel is what the prompt
        describes", texts first, then reference images"""
        x = self.pixels([image])
        out = []
        if texts:
            tokens = self.processor.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
            out.append(self.model(pixel_values=x.expand(len(texts), -1, -1, -1), **tokens).logits)
        if references:
            out.append(self.model(pixel_values=x.expand(len(references), -1, -1, -1),
                                  conditional_pixel_values=self.pixels(references)).logits)
        logits = torch.cat([o.reshape(-1, self.SIZE, self.SIZE) for o in out])
        return logits.float()


# ---------------------------------------------------------------------------
# What there is
# ---------------------------------------------------------------------------

def family_of(entry: dict) -> str:
    if entry["kind"] == "pretrained":
        return "dino" if "dinov2" in entry["id"] else "clip"
    return entry["kind"]


def has_dense(entry: dict) -> bool:
    """Does the model give one vector per patch (without loading it)?"""
    family = entry["family"]
    if family in ("dino", "clip"):
        return True
    if family != "classifier":
        return False
    config = read_config(entry["path"])
    backbone = ClassifierBackbone(config["backbone"])
    return backbone != ClassifierBackbone.IMAGE_NET or ImageNetArch(config["arch"]) in DENSE_IMAGENET


def find_models(root: str) -> list[dict]:
    """The models of embedders.find_models, then the segmentation models
    -> [{"id", "label", "kind", "family", "dense", "path"}]"""
    found = []
    for entry in find_embedders(root):
        entry["family"] = family_of(entry)
        try:
            entry["dense"] = has_dense(entry)
        except (OSError, ValueError, KeyError):
            entry["dense"] = False
        found.append(entry)
    for kind, table in (("segformer", SEGFORMERS), ("clipseg", CLIPSEGS)):
        found += [{"id": model_id, "label": label, "kind": kind, "family": kind, "dense": True, "path": None}
                  for model_id, (label, _) in table.items()]
    return found


def load_model(entry: dict, device: torch.device) -> Model:
    family = entry["family"]
    if family == "classifier":
        return ClassifierModel(Path(entry["path"]), device)
    if family == "autoencoder":
        return AutoencoderModel(Path(entry["path"]), device)
    if family == "dino":
        return DinoModel(entry["id"], device)
    if family == "clip":
        return ClipModel(entry["id"], device)
    if family == "segformer":
        return SegformerModel(entry["id"], device)
    if family == "clipseg":
        return ClipsegModel(entry["id"], device)
    raise ValueError(f"unknown model family {family}")


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()
