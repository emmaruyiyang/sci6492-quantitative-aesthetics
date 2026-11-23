"""(A copy of 06_autoencoders/embedders.py, so that 07_segmentation works on its own.)

Anything that turns images into vectors, behind one interface.

  autoencoder   a saved Autoencoder: its latent vector. Can also reconstruct.
  classifier    a saved Classifier (classifier_model.py): its logits, one score
                per class (softmax of them = the class probabilities).
  pretrained    a DINOv2 or CLIP image encoder, no folder needed: the vector
                it was trained to output.

find_models() looks for saved models (folders with a valid config.json) under
a root folder and adds the pretrained ones. load_embedder() builds one.
"""

import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from autoencoder_model import Autoencoder, folder_model_type
from classifier_model import (CLIPArch, CLIPPretrained, Classifier, ClassifierID, DINOV2Arch,
                              load_backbone)

# Pretrained encoders always in the list: id -> (label, ClassifierID)
PRETRAINED = {
    "pretrained/dinov2_vits14": ("DINOv2 ViT-S/14", ClassifierID.dinov2(DINOV2Arch.VITS14)),
    "pretrained/dinov2_vitb14": ("DINOv2 ViT-B/14", ClassifierID.dinov2(DINOV2Arch.VITB14)),
    "pretrained/dinov2_vitl14": ("DINOv2 ViT-L/14", ClassifierID.dinov2(DINOV2Arch.VITL14)),
    "pretrained/clip_b32_openai": ("CLIP ViT-B/32 openai",
                                   ClassifierID.clip(CLIPArch.VITB32, CLIPPretrained.OPENAI)),
    "pretrained/clip_b32_datacomp": ("CLIP ViT-B/32 datacomp",
                                     ClassifierID.clip(CLIPArch.VITB32, CLIPPretrained.DATACOMP_XL_S13B_B90K)),
    "pretrained/clip_l14_openai": ("CLIP ViT-L/14 openai",
                                   ClassifierID.clip(CLIPArch.VITL14, CLIPPretrained.OPENAI)),
    "pretrained/clip_l14_datacomp": ("CLIP ViT-L/14 datacomp",
                                     ClassifierID.clip(CLIPArch.VITL14, CLIPPretrained.DATACOMP_XL_S13B_B90K)),
}


class Embedder:
    kind = None
    can_reconstruct = False

    def embed(self, images: list[Image.Image]) -> torch.Tensor:
        """PIL images -> vectors [B, D] on the cpu"""
        raise NotImplementedError

    def reconstruct(self, images: list[Image.Image]) -> list[Image.Image]:
        raise NotImplementedError(f"a {self.kind} can't reconstruct images")


class AutoencoderEmbedder(Embedder):
    kind = "autoencoder"
    can_reconstruct = True

    def __init__(self, folder: Path, device: torch.device):
        self.model = Autoencoder.from_folder(folder).to(device).eval()
        self.description = str(self.model)

    def embed(self, images):
        return self.model.embed(images)

    def reconstruct(self, images):
        return self.model.reconstruct(images)[0]


class BackboneEmbedder(Embedder):
    """A backbone (images -> features) with the preprocessing it expects."""

    def __init__(self, backbone: torch.nn.Module, classifier_id: ClassifierID, device: torch.device):
        self.backbone = backbone.to(device).eval()
        self.device = device
        mean, std = classifier_id.image_normalization()
        size = classifier_id.image_size()
        self.preprocess = transforms.Compose([
            transforms.Resize(size[0]),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    @torch.inference_mode()
    def embed(self, images):
        x = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device)
        return self.backbone(x).float().cpu()


class ClassifierEmbedder(Embedder):
    """The vector is the classifier's logits: one raw score per class (softmax
    of them = the class probabilities)."""
    kind = "classifier"

    def __init__(self, folder: Path, device: torch.device):
        self.model = Classifier.from_folder(folder).to(device).eval()
        self.class_names = self.model.class_names
        self.description = str(self.model)

    def embed(self, images):
        return self.model.predict(images, "logits")


class PretrainedEmbedder(BackboneEmbedder):
    kind = "pretrained"

    def __init__(self, classifier_id: ClassifierID, device: torch.device):
        backbone, size = load_backbone(classifier_id)
        super().__init__(backbone, classifier_id, device)
        self.description = f"pretrained {classifier_id} -> {size} features"


def find_models(root: str) -> list[dict]:
    """Every folder under root whose config.json is an autoencoder or a
    classifier, then the pretrained encoders.
    -> [{"id", "label", "kind", "path"}]"""
    found = []
    root = Path(root)
    if root.is_dir():
        for folder, subfolders, files in os.walk(root):
            subfolders[:] = sorted(s for s in subfolders if not s.startswith((".", "__")))
            if "config.json" not in files:
                continue
            kind = folder_model_type(folder)
            if kind is None:
                continue
            relative = Path(folder).relative_to(root).as_posix()
            found.append({"id": f"folder/{relative}", "label": relative, "kind": kind, "path": folder})
            subfolders[:] = []      # a model's subfolders (snapshots...) are not models
    found += [{"id": model_id, "label": label, "kind": "pretrained", "path": None}
              for model_id, (label, _) in PRETRAINED.items()]
    return found


def load_embedder(entry: dict, device: torch.device) -> Embedder:
    if entry["kind"] == "autoencoder":
        return AutoencoderEmbedder(Path(entry["path"]), device)
    if entry["kind"] == "classifier":
        return ClassifierEmbedder(Path(entry["path"]), device)
    if entry["kind"] == "pretrained":
        return PretrainedEmbedder(PRETRAINED[entry["id"]][1], device)
    raise ValueError(f"unknown model kind {entry['kind']}")
