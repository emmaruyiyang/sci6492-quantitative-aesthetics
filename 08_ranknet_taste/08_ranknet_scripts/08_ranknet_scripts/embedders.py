"""Anything that turns images into vectors, behind one interface.

  autoencoder   a saved Autoencoder: its latent vector. Can also reconstruct.
  classifier    a saved Classifier (classifier_model.py): its logits, one score
                per class (softmax of them = the class probabilities).
  ranknet       a saved RankNet (ranknet_model.py): a vector of one number, the
                image's score.
  aesthetic     a ready-made aesthetic / preference score (aesthetic/
                aesthetic_scorers.py): one number. Some score against a prompt.
  pretrained    a DINOv2 or CLIP image encoder (embedding_models),
                no folder needed: the vector it was trained to output. CLIP
                also embeds text (a prompt) into the same space.

find_models() looks for saved models (folders with a valid config.json, and
RankNet card files, *.ranknet.json) under a root folder and adds the
pretrained ones and the aesthetic scores. load_embedder() builds one.
"""

import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from aesthetic.aesthetic_scorers import SCORERS, load_scorer
from autoencoder.autoencoder_model import Autoencoder, folder_model_type
from ranknet import model_card
from classifier.classifier_model import Classifier
from ranknet.ranknet_model import RankNet
from embedding_models import (INPUT_SIZE, CLIPImageEncoder, image_normalization, load_clip,
                                           load_feature_extractor)
from model_types import CLIPArch, CLIPPretrained, DINOV2Arch, ModelID

# Pretrained encoders always in the list: id -> (label, ModelID)
PRETRAINED = {
    "pretrained/dinov2_vits14": ("DINOv2 ViT-S/14", ModelID.dino(DINOV2Arch.VITS14)),
    "pretrained/dinov2_vitb14": ("DINOv2 ViT-B/14", ModelID.dino(DINOV2Arch.VITB14)),
    "pretrained/dinov2_vitl14": ("DINOv2 ViT-L/14", ModelID.dino(DINOV2Arch.VITL14)),
    "pretrained/clip_b32_openai": ("CLIP ViT-B/32 openai", ModelID.clip(CLIPArch.VITB32, CLIPPretrained.OPENAI)),
    "pretrained/clip_b32_datacomp": ("CLIP ViT-B/32 datacomp",
                                     ModelID.clip(CLIPArch.VITB32, CLIPPretrained.DATACOMP_XL_S13B_B90K)),
    "pretrained/clip_l14_openai": ("CLIP ViT-L/14 openai", ModelID.clip(CLIPArch.VITL14, CLIPPretrained.OPENAI)),
    "pretrained/clip_l14_datacomp": ("CLIP ViT-L/14 datacomp",
                                     ModelID.clip(CLIPArch.VITL14, CLIPPretrained.DATACOMP_XL_S13B_B90K)),
}


class Embedder:
    kind = None
    can_reconstruct = False
    can_embed_text = False

    def embed(self, images: list[Image.Image]) -> torch.Tensor:
        """PIL images -> vectors [B, D] on the cpu"""
        raise NotImplementedError

    def reconstruct(self, images: list[Image.Image]) -> list[Image.Image]:
        raise NotImplementedError(f"a {self.kind} can't reconstruct images")

    def embed_text(self, texts: list[str]) -> torch.Tensor:
        """texts -> vectors [B, D] on the cpu, comparable with the images' (CLIP)"""
        raise NotImplementedError(f"a {self.kind} can't embed text")


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


class RankNetEmbedder(Embedder):
    """The vector is the RankNet's score: one number (only differences of
    scores mean something, the zero is arbitrary)."""
    kind = "ranknet"

    def __init__(self, folder: Path, device: torch.device):
        self.model = RankNet.from_path(folder).to(device).eval()      # a folder or a card file
        self.description = str(self.model)

    def embed(self, images):
        return self.model.score(images).unsqueeze(1)


class AestheticEmbedder(Embedder):
    """The vector is a ready-made aesthetic score: one number"""
    kind = "aesthetic"

    def __init__(self, scorer_id: str, device: torch.device, prompt: str = None):
        self.scorer = load_scorer(scorer_id, device)
        if prompt is not None:
            self.set_prompt(prompt)

    @property
    def description(self) -> str:
        return str(self.scorer)

    def set_prompt(self, prompt: str) -> None:
        """only the text is encoded again, the model stays loaded"""
        self.scorer.set_prompt(prompt)

    def embed(self, images):
        return self.scorer.score(images).unsqueeze(1)


class PretrainedEmbedder(Embedder):
    """A pretrained encoder (embedding_models) with the
    preprocessing it expects. CLIP keeps its text tower too, on the cpu (a
    prompt is encoded once, the images on the gpu)."""
    kind = "pretrained"

    def __init__(self, model_id: ModelID, device: torch.device):
        self.text_model = self.tokenizer = None
        if model_id.is_clip():
            import open_clip
            model = load_clip(model_id.arch, model_id.pretrained)
            backbone, size = CLIPImageEncoder(model.visual), model.visual.output_dim
            model.visual = None
            self.text_model = model.eval()
            self.tokenizer = open_clip.get_tokenizer(model_id.arch)
            self.can_embed_text = True
        else:
            backbone, size = load_feature_extractor(model_id)
        self.backbone = backbone.to(device).eval()
        self.device = device
        mean, std = image_normalization(model_id)
        self.preprocess = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.description = f"pretrained {model_id} -> {size} features"

    @torch.inference_mode()
    def embed(self, images):
        x = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device)
        return self.backbone(x).float().cpu()

    @torch.inference_mode()
    def embed_text(self, texts):
        if self.text_model is None:
            return super().embed_text(texts)
        return self.text_model.encode_text(self.tokenizer(texts)).float()


def find_models(root: str) -> list[dict]:
    """Every folder under root whose config.json is an autoencoder, a
    classifier or a RankNet, then the pretrained encoders.
    -> [{"id", "label", "kind", "path"}]"""
    found = []
    root = Path(root)
    if root.is_dir():
        for folder, subfolders, files in os.walk(root):
            subfolders[:] = sorted(s for s in subfolders if not s.startswith((".", "__")))
            if "config.json" not in files:
                # RankNet cards on their own (a model folder's own card is the folder)
                for f in sorted(files):
                    if f.endswith(model_card.SUFFIX):
                        relative = (Path(folder) / f).relative_to(root).as_posix()
                        found.append({"id": f"card/{relative}", "label": relative, "kind": "ranknet",
                                      "path": str(Path(folder) / f)})
                continue
            kind = folder_model_type(folder)
            if kind is None:
                continue
            relative = Path(folder).relative_to(root).as_posix()
            found.append({"id": f"folder/{relative}", "label": relative, "kind": kind, "path": folder})
            subfolders[:] = []      # a model's subfolders (snapshots...) are not models
    found += [{"id": f"aesthetic/{scorer_id}", "label": label, "kind": "aesthetic", "path": None,
               "scorer": scorer_id, "prompt": uses_prompt}
              for scorer_id, (label, uses_prompt, _) in SCORERS.items()]
    found += [{"id": model_id, "label": label, "kind": "pretrained", "path": None}
              for model_id, (label, _) in PRETRAINED.items()]
    return found


def query_kinds(entry: dict) -> list[str]:
    """What else than a component a model's images can be ordered by: the
    cosine similarity to a "prompt" (CLIP) or to an "image" (CLIP, DINOv2)"""
    if entry["kind"] != "pretrained":
        return []
    return ["prompt", "image"] if PRETRAINED[entry["id"]][1].is_clip() else ["image"]


def load_embedder(entry: dict, device: torch.device) -> Embedder:
    if entry["kind"] == "autoencoder":
        return AutoencoderEmbedder(Path(entry["path"]), device)
    if entry["kind"] == "classifier":
        return ClassifierEmbedder(Path(entry["path"]), device)
    if entry["kind"] == "ranknet":
        return RankNetEmbedder(Path(entry["path"]), device)
    if entry["kind"] == "aesthetic":
        return AestheticEmbedder(entry["scorer"], device, entry.get("prompt_text"))
    if entry["kind"] == "pretrained":
        return PretrainedEmbedder(PRETRAINED[entry["id"]][1], device)
    raise ValueError(f"unknown model kind {entry['kind']}")
