"""Image classifier = pretrained backbone + small trainable head.

(From QA2026 05_b_classifiers/classifier_model.py; models saved by either one
can be loaded by the other. The architecture enums come from
model_types and the backbones are loaded by
embedding_models, the same ones ModelCentral uses.)

    image  ->  backbone  ->  feature vector  ->  head  ->  one score per class
               (pretrained)   e.g. 2048 numbers   (trained)  (logits)

The backbone is a network that somebody else trained on millions of images. We
throw away whatever task it was trained for and keep only the part that turns an
image into a feature vector (an "embedding"). The head is a small network that
we train on our own classes, on top of those features.

Three families of backbones:

  imagenet  torchvision models trained to recognise the 1000 ImageNet classes.
            Their last layer maps features -> 1000 ImageNet scores. We replace
            that layer with nn.Identity() so the model outputs the features
            that were feeding into it (2048 for ResNet-50, 768 for ViT-B, ...).
  dinov2    self-supervised ViTs from Meta (loaded with torch.hub). They were
            never trained on labels, they output an embedding directly.
  clip      image tower of a CLIP model (loaded with open_clip). Trained to
            match images with their captions, outputs an embedding directly.

Freezing the backbone means only the head learns. That is fast, needs little
data, and the backbone's features stay as they were. Unfreezing it fine-tunes
the whole network: slower, needs more memory and data, but can adapt the
features themselves to our images.

Saving writes a folder with

  config.json   everything needed to rebuild the classifier (backbone, classes,
                head settings, image size, normalization)
  head.pt       weights of the head
  backbone.pt   weights of the backbone, only when it was fine-tuned. A frozen
                backbone is identical to the pretrained one, so it is downloaded
                again instead of stored.
"""

import json
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from embedding_models import INPUT_SIZE, default_imagenet_weights, image_normalization, load_feature_extractor
from model_types import CLIPArch, CLIPPretrained, DINOV2Arch, ImageNetArch, ModelID


class ClassifierBackbone(Enum):
    CLIP = "clip"
    DINOV2 = "dinov2"
    IMAGE_NET = "imagenet"

    def __str__(self) -> str:
        return self.value


ARCHS = {
    ClassifierBackbone.CLIP: CLIPArch,
    ClassifierBackbone.DINOV2: DINOV2Arch,
    ClassifierBackbone.IMAGE_NET: ImageNetArch,
}


class ClassifierID:
    """Names a backbone: which family, which architecture, which weights."""

    @staticmethod
    def clip(arch: CLIPArch = CLIPArch.VITB32, pretrained: CLIPPretrained = CLIPPretrained.DATACOMP_XL_S13B_B90K) -> "ClassifierID":
        return ClassifierID(ClassifierBackbone.CLIP, arch, pretrained)

    @staticmethod
    def dinov2(arch: DINOV2Arch = DINOV2Arch.VITB14) -> "ClassifierID":
        return ClassifierID(ClassifierBackbone.DINOV2, arch)

    @staticmethod
    def imagenet(arch: ImageNetArch = ImageNetArch.RESNET50) -> "ClassifierID":
        return ClassifierID(ClassifierBackbone.IMAGE_NET, arch)

    @staticmethod
    def from_dict(d: dict) -> "ClassifierID":
        backbone = ClassifierBackbone(d["backbone"])
        arch = ARCHS[backbone](d["arch"])
        pretrained = CLIPPretrained(d["pretrained"]) if d.get("pretrained") else None
        return ClassifierID(backbone, arch, pretrained)

    def __init__(self, backbone: ClassifierBackbone, arch: Enum, pretrained: Enum = None):
        self.backbone = backbone
        self.arch = arch
        self.pretrained = pretrained

    def __str__(self) -> str:
        if self.pretrained is None:
            return f"[{self.backbone},{self.arch}]"
        return f"[{self.backbone},{self.arch},{self.pretrained}]"

    def __eq__(self, other) -> bool:
        return isinstance(other, ClassifierID) and self.to_dict() == other.to_dict()

    def to_dict(self) -> dict:
        return {
            "backbone": self.backbone.value,
            "arch": self.arch.value,
            "pretrained": self.pretrained.value if self.pretrained else None,
        }

    def model_id(self) -> ModelID:
        """The same backbone named as in shared.models (ModelCentral, embedding_models)"""
        if self.backbone == ClassifierBackbone.CLIP:
            return ModelID.clip(self.arch, self.pretrained)
        if self.backbone == ClassifierBackbone.DINOV2:
            return ModelID.dino(self.arch)
        return ModelID.imagenet(self.arch, default_imagenet_weights(self.arch))

    def image_normalization(self) -> tuple:
        return image_normalization(self.model_id())

    def image_size(self) -> tuple:
        return INPUT_SIZE, INPUT_SIZE


# ---------------------------------------------------------------------------
# Backbones: embedding_models loads them
# ---------------------------------------------------------------------------

def load_backbone(classifier_id: ClassifierID) -> tuple[nn.Module, int]:
    """-> (module that maps images -> features, feature size)"""
    return load_feature_extractor(classifier_id.model_id())


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

def make_head(feature_size: int, num_classes: int, head: str, hidden_size: int, dropout: float) -> nn.Module:
    """The trainable part.

    Both start with a LayerNorm: each backbone's features come at a different
    scale (CLIP's are small, ResNet's are large), the LayerNorm rescales them so
    the same learning rate works for all of them.

      linear  features -> class scores, one weighted sum per class
              (also called a "linear probe")
      mlp     features -> hidden layer -> class scores. Can learn non-linear
              combinations of features, but overfits more easily on few images.
    """
    if head == "linear":
        return nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, num_classes),
        )
    if head == "mlp":
        return nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
    raise ValueError(f"head must be 'linear' or 'mlp', not '{head}'")


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class Classifier(nn.Module):

    @staticmethod
    def clip(class_names: list[str], arch: CLIPArch = CLIPArch.VITB32, pretrained: CLIPPretrained = CLIPPretrained.DATACOMP_XL_S13B_B90K, **options) -> "Classifier":
        return Classifier(ClassifierID.clip(arch, pretrained), class_names, **options)

    @staticmethod
    def dinov2(class_names: list[str], arch: DINOV2Arch = DINOV2Arch.VITB14, **options) -> "Classifier":
        return Classifier(ClassifierID.dinov2(arch), class_names, **options)

    @staticmethod
    def imagenet(class_names: list[str], arch: ImageNetArch = ImageNetArch.RESNET50, **options) -> "Classifier":
        return Classifier(ClassifierID.imagenet(arch), class_names, **options)

    @staticmethod
    def from_folder(folder: str) -> "Classifier":
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text())
        classifier = Classifier(
            ClassifierID.from_dict(config),
            config["class_names"],
            head=config["head"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            freeze_backbone=config["freeze_backbone"],
        )
        classifier.head.load_state_dict(torch.load(folder / "head.pt", map_location="cpu"))
        if (folder / "backbone.pt").exists():
            classifier.backbone.load_state_dict(torch.load(folder / "backbone.pt", map_location="cpu"))
        return classifier

    def __init__(self, classifier_id: ClassifierID, class_names: list[str],
                 head: str = "linear", hidden_size: int = 512, dropout: float = 0.2,
                 freeze_backbone: bool = True):
        super().__init__()
        self.classifier_id = classifier_id
        self.class_names = list(class_names)
        self.head_type = head
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.backbone, self.feature_size = load_backbone(classifier_id)
        self.head = make_head(self.feature_size, len(self.class_names), head, hidden_size, dropout)
        self.set_freeze_backbone(freeze_backbone)

    def __str__(self) -> str:
        frozen = "frozen" if self.freeze_backbone else "fine-tuned"
        return (f"Classifier({self.classifier_id}, {frozen} backbone -> {self.feature_size} features, "
                f"{self.head_type} head -> {len(self.class_names)} classes)")

    def set_freeze_backbone(self, freeze: bool) -> None:
        self.freeze_backbone = freeze
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def train(self, mode: bool = True) -> "Classifier":
        # A frozen backbone stays in eval mode even while the head trains.
        # Otherwise layers like BatchNorm (in ResNet) and Dropout behave as in
        # training and the "frozen" features would still change.
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    # --- images -> tensors --------------------------------------------------

    def image_normalization(self) -> tuple:
        return self.classifier_id.image_normalization()

    def image_size(self) -> tuple:
        return self.classifier_id.image_size()

    def preprocess(self, random_transformations: list = None) -> transforms.Compose:
        """PIL image -> normalized tensor of the size the backbone expects.

        Random transformations (augmentations, used during training) are
        applied first. Then the image is resized so its short side fits, the
        center square is cropped, and the colors are normalized."""
        mean, std = self.image_normalization()
        size = self.image_size()
        return transforms.Compose(list(random_transformations or []) + [
            transforms.Resize(size[0]),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # --- forward --------------------------------------------------------------

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Batch of images [B, 3, H, W] -> features [B, feature_size]."""
        if self.freeze_backbone:
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batch of images [B, 3, H, W] -> logits [B, num_classes]."""
        return self.head(self.features(x))

    @torch.inference_mode()
    def predict(self, images: list[Image.Image], activation: str = "softmax") -> torch.Tensor:
        """List of PIL images -> scores [B, num_classes] between 0 and 1.

          softmax  scores of an image add up to 1: the classes compete, it
                   always picks something, even for an image of none of them.
          sigmoid  each class gets its own independent 0..1 score.
          logits   the raw scores of the head, before either of the above.
        """
        self.eval()
        device = next(self.parameters()).device
        preprocess = self.preprocess()
        x = torch.stack([preprocess(img.convert("RGB")) for img in images]).to(device)
        logits = self(x)
        if activation == "softmax":
            return logits.softmax(dim=-1).cpu()
        if activation == "sigmoid":
            return logits.sigmoid().cpu()
        if activation == "logits":
            return logits.cpu()
        raise ValueError(f"activation must be 'softmax', 'sigmoid' or 'logits', not '{activation}'")

    # --- saving ---------------------------------------------------------------

    def config(self) -> dict:
        mean, std = self.image_normalization()
        return {
            # model_type lets other programs (e.g. 06_autoencoders) tell what
            # kind of model a folder holds before loading it
            "model_type": "classifier",
            "version": "1.0",
            **self.classifier_id.to_dict(),
            "class_names": self.class_names,
            "head": self.head_type,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "freeze_backbone": self.freeze_backbone,
            "feature_size": self.feature_size,
            "image_size": list(self.image_size()),
            "image_mean": list(mean),
            "image_std": list(std),
        }

    def save(self, folder: str) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps(self.config(), indent=2))
        torch.save(self.head.state_dict(), folder / "head.pt")
        # A frozen backbone hasn't changed, so there is nothing new to store.
        # (If it was fine-tuned earlier, backbone.pt from back then stays.)
        if not self.freeze_backbone:
            torch.save(self.backbone.state_dict(), folder / "backbone.pt")
