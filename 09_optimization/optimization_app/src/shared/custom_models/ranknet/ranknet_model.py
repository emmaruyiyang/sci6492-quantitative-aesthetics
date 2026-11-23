"""RankNet: one score per image ("how good is it"), learned from pairwise
preferences ("of these two, I prefer the left one").

(From DiffDrawing's aesthetic_scoring: rank_net_nn.py, rank_net_train.py,
taste_train.py. Folders saved by its RankNetNN load here too.)

    image  ->  frozen backbone(s)  ->  embedding  ->  score head  ->  score
               (CLIP, DINOv2...)       concatenated    (trained)       one number

The backbones are pretrained encoders (shared.models.embedding_models) that are
never trained: every image is embedded once, then only the small score head
learns. With several backbones their embeddings are concatenated.

Training never sees a score, only pairs [a, b, pref] (pref 1 = a better, 0 =
equal, -1 = b better). The RankNet loss turns the difference of the two scores
into the probability that a is better, and compares it with the answer:

    P(a better than b) = sigmoid(score_a - score_b)     target 1 / 0.5 / 0

So only differences of scores mean anything: the zero point is arbitrary, and
an "equal" answer pulls the two scores together.

Saving writes a folder with

  config.json   model_type ("ranknet"), version, the backbones (model_ids),
                the head size, how each embedding is normalized, what it
                was trained on (dataset, ranking file)
  weights.pt    weights of the score head
  <folder name>.ranknet.json
                the whole model as one file (model_card.py): config, how it
                was trained, weights. RankNet.from_path() loads a folder or a
                card file.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from shared.custom_models.ranknet import model_card
from shared.models.embedding_models import INPUT_SIZE, image_normalization, load_feature_extractor
from shared.models.model_types import ModelID

MODEL_TYPE = "ranknet"
VERSION = "1.0"
SUPPORTED_VERSIONS = {"1.0"}


class ScoreHead(nn.Module):
    """The trainable part: embedding -> one score. The same layers as
    DiffDrawing's ScoreHead, so their weights load here."""

    def __init__(self, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def ranknet_loss(score_a: torch.Tensor, score_b: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
    """pref 1 -> a better, 0 -> equal, -1 -> b better: target probability 1, 0.5, 0"""
    return F.binary_cross_entropy_with_logits(score_a - score_b, (pref + 1.0) / 2.0)


@torch.no_grad()
def pair_accuracy(score_a: torch.Tensor, score_b: torch.Tensor, pref: torch.Tensor) -> float | None:
    """How often the scores order a pair the way it was answered. "Equal"
    answers are left out (no order is right for them). None if there are none."""
    strict = pref != 0
    if not bool(strict.any()):
        return None
    return float(((score_a[strict] > score_b[strict]) == (pref[strict] > 0)).float().mean())


class RankNet(nn.Module):

    def __init__(self, model_ids: list[ModelID], hidden_dim: int = 128, dropout: float = 0.3,
                 normalize: list[bool] = None):
        """normalize: per backbone, scale its embedding to length 1 (default:
        all of them, as ModelCentral compares them)."""
        super().__init__()
        self.model_ids = list(model_ids)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.normalize = list(normalize) if normalize is not None else [True] * len(self.model_ids)
        self.trained_on = {}        # saved in config.json: the dataset and ranking file of the training
        self.training = {}          # saved in the card: a summary of the training (pairs, agreement...)
        backbones, sizes = zip(*(load_feature_extractor(m) for m in self.model_ids))
        self.backbones = nn.ModuleList(backbones).requires_grad_(False)
        self.embed_dim = sum(sizes)
        self.head = ScoreHead(self.embed_dim, hidden_dim, dropout)
        self.preprocesses = [transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(*image_normalization(m)),
        ]) for m in self.model_ids]

    def __str__(self) -> str:
        backbones = " + ".join(str(m) for m in self.model_ids)
        return f"RankNet({backbones} -> {self.embed_dim} features, head {self.hidden_dim}-64-1)"

    @property
    def device(self) -> torch.device:
        return next(self.head.parameters()).device

    def train(self, mode: bool = True) -> "RankNet":
        super().train(mode)
        self.backbones.eval()       # frozen: always in eval mode (no dropout, fixed norms)
        return self

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> torch.Tensor:
        """PIL images -> concatenated embeddings [B, embed_dim], on the model's device"""
        parts = []
        for backbone, preprocess, normalize in zip(self.backbones, self.preprocesses, self.normalize):
            x = torch.stack([preprocess(img.convert("RGB")) for img in images]).to(self.device)
            e = backbone(x).float()
            parts.append(F.normalize(e, dim=-1) if normalize else e)
        return torch.cat(parts, dim=-1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings [B, embed_dim] -> scores [B]"""
        return self.head(embeddings)

    @torch.no_grad()
    def score(self, images: list[Image.Image]) -> torch.Tensor:
        """PIL images -> scores [B] on the cpu"""
        return self.head(self.embed(images)).cpu()

    def config(self) -> dict:
        return {
            "model_type": MODEL_TYPE,
            "version": VERSION,
            "model_ids": [str(m) for m in self.model_ids],
            "normalize": self.normalize,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "trained_on": self.trained_on,
        }

    def save(self, folder) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), folder / "weights.pt")
        (folder / "config.json").write_text(json.dumps(self.config(), indent=2))
        model_card.write_card(model_card.card_path(folder), self.card())

    def card(self) -> dict:
        """The model as one JSON object (model_card.py)"""
        state = {name: (list(t.shape), t.detach().cpu().flatten().tolist()) for name, t in self.head.state_dict().items()}
        return model_card.make_card(self.config(), state, self.training)

    @staticmethod
    def from_card(card: dict) -> "RankNet":
        """A RankNet from a card (ours, or a DiffDrawing taste model card)"""
        model_card.check(card)
        model_ids = [ModelID.from_string(s) for s in card["model_ids"]]
        # DiffDrawing's cards: embeddings from ModelCentral, all unit length
        normalize = card.get("normalize") or [True] * len(model_ids)
        model = RankNet(model_ids, int(card["hidden_dim"]), card.get("dropout") or 0.3, normalize)
        if model.embed_dim != int(card["embed_dim"]):
            raise ValueError(f"the backbones give {model.embed_dim} features, the card's head takes {card['embed_dim']}")
        state = {}
        for name, entry in card["weights"].items():
            shape, values = model_card.decode_array(entry)
            state[name] = torch.tensor(values, dtype=torch.float32).reshape(shape)
        model.head.load_state_dict(state, strict=True)
        model.trained_on = card.get("trained_on", {})
        model.training = card.get("training", {})
        return model.eval()

    @staticmethod
    def from_path(path) -> "RankNet":
        """A RankNet folder or a card file"""
        path = Path(path)
        if path.is_file():
            return RankNet.from_card(model_card.read_card(path))
        return RankNet.from_folder(path)

    @staticmethod
    def from_folder(folder) -> "RankNet":
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text())
        model_ids = [ModelID.from_string(s) for s in config["model_ids"]]
        # DiffDrawing's RankNetNN folders: no model_type, CLIP embeddings
        # normalized, DINOv2 ones as they are
        normalize = config.get("normalize", [m.is_clip() for m in model_ids])
        model = RankNet(model_ids, config["hidden_dim"], config.get("dropout", 0.3), normalize)
        if model.embed_dim != config["embed_dim"]:
            raise ValueError(f"{folder}: the backbones give {model.embed_dim} features, "
                             f"the head was trained on {config['embed_dim']}")
        state = torch.load(folder / "weights.pt", map_location="cpu")
        # DiffDrawing saved the head as "net.*" too: same keys
        model.head.load_state_dict(state)
        model.trained_on = config.get("trained_on", {})
        return model.eval()


def is_ranknet_config(config: dict) -> bool:
    """A RankNet config.json: ours, or a DiffDrawing RankNetNN one (no model_type)"""
    if config.get("model_type") == MODEL_TYPE:
        return config.get("version") in SUPPORTED_VERSIONS
    return config.get("model_type") is None and "model_ids" in config and "embed_dim" in config
