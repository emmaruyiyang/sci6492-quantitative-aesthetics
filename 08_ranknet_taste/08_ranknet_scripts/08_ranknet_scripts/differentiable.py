"""The custom models and the aesthetic scores as differentiable functions of an
image: a batch (B, 3, H, W) with values 0..1 in, a vector per image out, the
gradient flowing back to the pixels. For optimizing an image against them
(03_optimization).

  classifier    -> its logits, one per class
  autoencoder   -> its latent vector
  ranknet       -> its score (B, 1)
  aesthetic     -> the score (B, 1); PickScore / HPSv2 against a prompt

Every model is frozen (no gradient for its weights, eval mode). An image is
resized so its short side is the model's input size and the center square
cropped, as when the model was trained (the PIL preprocessing of each model),
here with differentiable tensor operations.

For the scores, features() is the vector the score is computed from (the
RankNet's concatenated backbone embeddings, the CLIP image embedding of an
aesthetic score) and score_of(features) the score: the optimization compares
images (edit size, the start image anchor) in that feature space.

Precision: "fp32", or "bf16" for the big parts (backbones, image towers; the
small heads stay fp32): about half the memory, slightly different scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import env_config
from embedders import find_models
from embedding_models import CLIP_MEAN, CLIP_STD, INPUT_SIZE

KINDS = ("classifier", "autoencoder", "ranknet", "aesthetic")
SCORE_KINDS = ("ranknet", "aesthetic")
PRECISIONS = {"fp32": torch.float32, "bf16": torch.bfloat16}


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


class DifferentiableModel(nn.Module):
    kind = ""
    class_names = None          # classifiers
    uses_prompt = False         # PickScore, HPSv2

    def __init__(self, entry: dict, dtype: torch.dtype):
        super().__init__()
        self.entry = entry
        self.label = entry["label"]
        self.dtype = dtype          # of the big parts; inputs are cast to it, outputs back to fp32

    @property
    def is_score(self) -> bool:
        return self.kind in SCORE_KINDS

    def set_prompt(self, prompt: str) -> None:
        pass

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """images (B, 3, H, W) in 0..1 -> (B, D)"""
        raise NotImplementedError

    def prompt_embedding(self, prompt: str) -> torch.Tensor | None:
        """PickScore / HPSv2: the prompt's text embedding, for score_of"""
        return None

    def score_of(self, features: torch.Tensor, text: torch.Tensor | None = None) -> torch.Tensor:
        """scores only: features (B, D) -> scores (B,); text: a prompt_embedding (PickScore /
        HPSv2, default: set_prompt's)"""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """images -> the model's vector: logits, latent, or (B, 1) score"""
        f = self.features(x)
        return self.score_of(f).unsqueeze(1) if self.is_score else f


def _frozen(module: nn.Module) -> nn.Module:
    return module.eval().requires_grad_(False)


class ClassifierFunction(DifferentiableModel):
    kind = "classifier"

    def __init__(self, entry, device, dtype):
        super().__init__(entry, dtype)
        from classifier.classifier_model import Classifier
        self.model = _frozen(Classifier.from_folder(entry["path"]).to(device))
        self.model.backbone.to(dtype)
        self.class_names = list(self.model.class_names)
        self.size = self.model.image_size()[0]
        self.mean, self.std = self.model.image_normalization()

    def features(self, x):
        # backbone then head directly: Classifier.features() runs a frozen backbone
        # under no_grad, which would cut the gradient to the image
        x = normalize(fit(x, self.size), self.mean, self.std).to(self.dtype)
        return self.model.head(self.model.backbone(x).float()).float()


class AutoencoderFunction(DifferentiableModel):
    kind = "autoencoder"

    def __init__(self, entry, device, dtype):
        super().__init__(entry, dtype)
        from autoencoder.autoencoder_model import Autoencoder
        self.model = _frozen(Autoencoder.from_folder(entry["path"]).to(device, dtype))

    def features(self, x):
        return self.model.encode(fit(x, self.model.image_size).to(self.dtype)).flatten(1).float()


class RankNetFunction(DifferentiableModel):
    kind = "ranknet"

    def __init__(self, entry, device, dtype):
        super().__init__(entry, dtype)
        from ranknet.ranknet_model import RankNet
        from embedding_models import image_normalization
        self.model = _frozen(RankNet.from_path(entry["path"]).to(device))
        for backbone in self.model.backbones:
            backbone.to(dtype)
        self.normalizations = [image_normalization(m) for m in self.model.model_ids]

    def features(self, x):
        x = fit(x, INPUT_SIZE)
        parts = []
        for backbone, (mean, std), unit in zip(self.model.backbones, self.normalizations, self.model.normalize):
            e = backbone(normalize(x, mean, std).to(self.dtype)).float()
            parts.append(F.normalize(e, dim=-1) if unit else e)
        return torch.cat(parts, dim=-1)

    def score_of(self, features, text=None):
        return self.model.head(features)


class AestheticFunction(DifferentiableModel):
    """LAION aesthetic (CLIP ViT-L/14 + MLP), PickScore and HPSv2 (ViT-H/14 against a prompt)"""
    kind = "aesthetic"

    def __init__(self, entry, device, dtype):
        super().__init__(entry, dtype)
        from aesthetic.aesthetic_scorers import load_scorer
        self.scorer = load_scorer(entry["scorer"], device)
        self.uses_prompt = self.scorer.uses_prompt
        for name in ("clip", "mlp", "model"):
            if isinstance(getattr(self.scorer, name, None), nn.Module):
                _frozen(getattr(self.scorer, name))
        # the image tower (on the gpu; the scorers keep the text tower on the cpu) in dtype
        s = self.scorer
        if s.__class__.__name__ == "PickScore":
            s.model.vision_model.to(dtype)
            s.model.visual_projection.to(dtype)
        else:
            (s.clip if hasattr(s, "clip") else s.model).visual.to(dtype)
        self.text = None
        if self.uses_prompt:
            self.set_prompt(self.scorer.prompt)

    def set_prompt(self, prompt):
        if self.uses_prompt:
            self.text = self.prompt_embedding(prompt)
            self.label = str(self.scorer)

    def prompt_embedding(self, prompt):
        if not self.uses_prompt:
            return None
        self.scorer.set_prompt(prompt)
        # the scorer encodes the text under inference_mode: such a tensor can't
        # take part in a backward pass, a clone made here can
        return self.scorer.text.clone().float()

    def features(self, x):
        x = normalize(fit(x, INPUT_SIZE), CLIP_MEAN, CLIP_STD).to(self.dtype)
        s = self.scorer
        if s.__class__.__name__ == "PickScore":
            out = s.model.vision_model(pixel_values=x)
            return F.normalize(s.model.visual_projection(out.pooler_output).float(), dim=-1)
        model = s.clip if hasattr(s, "clip") else s.model
        return F.normalize(model.encode_image(x).float(), dim=-1)

    def score_of(self, features, text=None):
        s = self.scorer
        if hasattr(s, "mlp"):
            return s.mlp(features).squeeze(-1)
        return (features @ (self.text if text is None else text).T).squeeze(-1)


_FUNCTIONS = {"classifier": ClassifierFunction, "autoencoder": AutoencoderFunction,
              "ranknet": RankNetFunction, "aesthetic": AestheticFunction}


def list_models(root=None) -> list[dict]:
    """The models these functions can be made of: the saved custom models under
    root (default: env_config's models folder) and the aesthetic scores.
    -> [{"id", "label", "kind", "path", ...}]"""
    return [m for m in find_models(str(root or env_config.ROOT_MODEL_FOLDER)) if m["kind"] in KINDS]


_cache: dict[tuple[str, str, str], DifferentiableModel] = {}


def cache_key(model_id: str, device, precision: str = "fp32") -> tuple[str, str, str]:
    return (model_id, str(device), precision)


def release(keep=()) -> None:
    """Drop the loaded models but those of `keep` (cache_key()s): frees their gpu memory.
    Call it before loading what a run needs, so the old and the new don't share the gpu."""
    for key in [k for k in _cache if k not in keep]:
        del _cache[key]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load(model_id: str, device, root=None, precision: str = "fp32") -> DifferentiableModel:
    """The function of a model of list_models(), loaded once per (id, device, precision)
    and kept until release()."""
    if precision not in PRECISIONS:
        raise ValueError(f"bad precision {precision!r}: one of {list(PRECISIONS)}")
    key = cache_key(model_id, device, precision)
    if key not in _cache:
        entry = next((m for m in list_models(root) if m["id"] == model_id), None)
        if entry is None:
            raise ValueError(f"no custom model {model_id!r} in {root or env_config.ROOT_MODEL_FOLDER}")
        _cache[key] = _FUNCTIONS[entry["kind"]](entry, device, PRECISIONS[precision])
    return _cache[key]
