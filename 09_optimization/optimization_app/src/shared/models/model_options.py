"""The image models an app can offer, by family: the one list the optimization app
(03_optimization) and the CLIP explorer (02_all_are_found) both choose from.

  clip       every CLIPArch x CLIPPretrained pair with weights (open_clip's own and
             those on the Hugging Face hub, HUB_CLIP_WEIGHTS)
  imagenet   every ImageNetArch with its default (best) torchvision weights
  dino       every DINOV2Arch
  vlm        every VLMArch (shared.models.vlm)
  aesthetic  the ready-made aesthetic scores    } ModelID.custom(id), listed by
  custom     the models trained in custom_models } custom_models()

Each *_model_ids() returns ModelIDs; the apps send their str() to the page.
"""

from .model_types import CLIPArch, CLIPPretrained, DINOV2Arch, HUB_CLIP_WEIGHTS, ImageNetArch, ModelID, VLMArch

MODEL_FAMILIES = ["clip", "imagenet", "dino", "vlm", "aesthetic", "custom"]
CUSTOM_FAMILIES = ("aesthetic", "custom")   # both custom ModelIDs; an aesthetic score's id is aesthetic/<scorer>
SCORE_GOALS = ["maximize", "minimize", "value"]
IMAGENET_SCORES = ["logits", "softmax"]


def clip_model_ids() -> list[ModelID]:
    import open_clip
    available = set(open_clip.list_pretrained()) | set(HUB_CLIP_WEIGHTS)
    return [m for m in (ModelID.clip(a, p) for a in CLIPArch for p in CLIPPretrained)
            if (m.arch, m.pretrained) in available]


def imagenet_model_ids() -> list[ModelID]:
    from .embedding_models import default_imagenet_weights
    return [ModelID.imagenet(a, default_imagenet_weights(a)) for a in ImageNetArch]


def dino_model_ids() -> list[ModelID]:
    return [ModelID.dino(a) for a in DINOV2Arch]


def vlm_model_ids() -> list[ModelID]:
    return [ModelID.vlm(a) for a in VLMArch]


def custom_family(kind: str) -> str:
    """A custom model's family by its kind (custom_models())"""
    return "aesthetic" if kind == "aesthetic" else "custom"


def family_of(model_id: ModelID) -> str:
    if model_id.is_custom():
        return "aesthetic" if model_id.arch.startswith("aesthetic/") else "custom"
    return next(f for f in MODEL_FAMILIES if f not in CUSTOM_FAMILIES and getattr(model_id, f"is_{f}")())


def custom_models() -> list[dict]:
    """The custom models (shared.custom_models.differentiable: classifiers, autoencoders,
    RankNets in env_config's models folder, the aesthetic scores) for a page: {id, label,
    kind, prompt (scores against a text), classes (classifiers)}. Read again on every
    call; a classifier whose config can't be read is left out."""
    from shared.custom_models import differentiable
    from shared.custom_models.autoencoder.autoencoder_model import read_config
    out = []
    for m in differentiable.list_models():
        entry = {"id": m["id"], "label": m["label"], "kind": m["kind"], "prompt": bool(m.get("prompt"))}
        if m["kind"] == "classifier":
            try:
                entry["classes"] = read_config(m["path"])["class_names"]
            except (OSError, ValueError, KeyError):
                continue
        out.append(entry)
    return out
