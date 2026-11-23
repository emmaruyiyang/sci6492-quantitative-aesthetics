"""Ready-made aesthetic / preference scores: one number per image, from models
somebody else trained on many people's ratings. On the map they are shown like
a RankNet's score (a histogram, best on the right).

(From DiffDrawing's aesthetic_scoring/image_ranker.py.)

  laion_ava   LAION aesthetic predictor: a small MLP on CLIP ViT-L/14 image
              embeddings, trained on AVA photo ratings (1-10). Favors
              photographic texture and color complexity.
  laion_sac   the same, trained on SAC (ratings of AI-generated images) + AVA
              + logos: ranks coherent structure higher, random-looking shapes
              lower.
  pickscore   PickScore (CLIP ViT-H/14 fine-tuned on Pick-a-Pic, people's
              choices between two generated images): how well the image fits
              a text prompt, as people judged it.
  hpsv2       Human Preference Score v2 (CLIP ViT-H/14 fine-tuned on human
              preference ratings of generated images): also against a prompt,
              more robust to artefacts than the AVA weights.

The prompt models compare the image with a text: a prompt that describes the
dataset ("a drawing of a cat") gives more telling scores than the neutral
default "an image".

Everything is downloaded on first use (the ViT-H/14 ones are several GB).
"""

import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import env_config
from shared.models.embedding_models import CLIP_MEAN, CLIP_STD, INPUT_SIZE, load_clip

DEFAULT_PROMPT = "an image"

_LAION_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/"
_LAION_WEIGHTS = {"laion_ava": "ava+logos-l14-linearMSE.pth", "laion_sac": "sac+logos+ava1-l14-linearMSE.pth"}
_LAION_CACHE = Path.home() / ".cache" / "laion_aesthetic"


def clip_preprocess(size: int = INPUT_SIZE):
    """What open_clip does to an image for its CLIP models"""
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def vision_on(model: nn.Module, device: torch.device, vision: tuple[str, ...]) -> nn.Module:
    """A CLIP model with only its image tower (the submodules named in vision) on device:
    the text tower is needed only to encode a prompt, so it stays on the cpu (for a
    ViT-H/14 model that is a third of the weights, ~1.3 GB)."""
    model.to("cpu").eval()
    for name in vision:
        getattr(model, name).to(device)
    return model


class AestheticScorer:
    label = ""
    uses_prompt = False

    def __init__(self, device: torch.device):
        self.device = device
        self.prompt = DEFAULT_PROMPT

    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt or DEFAULT_PROMPT

    def score(self, images: list[Image.Image]) -> torch.Tensor:
        """PIL images -> scores [B] on the cpu"""
        raise NotImplementedError

    def __str__(self) -> str:
        return self.label + (f', prompt "{self.prompt}"' if self.uses_prompt else "")


class _LaionMLP(nn.Module):
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class LaionAesthetic(AestheticScorer):
    """CLIP ViT-L/14 (openai) image embedding, unit length -> MLP -> about 1..10"""

    def __init__(self, device: torch.device, weights: str):
        super().__init__(device)
        self.label = {"laion_ava": "LAION aesthetic (AVA)", "laion_sac": "LAION aesthetic (SAC)"}[weights]
        file = _LAION_CACHE / _LAION_WEIGHTS[weights]
        if not file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {_LAION_WEIGHTS[weights]} -> {file}")
            urllib.request.urlretrieve(_LAION_URL + _LAION_WEIGHTS[weights].replace("+", "%2B"), file)
        self.clip = vision_on(load_clip("ViT-L-14", "openai"), device, ("visual",))  # no text used
        self.mlp = _LaionMLP(768)
        self.mlp.load_state_dict(torch.load(file, map_location="cpu"))
        self.mlp = self.mlp.to(device).eval()
        self.preprocess = clip_preprocess()

    @torch.inference_mode()
    def score(self, images):
        x = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device)
        embedding = F.normalize(self.clip.encode_image(x).float(), dim=-1)
        return self.mlp(embedding).squeeze(-1).float().cpu()


class HPSv2(AestheticScorer):
    """open_clip ViT-H/14 with the HPSv2 weights: cosine of image and prompt"""
    label = "Human Preference Score v2"
    uses_prompt = True

    def __init__(self, device: torch.device):
        super().__init__(device)
        import open_clip
        from huggingface_hub import hf_hub_download
        self.model = open_clip.create_model("ViT-H-14", pretrained="laion2b_s32b_b79k")
        checkpoint = torch.load(hf_hub_download("xswu/HPSv2", "HPS_v2_compressed.pt", token=env_config.HF_TOKEN),
                                map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=False)
        self.model = vision_on(self.model, device, ("visual",))
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self.preprocess = clip_preprocess()
        self.text = None
        self.set_prompt(DEFAULT_PROMPT)

    @torch.inference_mode()
    def set_prompt(self, prompt):
        super().set_prompt(prompt)
        # on the cpu, where the text tower is
        self.text = F.normalize(self.model.encode_text(self.tokenizer([self.prompt])).float(), dim=-1).to(self.device)

    @torch.inference_mode()
    def score(self, images):
        x = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device)
        embedding = F.normalize(self.model.encode_image(x).float(), dim=-1)
        return (embedding @ self.text.T).squeeze(-1).cpu()


class PickScore(AestheticScorer):
    """yuvalkirstain/PickScore_v1 (transformers): cosine of image and prompt"""
    label = "PickScore"
    uses_prompt = True

    def __init__(self, device: torch.device):
        super().__init__(device)
        from transformers import AutoModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", token=env_config.HF_TOKEN)
        self.model = vision_on(AutoModel.from_pretrained("yuvalkirstain/PickScore_v1", token=env_config.HF_TOKEN),
                               device, ("vision_model", "visual_projection"))
        self.text = None
        self.set_prompt(DEFAULT_PROMPT)

    @torch.inference_mode()
    def set_prompt(self, prompt):
        super().set_prompt(prompt)
        inputs = self.processor(text=self.prompt, return_tensors="pt", padding=True, truncation=True)
        # the sub-models directly: get_text_features' return type changes between versions.
        # On the cpu, where the text tower is.
        out = self.model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        self.text = F.normalize(self.model.text_projection(out.pooler_output).float(), dim=-1).to(self.device)

    @torch.inference_mode()
    def score(self, images):
        inputs = self.processor(images=[img.convert("RGB") for img in images], return_tensors="pt")
        out = self.model.vision_model(pixel_values=inputs["pixel_values"].to(self.device))
        embedding = F.normalize(self.model.visual_projection(out.pooler_output).float(), dim=-1)
        return (embedding @ self.text.T).squeeze(-1).cpu()


# id -> (label, uses a prompt, constructor)
SCORERS = {
    "laion_ava": ("LAION aesthetic (AVA)", False, lambda device: LaionAesthetic(device, "laion_ava")),
    "laion_sac": ("LAION aesthetic (SAC)", False, lambda device: LaionAesthetic(device, "laion_sac")),
    "pickscore": ("PickScore", True, PickScore),
    "hpsv2": ("Human Preference Score v2", True, HPSv2),
}


def load_scorer(scorer_id: str, device: torch.device) -> AestheticScorer:
    return SCORERS[scorer_id][2](device)
