import json
import sys
from collections.abc import Callable
from dataclasses import replace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from pathlib import Path
from PIL import Image
from .model_types import (DINOV2Arch, CLIPArch, CLIPPretrained, ImageNetArch, ModelID,
                          default_clip_model_id, default_dino_model_id)
from .embedding_models import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, image_normalization, load_model
from .frozen import precision_dtype


class EmbeddingCache:
    """Caches image embeddings for multiple models, keyed by ModelID."""

    ATTR_NAME = '_embedding_cache'

    @staticmethod
    def try_get_cache(obj: object) -> 'EmbeddingCache':
        return getattr(obj, EmbeddingCache.ATTR_NAME, None)
    
    @staticmethod
    def try_get_embeddings(obj: object, model_id: ModelID) -> torch.Tensor | None:
        cache = EmbeddingCache.try_get_cache(obj)
        if cache is None:
            return None
        return cache.get(model_id)

    @staticmethod
    def set_embeddings(obj: object, model_id: ModelID, embeddings: torch.Tensor) -> None:
        cache = EmbeddingCache.try_get_cache(obj)
        if cache is None:
            cache = EmbeddingCache()
            setattr(obj, EmbeddingCache.ATTR_NAME, cache)
        cache.set(model_id, embeddings)

    def __init__(self):
        self._cache: dict[ModelID, torch.Tensor] = {}

    def get(self, model_id: ModelID) -> torch.Tensor | None:
        return self._cache.get(model_id)

    def set(self, model_id: ModelID, embeddings: torch.Tensor) -> None:
        self._cache[model_id] = embeddings

_CLIP_INPUT_SIZE = INPUT_SIZE
_CLIP_MEAN = CLIP_MEAN
_CLIP_STD  = CLIP_STD

_DINO_INPUT_SIZE = INPUT_SIZE
_DINO_MEAN = IMAGENET_MEAN
_DINO_STD  = IMAGENET_STD
DINO_TOKENS = ("cls", "patch")  # dino_image_embedding: CLS token | all patch tokens, flattened

# torchvision ImageNet classifiers: same ImageNet statistics as DINOv2.
_IMAGENET_INPUT_SIZE = INPUT_SIZE
_IMAGENET_MEAN = IMAGENET_MEAN
_IMAGENET_STD  = IMAGENET_STD

_EMBED_BATCH = 32


# ── Image-folder helpers ───────────────────────────────────────────────────────

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def print_progress(done: int, total: int, label: str = "Embedding") -> None:
    """Single-line, in-place console progress bar (flushed, so it shows in real time)."""
    width = 30
    filled = width * done // max(total, 1)
    sys.stdout.write(f"\r  {label} [{'#' * filled}{'.' * (width - filled)}] {done}/{total}")
    if done >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def _emb_filename(model_id: 'ModelID') -> str:
    return f"{model_id.filename_stem()}_embeddings.npy"


def _letterbox_to_tensor(img: Image.Image, size: int = 224) -> torch.Tensor:
    """Resize longest side to `size`, pad shorter side with mid-grey, return CHW tensor."""
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return TF.to_tensor(canvas)


def letterbox_tensor(images: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Differentiable twin of _letterbox_to_tensor for a (B, 3, H, W) batch in [0, 1]:
    antialiased bilinear resize (vs. LANCZOS there) onto a mid-grey square canvas."""
    _, _, h, w = images.shape
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = F.interpolate(images, size=(new_h, new_w), mode="bilinear",
                            antialias=True, align_corners=False)
    canvas = torch.full((images.shape[0], 3, size, size), 128 / 255,
                        dtype=images.dtype, device=images.device)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[:, :, top : top + new_h, left : left + new_w] = resized
    return canvas


# ── MPS patch-embedding workaround ─────────────────────────────────────────────
# A ViT's first layer is a non-overlapping conv (kernel == stride) that cuts the
# image into patches. On MPS, `convolution_backward`'s grad_input kernel costs
# O(kernel_area): measured on ViT-B/32's 32x32 patches, that ONE op took ~7000ms
# per step while the entire rest of the CLIP tower took ~46ms (k=16 -> 1740ms,
# k=8 -> 437ms, a clean 4x per doubling). It only bites when the input image
# requires grad -- i.e. exactly the CLIP-loss path, which is why SDS (a frozen
# no_grad oracle, see model_diffusion_central) never hit it and CUDA, whose cuDNN
# kernels are fine, shows the opposite CLIP/SDS ranking.
#
# For kernel == stride the conv is exactly "cut into patches, then one matmul",
# whose backward is a plain matmul. Gradients come out bit-identical (verified:
# max |grad diff| == 0.0); the forward differs only by float32 summation-order
# rounding (~7e-6 absolute). MPS-only -- CUDA keeps the real conv.


class _PatchEmbedLinear(nn.Module):
    """Non-overlapping patch-embedding conv rewritten as patchify + matmul."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        out_ch, in_ch, kh, kw = conv.weight.shape
        self.out_ch, self.in_ch, self.kh, self.kw = out_ch, in_ch, kh, kw
        # (out, in*kh*kw) -> transposed once here so forward is a straight matmul.
        weight = conv.weight.detach().reshape(out_ch, in_ch * kh * kw).t().contiguous()
        self.weight = nn.Parameter(weight, requires_grad=conv.weight.requires_grad)
        if conv.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(conv.bias.detach().clone(),
                                     requires_grad=conv.bias.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H % self.kh or W % self.kw:
            # Ragged input the patch grid can't tile — fall back to the real conv
            # (correctness over speed; the preprocessors resize to a multiple).
            weight = self.weight.t().reshape(self.out_ch, self.in_ch, self.kh, self.kw)
            return F.conv2d(x, weight, self.bias, stride=(self.kh, self.kw))
        gh, gw = H // self.kh, W // self.kw
        # (B, C, gh, kh, gw, kw) -> (B, gh, gw, C, kh, kw): each patch flattened in
        # (in, kh, kw) order, matching how the conv weight rows are laid out.
        x = x.reshape(B, C, gh, self.kh, gw, self.kw)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, gh * gw, C * self.kh * self.kw)
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out.permute(0, 2, 1).reshape(B, self.out_ch, gh, gw)


def _install_fast_patch_embed(module: object, attr: str) -> bool:
    """Swap ``module.attr`` for its patchify+matmul equivalent, if it's a
    non-overlapping conv. Returns whether the swap happened."""
    conv = getattr(module, attr, None)
    if not isinstance(conv, nn.Conv2d):
        return False
    kh, kw = conv.kernel_size
    if (tuple(conv.stride) != (kh, kw) or conv.dilation != (1, 1)
            or conv.groups != 1 or conv.padding not in (0, (0, 0), 'valid')):
        return False
    setattr(module, attr, _PatchEmbedLinear(conv).to(conv.weight.device))
    return True


# ── ModelCentral ───────────────────────────────────────────────────────────────

class ModelCentral:
    """Lazily loads and caches inference models shared across loss evaluators.

    A CLIP ModelID's `extra` is its image tower's precision: "" or "fp32", or "bf16" (a
    separate model, with its own cache entries). The text tower always stays fp32.
    text_on_cpu keeps CLIP text towers off the device: slower text encoding, for callers
    that encode a few prompts rather than long word lists, and a third less device memory."""

    def __init__(self, device: torch.device, text_on_cpu: bool = False):
        self.device = device
        self.text_on_cpu = text_on_cpu
        self._models: dict[ModelID, object] = {}
        self._text_embedding_cache: dict[tuple[ModelID, str], torch.Tensor] = {}
        self._tokenizers: dict[ModelID, Callable] = {}

    def release(self, keep=()) -> None:
        """Drop the loaded models (and their cached text embeddings) but those of `keep`
        (ModelIDs): frees their gpu memory."""
        for model_id in [m for m in self._models if m not in keep]:
            del self._models[model_id]
        for key in [k for k in self._text_embedding_cache if k[0] not in keep]:
            del self._text_embedding_cache[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_model(self, model_id: ModelID, device: torch.device) -> object | None:
        if model_id in self._models:
            return self._models[model_id]
        try:
            model = load_model(model_id).eval()
            if model_id.is_clip():
                if not self.text_on_cpu:
                    model.to(device)
                model.visual.to(device, precision_dtype(model_id.extra or "fp32"))
            else:
                model.to(device)
            # These are frozen oracles: the loss backprops into the *image*, never
            # into CLIP/DINO. `.eval()` alone doesn't say so — open_clip's params
            # arrive with requires_grad=True, so every backward computed (and
            # buffered) a gradient for all ~88M ViT-B/32 weights that nothing ever
            # reads, since the optimizer only sees graph + image params. Measured
            # 1.35x on the CLIP loss; grads still reach the input either way.
            model.requires_grad_(False)
            if device.type == 'mps':
                # See _PatchEmbedLinear: MPS's conv grad_input is O(kernel_area) and
                # dominates every step that backprops into the image.
                host, attr = ((getattr(model, 'patch_embed', None), 'proj') if model_id.is_dino()
                              else (model, 'conv_proj') if model_id.is_imagenet()  # torchvision ViT
                              else (getattr(model, 'visual', None), 'conv1'))
                if _install_fast_patch_embed(host, attr):
                    print(f'[ModelCentral] {model_id}: patch-embed conv -> patchify+matmul (MPS)')
            self._models[model_id] = model
            return model
        except Exception as exc:
            print(f"[ModelCentral] model unavailable ({model_id}): {exc}")
            self._models[model_id] = None

    # ── CLIP ──────────────────────────────────────────────────────────────────

    def _preprocess_for_clip(self, image: torch.Tensor, model_id: ModelID) -> torch.Tensor:
        device = image.device
        img = transforms.Resize((_CLIP_INPUT_SIZE, _CLIP_INPUT_SIZE))(image)
        clip_mean, clip_std = image_normalization(model_id)  # CLIP's statistics, SigLIP's -1..1
        mean = torch.tensor(clip_mean, device=device).view(1, 3, 1, 1)
        std  = torch.tensor(clip_std,  device=device).view(1, 3, 1, 1)
        return (img - mean) / std

    def _tokenize(self, texts: list[str], model_id: ModelID) -> torch.Tensor:
        """open_clip's tokenizer of the model (CLIP's BPE; SigLIP: its own vocabulary and length)"""
        if model_id not in self._tokenizers:
            import open_clip
            self._tokenizers[model_id] = open_clip.get_tokenizer(model_id.arch)
        return self._tokenizers[model_id](texts)

    def clip_image_embedding(self, image: torch.Tensor, model_id: ModelID = default_clip_model_id()) -> torch.Tensor | None:
        """Unit-normalized CLIP image embedding. Gradients flow through."""
        embeddings = EmbeddingCache.try_get_embeddings(image, model_id)
        if embeddings is not None:
            return embeddings.to(image.device)

        model = self._ensure_model(model_id, image.device)
        if model is None:
            return None
        dtype = next(model.visual.parameters()).dtype
        features = model.encode_image(self._preprocess_for_clip(image, model_id).to(dtype)).float()
        embeddings =  features / features.norm(dim=-1, keepdim=True)
        EmbeddingCache.set_embeddings(image, model_id, embeddings)
        return embeddings

    def clip_text_embedding(self, text: str, device: torch.device, model_id: ModelID = default_clip_model_id()) -> torch.Tensor | None:
        """Unit-normalized CLIP text embedding, cached per (model, text) pair."""
        model = self._ensure_model(model_id, device)
        if model is None:
            return None
        cache_key = (model_id, text)
        if cache_key not in self._text_embedding_cache:
            tokens = self._tokenize([text], model_id).to("cpu" if self.text_on_cpu else device)
            with torch.no_grad():
                features = model.encode_text(tokens)
            self._text_embedding_cache[cache_key] = (
                features / features.norm(dim=-1, keepdim=True)
            ).detach()
        return self._text_embedding_cache[cache_key].to(device)

    def clip_text_embeddings(self, texts: list[str], device: torch.device, model_id: ModelID = default_clip_model_id(),
                             batch: int = 256) -> torch.Tensor | None:
        """(len(texts), D) unit-normalized CLIP text embeddings, encoded in batches (not cached)."""
        model = self._ensure_model(model_id, device)
        if model is None:
            return None
        parts = []
        with torch.no_grad():
            for start in range(0, len(texts), batch):
                tokens = self._tokenize(texts[start : start + batch], model_id)
                features = model.encode_text(tokens.to("cpu" if self.text_on_cpu else device))
                parts.append(features / features.norm(dim=-1, keepdim=True))
        return torch.cat(parts).float().to(device)

    # ── DINOv2 ────────────────────────────────────────────────────────────────

    def _preprocess_for_dino(self, image: torch.Tensor) -> torch.Tensor:
        device = image.device
        img = transforms.Resize((_DINO_INPUT_SIZE, _DINO_INPUT_SIZE))(image)
        mean = torch.tensor(_DINO_MEAN, device=device).view(1, 3, 1, 1)
        std  = torch.tensor(_DINO_STD,  device=device).view(1, 3, 1, 1)
        return (img - mean) / std

    def dino_image_embedding(self, image: torch.Tensor, model_id: ModelID = default_dino_model_id(),
                             tokens: str = "cls") -> torch.Tensor | None:
        """DINOv2 embedding: the CLS token (tokens="cls"), or (tokens="patch") the M patch
        tokens, each unit-normalized, flattened to M*D in grid order and scaled to unit
        length -- a dot product is then the mean per-patch cosine, so layout counts.
        Gradients flow through."""
        if tokens not in DINO_TOKENS:
            raise ValueError(f"Bad DINOv2 tokens: {tokens!r}")
        cache_key = model_id if tokens == "cls" else replace(model_id, extra=f"{model_id.extra}+patch")
        embeddings = EmbeddingCache.try_get_embeddings(image, cache_key)
        if embeddings is not None:
            return embeddings.to(image.device)

        model = self._ensure_model(model_id, image.device)
        if model is None:
            return None
        if tokens == "cls":
            features = model(self._preprocess_for_dino(image))
        else:
            patches = model.forward_features(self._preprocess_for_dino(image))["x_norm_patchtokens"]  # (B, M, D)
            features = F.normalize(patches, dim=-1).flatten(1)  # norm sqrt(M); scaled to 1 below
        embeddings = features / features.norm(dim=-1, keepdim=True)
        EmbeddingCache.set_embeddings(image, cache_key, embeddings)
        return embeddings
    # ── torchvision ImageNet classifiers ─────────────────────────────────────

    def _preprocess_for_imagenet(self, image: torch.Tensor) -> torch.Tensor:
        device = image.device
        img = transforms.Resize((_IMAGENET_INPUT_SIZE, _IMAGENET_INPUT_SIZE))(image)
        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  device=device).view(1, 3, 1, 1)
        return (img - mean) / std

    def imagenet_logits(self, image: torch.Tensor, model_id: ModelID) -> torch.Tensor | None:
        """Raw (B, 1000) ImageNet class logits -- NOT normalized; softmax them for
        probabilities. Gradients flow through."""
        logits = EmbeddingCache.try_get_embeddings(image, model_id)
        if logits is not None:
            return logits.to(image.device)

        model = self._ensure_model(model_id, image.device)
        if model is None:
            return None
        logits = model(self._preprocess_for_imagenet(image))
        EmbeddingCache.set_embeddings(image, model_id, logits)
        return logits

    @staticmethod
    def imagenet_classes(model_id: ModelID) -> list[str]:
        """The 1000 class names, in logit order, for an ImageNet model's weights."""
        import torchvision.models
        weights = torchvision.models.get_model_weights(model_id.arch)[model_id.pretrained]
        return list(weights.meta["categories"])

    # ── Unified ───────────────────────────────────────────────────────────────

    def image_embedding(self, image: torch.Tensor, model_id: ModelID, dino_tokens: str = "cls") -> torch.Tensor | None:
        """Per-image vector by model_id.family: unit-normalized CLIP / DINOv2 embedding
        (DINOv2: see dino_image_embedding's tokens), raw class logits for ImageNet classifiers,
        or a custom model's features (custom_models.differentiable: a classifier's logits, an
        autoencoder's latent, the vector a RankNet / aesthetic score is computed from; its
        `extra` is the precision, '' = fp32). Gradients flow through."""
        if model_id.is_custom():
            from shared.custom_models import differentiable
            return differentiable.load(model_id.arch, image.device, precision=model_id.extra or "fp32").features(image)
        if model_id.is_dino():
            return self.dino_image_embedding(image, model_id, dino_tokens)
        if model_id.is_imagenet():
            return self.imagenet_logits(image, model_id)
        return self.clip_image_embedding(image, model_id)

    # ── Image-folder embedding cache ──────────────────────────────────────────

    def ensure_image_order(self, folder: Path) -> list[str]:
        """Return the image list for a folder from image_order.json, or by scanning."""
        folder = Path(folder)
        order_file = folder / "image_order.json"
        if order_file.exists():
            return json.loads(order_file.read_text())
        order = sorted(
            p.name for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        order_file.write_text(json.dumps(order))

        return order

    def embed_pil_images(self, pil_images: list[Image.Image], model_id: ModelID) -> np.ndarray:
        """(N, D) image_embedding rows, preprocessed exactly like ensure_embeddings."""
        batch = torch.stack([_letterbox_to_tensor(img.convert("RGB")) for img in pil_images]).to(self.device)
        with torch.no_grad():
            emb = self.image_embedding(batch, model_id)
        if emb is None:
            raise RuntimeError(f"Failed to embed images with {model_id}")
        return emb.cpu().float().numpy()

    def ensure_embeddings(self, folder: Path, model_id: ModelID,
                          progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        """Return cached embeddings for model_id in folder, creating them if absent.
        `progress(done, total)` is called after each batch while creating."""
        folder = Path(folder)
        cache_file = folder / _emb_filename(model_id)
        if cache_file.exists():
            print(f"Loading cached embeddings from {cache_file.name}")
            return np.load(str(cache_file))

        image_order = self.ensure_image_order(folder)
        if not image_order:
            raise RuntimeError(f"No images found in {folder}")

        print(f"Creating embeddings for {model_id} ...", flush=True)
        parts = []
        total = len(image_order)
        for start in range(0, total, _EMBED_BATCH):
            batch_names = image_order[start : start + _EMBED_BATCH]
            parts.append(self.embed_pil_images([Image.open(folder / n) for n in batch_names], model_id))
            done = min(start + _EMBED_BATCH, total)
            print_progress(done, total)
            if progress is not None:
                progress(done, total)

        result = np.concatenate(parts, axis=0)
        np.save(str(cache_file), result)
        print(f"Saved -> {cache_file.name}")
        return result
