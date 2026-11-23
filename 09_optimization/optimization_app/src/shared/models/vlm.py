"""Vision-language models (VLMArch: Qwen3-VL, Qwen2.5-VL, SmolVLM2, InternVL3.5, through
transformers) as differentiable functions of an image: asked a question about it, how
likely the model is to give an answer. For optimizing an image against them
(03_optimization).

A query is a (question, answer) pair. Its score is log p(answer | image, question), the
answer teacher forced after the model's chat template ("user: <image> question /
assistant: answer"): the sum of its tokens' log-probabilities (the answer's probability),
or their mean (per token: long answers are not penalized for their length). Two answers to
one question compare as a log-odds -- log p(Yes) - log p(No) is how sure the model is that
the answer is yes (the VQAScore of Lin et al. 2024).

The image path is the processor's, made with tensor operations so the gradient reaches the
pixels: the canvas to the model's fixed square input (fit: short side, center crop), its
normalization, and its pixel layout (Qwen's flattened patches, SmolVLM's image list,
InternVL's single tile). The processor itself only tokenizes the text, against a dummy
image of that size, so its image placeholder tokens match the pixels given instead.

Every image goes through the vision encoder once, its tokens shared by all the queries.
The image features (features(); the edit size of an optimization) are those tokens'
mean, as the language model reads them.

Models load once per (repo, device, precision) and stay until release(). Precision "bf16"
(the default) or "fp32".
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from PIL import Image

from .frozen import ModelCache, fit, frozen, normalize, precision_dtype
from .model_types import VLMArch

__all__ = ["VLMArch", "VLM", "VLMQueries", "SCORES", "load", "release", "cache_key"]

SCORES = ("logprob", "mean")   # a query's score: its answer's summed / mean token log-probability
QWEN_IMAGE_SIZE = 448          # Qwen's input side (dynamic resolution, fixed here; rounded to its
                               # patch x merge size): 448 -> 196 image tokens (Qwen3-VL), 256 (Qwen2.5-VL)
GRADIENT_CHECKPOINTING = True  # backward recomputes each layer instead of keeping its activations: every
                               # view x query is a sequence of a few hundred tokens through the whole
                               # model, too much to keep (InternVL3.5-1B, 8 views x 5 queries: > 20 GB);
                               # about a third more compute

# The processor's options that keep one image one fixed square input, by layout
_PROCESSOR_KWARGS = {"qwen": {}, "idefics3": {"do_image_splitting": False}, "plain": {"crop_to_patches": False}}
_LAYOUTS = {"qwen2_vl": "qwen", "qwen2_5_vl": "qwen", "qwen3_vl": "qwen",
            "smolvlm": "idefics3", "idefics3": "idefics3", "internvl": "plain"}


@dataclass
class VLMQueries:
    """(question, answer) pairs tokenized once (VLM.queries), Q rows right-padded to one
    length L. answer_pos (Q, A): the positions whose logits predict the answer's tokens
    answer_ids (Q, A), answer_mask (Q, A) the real ones."""
    pairs: list[tuple[str, str]]
    text: dict[str, torch.Tensor]       # input_ids, attention_mask, ... (Q, L)
    image: dict[str, torch.Tensor]      # the processor's other image inputs (image_grid_thw, ...), (Q, ...)
    answer_pos: torch.Tensor
    answer_ids: torch.Tensor
    answer_mask: torch.Tensor


class VLM(nn.Module):
    def __init__(self, arch: str, device, dtype: torch.dtype):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.arch = arch
        self.label = arch.split("/")[-1]
        self.dtype = dtype
        self.device = torch.device(device)
        try:
            self.processor = AutoProcessor.from_pretrained(arch)
            self.model = frozen(AutoModelForImageTextToText.from_pretrained(arch, dtype=dtype).to(device))
        except Exception as exc:
            raise RuntimeError(f"VLM {arch!r} could not be loaded ({type(exc).__name__}: {exc}). It needs "
                               "`transformers` and the model in the Hugging Face cache or online.") from exc
        model_type = self.model.config.model_type
        if model_type not in _LAYOUTS:
            raise ValueError(f"VLM {arch!r}: no pixel layout for model type {model_type!r}")
        self.layout = _LAYOUTS[model_type]
        ip = self.processor.image_processor
        self.mean, self.std = list(ip.image_mean), list(ip.image_std)
        if self.layout == "qwen":
            self.patch, self.merge, self.temporal = ip.patch_size, ip.merge_size, ip.temporal_patch_size
            factor = self.patch * self.merge
            self.size = max(factor, round(QWEN_IMAGE_SIZE / factor) * factor)
        elif self.layout == "idefics3":
            self.size = ip.max_image_size["longest_edge"]
        else:
            self.size = ip.size["height"]
        self.processor_kwargs = _PROCESSOR_KWARGS[self.layout]
        if GRADIENT_CHECKPOINTING and self.model.supports_gradient_checkpointing:
            # Checkpointing acts only on a layer in training mode: set just the layers' own
            # flag, not their submodules' (their dropout stays off).
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            for module in self.model.modules():
                if getattr(module, "gradient_checkpointing", False):
                    module.training = True
        tokenizer = self.processor.tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # ── Text ──────────────────────────────────────────────────────────────────

    def _prompt(self, question: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def queries(self, pairs: list[tuple[str, str]]) -> VLMQueries:
        """Tokenize (question, answer) pairs: each the chat prompt with the image (a dummy of
        the input size: the placeholder tokens of this many pixels), the answer's tokens after it."""
        if not pairs:
            raise ValueError("no queries")
        dummy = Image.new("RGB", (self.size, self.size), (128, 128, 128))
        rows, images, answers = [], [], []
        for question, answer in pairs:
            prompt = self._prompt(question)
            enc = self.processor(text=[prompt], images=[[dummy]], return_tensors="pt", **self.processor_kwargs)
            ids = enc["input_ids"]
            # The prompt ends with the assistant's turn: an answer after a word (SmolVLM's
            # "Assistant:") starts with a space, as the model would write it.
            if not prompt[-1].isspace():
                answer = " " + answer.lstrip()
            answer_ids = torch.tensor(self.processor.tokenizer(answer, add_special_tokens=False)["input_ids"])
            if not len(answer_ids):
                raise ValueError(f"empty answer to {question!r}")
            text = {k: v[0] for k, v in enc.items() if torch.is_tensor(v) and v.shape == ids.shape}
            text = {k: torch.cat([v, answer_ids if k == "input_ids" else
                                  torch.full_like(answer_ids, 1 if k == "attention_mask" else 0)])
                    for k, v in text.items()}
            rows.append(text)
            images.append({k: v for k, v in enc.items() if k not in text and k != "pixel_values" and torch.is_tensor(v)})
            answers.append((len(ids[0]), answer_ids))
        length = max(len(r["input_ids"]) for r in rows)
        text = {k: torch.stack([torch.cat([r[k], r[k].new_full((length - len(r[k]),),
                                                               self.pad_id if k == "input_ids" else 0)])
                                for r in rows]) for k in rows[0]}
        image = {k: torch.cat([im[k] for im in images]) for k in images[0]}
        width = max(len(a) for _, a in answers)
        pos = torch.zeros(len(pairs), width, dtype=torch.long)
        ids = torch.zeros(len(pairs), width, dtype=torch.long)
        mask = torch.zeros(len(pairs), width, dtype=torch.bool)
        for q, (start, a) in enumerate(answers):
            pos[q, :len(a)] = torch.arange(start - 1, start - 1 + len(a))  # the logits before each answer token
            ids[q, :len(a)] = a
            mask[q, :len(a)] = True
        to = lambda d: {k: v.to(self.device) for k, v in d.items()}
        return VLMQueries(list(pairs), to(text), to(image), pos.to(self.device), ids.to(self.device),
                          mask.to(self.device))

    # ── Image ─────────────────────────────────────────────────────────────────

    def pixels(self, x: torch.Tensor) -> torch.Tensor:
        """images (N, 3, H, W) in 0..1 -> the model's pixel_values for each, with a leading N:
        Qwen (N, patches, C * temporal * patch^2), SmolVLM (N, 1, 3, S, S), InternVL (N, 3, S, S)"""
        x = normalize(fit(x.float(), self.size), self.mean, self.std)
        if self.layout == "qwen":
            # Qwen2VLImageProcessor.patchify: merge-size blocks of patches, each patch
            # repeated over the temporal patch (a still image is two identical frames)
            n, c, h, w = x.shape
            p, m, t = self.patch, self.merge, self.temporal
            gh, gw = h // p, w // p
            x = x.reshape(n, c, gh // m, m, p, gw // m, m, p).permute(0, 2, 5, 3, 6, 1, 4, 7)
            x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, t, -1, -1).reshape(n, gh * gw, c * t * p * p)
        elif self.layout == "idefics3":
            x = x.unsqueeze(1)
        return x.to(self.dtype)

    def encode(self, x: torch.Tensor, queries: VLMQueries):
        """images (N, 3, H, W) in 0..1 -> the vision encoder's output for them (the model's
        get_image_features: pooler_output, the projected image tokens; Qwen3-VL also its
        deepstack features), each image once"""
        kwargs = {k: v[:1].repeat(len(x), *[1] * (v.dim() - 1)) for k, v in queries.image.items()}
        config = self.model.config
        for key in ("vision_feature_layer", "vision_feature_select_strategy"):   # InternVL
            if hasattr(config, key):
                kwargs[key] = getattr(config, key)
        return self.model.model.get_image_features(pixel_values=self.pixels(x), return_dict=True, **kwargs)

    def forward(self, x: torch.Tensor, queries: VLMQueries, score: str = "logprob",
                with_features: bool = False):
        """images (N, 3, H, W) in 0..1 -> (N, Q) query scores (see SCORES), and with_features
        also (N, D) image features. Gradients flow to the images. Every image goes through
        the vision encoder once; its tokens are shared by its Q sequences (image-major:
        sequence n * Q + q is image n, query q)."""
        if score not in SCORES:
            raise ValueError(f"bad score {score!r}: one of {SCORES}")
        n, q = len(x), len(queries.pairs)
        image = self.encode(x, queries)
        features = self._pool(image.pooler_output) if with_features else None
        image.pooler_output = _repeat_images(image.pooler_output, q)
        if getattr(image, "deepstack_features", None) is not None:
            image.deepstack_features = [_repeat_images(f, q) for f in image.deepstack_features]
        inputs = {k: v.repeat(n, *[1] * (v.dim() - 1)) for k, v in {**queries.text, **queries.image}.items()
                  if k != "pixel_attention_mask"}
        hidden = self.model.model(**inputs, mm_encoder_outputs={"image": image}, use_cache=False).last_hidden_state
        b = len(hidden)                                                    # N * Q
        rows = torch.arange(b, device=hidden.device)[:, None]
        pos, ids, mask = (t.repeat(n, 1) for t in (queries.answer_pos, queries.answer_ids, queries.answer_mask))
        logits = self.model.lm_head(hidden[rows, pos]).float()             # (N * Q, A, vocab)
        token_lp = logits.log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1) * mask
        scores = token_lp.sum(-1)
        if score == "mean":
            scores = scores / mask.sum(-1)
        scores = scores.view(n, q)
        return (scores, features) if with_features else scores

    @staticmethod
    def _pool(tokens) -> torch.Tensor:
        """The projected image tokens ((N, T, D), or Qwen's tuple of N (T, D)) -> (N, D), their mean"""
        if torch.is_tensor(tokens):
            return tokens.float().mean(1)
        return torch.stack([t.float().mean(0) for t in tokens])

    def features(self, x: torch.Tensor, queries: VLMQueries) -> torch.Tensor:
        """images -> (N, D) image features: the mean of the image tokens the language model
        reads (the vision encoder's projected output; no language model pass)"""
        return self._pool(self.encode(x, queries).pooler_output)


def _repeat_images(v, k: int):
    """Per-image vision outputs, each image repeated k times in place: a (N, ...) tensor, or a
    tuple of N per-image tensors (Qwen)"""
    if torch.is_tensor(v):
        return v.repeat_interleave(k, dim=0)
    return tuple(t for t in v for _ in range(k))


_cache = ModelCache()
cache_key = ModelCache.key
release = _cache.release


def load(arch: str | VLMArch, device, precision: str = "bf16") -> VLM:
    """The VLM of a Hugging Face repo, loaded once per (repo, device, precision) and kept
    until release()."""
    dtype = precision_dtype(precision)
    arch = str(arch)
    return _cache.get(cache_key(arch, device, precision), lambda: VLM(arch, device, dtype))
