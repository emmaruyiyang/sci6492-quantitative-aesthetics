"""Web front end for optimization.py: set every parameter in the browser, (re)start
runs, and watch the canvas plus a map of the targets -- each text / direction / image
target of every model is a line through the origin at equal angles, the image's position is the
sum of its per-target values along those lines, and its path is drawn as a trace
(both poles of a BIDIRECTIONAL run; the cloud of dots is the current augmented views).

The optimization itself is optimization.py, configured through its module-level
settings (see apply_config), run on a background thread with run_pole's step
callback feeding the shared state that the page polls.

Run:  uv run python src/optimization_app.py   (then http://localhost:5001)
"""
import argparse
import base64
import collections
import hashlib
import io
import json
import logging
import random
import threading
import time
import traceback
import uuid
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image
from torchvision import transforms

# the app's modules live in src/engine and import each other by bare name
sys.path.insert(0, str(Path(__file__).resolve().parent / "engine"))

import env_config
import gaussian_splats_3d
import mesh_texture
import optimization as opt
import parametric_surface
import surface_of_revolution
import solid_fields
import voxel_field
from optimization import FieldType, Pattern
from shared.custom_models import differentiable
from shared.models.frozen import cap_gpu_memory
from shared.models.model_central import ModelCentral, _IMAGE_EXTENSIONS
from shared.models import model_options, vlm
from shared.models.model_types import ModelID

WEB_DIR = Path(__file__).resolve().parent / "web"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
PATTERNS_DIR = Path(__file__).resolve().parent / "patterns"
RHINO_EXT = ".3dm"
VOLUME_EXTS = (".npz", ".npy")  # dvolume fields: ASSEMBLY's groups take them besides .3dm files
FRAME_INTERVAL = 0.08      # min seconds between canvas snapshots for the page
ALL_VIEWS_INTERVAL = 0.5   # min seconds between renders of every view of a 3D field (it costs the run)
FPS_WINDOW = 30            # steps averaged for the page's steps / second estimate
VIEW_THUMB = 96            # px, augmented-view thumbnails
SURFACE_PREVIEW_MAX = 640  # px, long side of the surface preview sheet
REVOLUTION_PREVIEW_MAX = 256  # px, long side of one revolution preview tile
MESH_PREVIEW_MAX = 512     # px, long side of the mesh texture preview sheet
GAUSSIANS_3D_PREVIEW_MAX = 384  # px, long side of the 3D gaussians preview sheet
VOXELS_PREVIEW_MAX = 384   # px, long side of the voxels preview sheet
IMAGE_FITS = ["stretch", "letterbox", "crop"]  # optimization.START_IMAGE_FIT values
CANVAS_INITS = ["start", "gray", "noise"]  # optimization.CANVAS_INIT values
SYMMETRY_AXES = ["left-right", "top-bottom", "both"]  # optimization.SYMMETRY_AXIS values
EDGE_OPERATORS = ["sobel", "laplacian"]  # optimization.EDGE_OPERATOR values
DEPTH_NORMALIZATIONS = ["none", "mean", "mad"]  # optimization.DEPTH_NORMALIZATION values
DISTORTION_OFFSET_MODES = ["absolute", "resolution"]  # optimization.DISTORTION_OFFSET_MODE values
DISTORTION_EDGE_MODES = ["reflection", "border", "gray"]  # optimization.DISTORTION_EDGE_MODE values
VAE_PRECISIONS = ["fp32", "bf16"]  # optimization.VAE_PRECISION values
CONV_NORMS = ["none", "instance", "group"]  # optimization.CONV_NORM values
CONV_ACTIVATIONS = ["leaky_relu", "relu", "silu"]  # optimization.CONV_ACTIVATION values
REARRANGE_INITS = ["random", "identity"]  # optimization.REARRANGE_INIT values
PATCH_SOURCES = ["pixels", "images"]  # optimization.PATCH_SOURCE values
SHAPE_MODES = ["color", "monochrome", "masking"]  # optimization.SHAPE_MODE values

# The model choices, by family: shared.models.model_options (the CLIP explorer offers the same).
MODEL_IDS = [str(m) for m in model_options.clip_model_ids()]
MODEL_FAMILIES = model_options.MODEL_FAMILIES
CUSTOM_FAMILIES = model_options.CUSTOM_FAMILIES
SCORE_GOALS = model_options.SCORE_GOALS  # ModelSpec.goal values
CUSTOM_PRECISIONS = list(differentiable.PRECISIONS)  # ModelSpec.precision values
DINO_MODEL_IDS = [str(m) for m in model_options.dino_model_ids()]
VLM_MODEL_IDS = [str(m) for m in model_options.vlm_model_ids()]
VLM_SCORES = list(vlm.SCORES)  # ModelSpec.score values of a VLM
IMAGENET_MODEL_IDS = [str(m) for m in model_options.imagenet_model_ids()]
MATCH_TERMS = ["rgb", "gray", "depth", "edges", "color_stats", "edge_stats", "lines"]  # optimization.MATCH_TARGETS keys
IMAGENET_SCORES = model_options.IMAGENET_SCORES  # ModelSpec.score values
DINO_TOKENS = ["cls", "patch"]  # ModelSpec.tokens values
IMAGENET_CLASSES = opt.imagenet_classes()
custom_models = model_options.custom_models

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/static")


def available_devices() -> list[dict]:
    """The torch devices a run can use, best first: {id, label}."""
    out = [{"id": f"cuda:{i}", "label": f"cuda:{i} ({torch.cuda.get_device_name(i)})"}
           for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        out.append({"id": "mps", "label": "mps (Apple GPU)"})
    out.append({"id": "cpu", "label": "cpu"})
    return out


DEVICES = available_devices()
cap_gpu_memory()  # a run past the cap fails with CUDA out of memory instead of freezing the machine
DEFAULT_DEVICE = DEVICES[0]["id"]
_centrals: dict[str, ModelCentral] = {}


def resolve_device(name) -> torch.device:
    """A config's device -> torch.device; empty or not on this machine (a config saved
    elsewhere) -> the best available."""
    name = name if name in {d["id"] for d in DEVICES} else DEFAULT_DEVICE
    return torch.device(name)


def central_for(device: torch.device) -> ModelCentral:
    """The ModelCentral for a device; switching device drops the other devices' models
    (frees their memory)."""
    key = str(device)
    for other in [k for k in _centrals if k != key]:
        _centrals.pop(other).release()
    if key not in _centrals:
        _centrals[key] = ModelCentral(device, text_on_cpu=True)
    return _centrals[key]


# ── Config <-> optimization module settings ────────────────────────────────────

def _prompt_to_json(prompt):
    """Prompt / class target -> page value: 'a | b' for paraphrases, {"image": path} for an
    image, {"question", "answer"} for a VLM query, a class index as is."""
    if isinstance(prompt, opt.ImagePrompt):
        return {"image": prompt.path}
    if isinstance(prompt, opt.VLMQuery):
        return {"question": prompt.question, "answer": prompt.answer}
    return " | ".join(prompt) if isinstance(prompt, list) else prompt


def _image_prompt(value: dict) -> 'opt.ImagePrompt':
    path = Path(str(value.get("image") or ""))
    if path.suffix.lower() not in _IMAGE_EXTENSIONS or not path.is_file():
        raise ValueError(f"Target image not found: {path}")
    return opt.ImagePrompt(str(path))


def _vlm_query(value) -> 'opt.VLMQuery':
    """{"question", "answer"} -> VLMQuery"""
    if not isinstance(value, dict) or "image" in value:
        raise ValueError("A VLM target is a question and an answer")
    question, answer = str(value.get("question") or "").strip(), str(value.get("answer") or "").strip()
    if not question or not answer:
        raise ValueError(f"A VLM target needs a question and an answer: {value!r}")
    return opt.VLMQuery(question, answer)


def _prompt_from_json(text):
    """'a | b' -> ['a', 'b'] (paraphrases); a single phrase stays a string; {"image": path}
    -> ImagePrompt."""
    if isinstance(text, dict):
        return _image_prompt(text)
    parts = [p.strip() for p in str(text).split("|") if p.strip()]
    if not parts:
        raise ValueError("Empty prompt")
    return parts[0] if len(parts) == 1 else parts


# Setting overrides for the page's preset buttons. They leave the models and their
# targets, start image, pattern and field type alone (the page switches DISTORTION to PIXELS_COLOR
# for generation, which needs a field that can paint from scratch).
PRESETS = {
    # A subtle, measured edit of the start image: both poles, equal edit size, anchored.
    "editing": {
        "canvas_init": "start", "bidirectional": True, "stop_at_similarity": 0.85,
        "learning_rate": 0.002, "pyramid_lr_decay": 1.0, "epochs": 1000,
        "tv_weight": 0.05, "blur_strength": 0.0,
        "transform_count": 8, "crop_min_scale": 0.5, "rotation": 10, "perspective": 0.0, "flip": False,
        "view_margin": 0.05, "jitter": 0.2, "hue": 0.05, "noise": 0.03, "include_untransformed": True,
        "image_sim_weight": 1.0,
    },
    # Painting the text targets from noise: coarse-to-fine, hard augmentation, no image target.
    "generation": {
        "canvas_init": "noise", "canvas_noise_std": 0.4, "bidirectional": False, "stop_at_similarity": None,
        "learning_rate": 0.03, "pyramid_lr_decay": 0.5, "epochs": 400,
        "tv_weight": 0.05, "blur_strength": 0.0,
        "transform_count": 32, "crop_min_scale": 0.1, "rotation": 20, "perspective": 0.3, "flip": False,
        "view_margin": 0.05, "jitter": 0.2, "hue": 0.05, "noise": 0.1, "include_untransformed": False,
        "image_sim_weight": 0.0, "image_gray_weight": 0.0, "image_depth_weight": 0.0, "image_edge_weight": 0.0,
        "image_color_stats_weight": 0.0, "image_edge_stats_weight": 0.0, "image_lines_weight": 0.0,
    },
}


_custom_family = model_options.custom_family
_family = model_options.family_of


def _spec_to_json(spec: 'opt.ModelSpec') -> dict:
    """A ModelSpec -> the page's model entry: {family, model (a ModelID string; a custom
    model's id), targets, directions, and the family's options}."""
    custom = spec.model_id.is_custom()
    return {
        "family": _family(spec.model_id),
        "model": spec.model_id.arch if custom else str(spec.model_id),
        "targets": [[_prompt_to_json(t), w] for t, w in spec.targets],
        "directions": [[_prompt_to_json(src), _prompt_to_json(t), w] for src, t, w in spec.directions],
        **{key: getattr(spec, key) for key in ("score", "tokens", "prompt", "goal", "value", "weight", "precision",
                                               "use_templates")},
        "templates": list(spec.templates or opt.TEXT_TEMPLATES),
    }


def default_config() -> dict:
    """The settings at the top of optimization.py, as the page's config object."""
    return {
        "field_type": opt.FIELD_TYPE.name,
        "models": [_spec_to_json(spec) for spec in opt.MODELS],
        "canvas": list(opt.CANVAS_RESOLUTION),
        "start_image": opt.START_IMAGE or "",
        "start_image_fit": opt.START_IMAGE_FIT,
        "canvas_init": opt.CANVAS_INIT,
        "pixel_src": list(opt.PIXEL_SRC_RESOLUTION),
        "pyramid_levels": opt.PIXEL_PYRAMID_LEVELS,
        "pyramid_lr_decay": opt.PIXEL_PYRAMID_LR_DECAY,
        "canvas_noise_std": opt.CANVAS_NOISE_STD,
        "pixel_contrast": opt.PIXEL_CONTRAST,
        "pixel_coverage": opt.PIXEL_COVERAGE,
        "pixel_coverage_weight": opt.PIXEL_COVERAGE_WEIGHT,
        "levelset": opt.normalize_levelset(opt.LEVELSET),
        "function_stack": opt.normalize_function_stack(opt.FUNCTION_STACK),
        "revolution": surface_of_revolution.normalize_config(opt.REVOLUTION),
        "mesh_texture": mesh_texture.normalize_config(opt.MESH_TEXTURE),
        "gaussians_3d": gaussian_splats_3d.normalize_config(opt.GAUSSIANS_3D),
        "voxels": voxel_field.normalize_config(opt.VOXELS),
        "assembly": solid_fields.normalize_assembly(opt.ASSEMBLY),
        "solids_on_off": solid_fields.normalize_on_off(opt.SOLIDS_ON_OFF),
        "extrusions": solid_fields.normalize_extrusions(opt.EXTRUSIONS),
        "distortion_res": list(opt.DISTORTION_FIELD_RESOLUTION),
        "distortion_levels": opt.DISTORTION_PYRAMID_LEVELS,
        "distortion_lr_decay": opt.DISTORTION_PYRAMID_LR_DECAY,
        "distortion_offset_mode": opt.DISTORTION_OFFSET_MODE,
        "distortion_max": opt.DISTORTION_MAX,
        "distortion_edge_mode": opt.DISTORTION_EDGE_MODE,
        "distortion_prevent_folds": opt.DISTORTION_PREVENT_FOLDS,
        "distortion_fold_weight": opt.DISTORTION_FOLD_WEIGHT,
        "shape_src": opt.SHAPE_SRC,
        "shape_border": opt.SHAPE_BORDER,
        "shape_mode": opt.SHAPE_MODE,
        "shape_color_1": _hex(opt.SHAPE_COLOR_1),
        "shape_color_1_optimize": opt.SHAPE_COLOR_1_OPTIMIZE,
        "shape_color_2": _hex(opt.SHAPE_COLOR_2),
        "shape_color_2_optimize": opt.SHAPE_COLOR_2_OPTIMIZE,
        "shape_opacity_budget": opt.SHAPE_OPACITY_BUDGET,
        "shape_opacity_weight": opt.SHAPE_OPACITY_WEIGHT,
        "shape_harden_steps": opt.SHAPE_HARDEN_STEPS,
        "shape_harden_weight": opt.SHAPE_HARDEN_WEIGHT,
        "relief_color": _hex(opt.RELIEF_COLOR),
        "relief_color_optimize": opt.RELIEF_COLOR_OPTIMIZE,
        "relief_depth": opt.RELIEF_DEPTH,
        "relief_smooth": opt.RELIEF_SMOOTH,
        "light_count": opt.LIGHT_COUNT,
        "light_azimuth": opt.LIGHT_AZIMUTH,
        "light_elevation": opt.LIGHT_ELEVATION,
        "light_random_count": opt.LIGHT_RANDOM_COUNT,
        "light_random_elevation": list(opt.LIGHT_RANDOM_ELEVATION),
        "light_ambient": opt.LIGHT_AMBIENT,
        "patch_source": opt.PATCH_SOURCE,
        "patch_images": [dict(e) for e in opt.PATCH_IMAGES],
        "patch_image_resolution": opt.PATCH_IMAGE_RESOLUTION,
        "patch_count": opt.PATCH_COUNT,
        "patch_resolution": opt.PATCH_RESOLUTION,
        "patch_identical": opt.PATCH_IDENTICAL,
        "patch_rotation": opt.PATCH_ROTATION,
        "patch_scale_range": list(opt.PATCH_SCALE_RANGE),
        "patch_monochrome": opt.PATCH_MONOCHROME,
        "patch_opaque": opt.PATCH_OPAQUE,
        "patch_transform_lr": opt.PATCH_TRANSFORM_LR,
        "patch_randomize_transforms": opt.PATCH_RANDOMIZE_TRANSFORMS,
        "gaussian_count": opt.GAUSSIAN_COUNT,
        "gaussian_negative_count": opt.GAUSSIAN_NEGATIVE_COUNT,
        "gaussian_max_size": opt.GAUSSIAN_MAX_SIZE,
        "gaussian_monochrome": opt.GAUSSIAN_MONOCHROME,
        "gaussian_spread": opt.GAUSSIAN_SPREAD,
        "gaussian_contrast": opt.GAUSSIAN_CONTRAST,
        "gaussian_fg_color": _hex(opt.GAUSSIAN_FG_COLOR),
        "gaussian_fg_optimize": opt.GAUSSIAN_FG_OPTIMIZE,
        "gaussian_rectilinear": opt.GAUSSIAN_RECTILINEAR,
        "gaussian_roundness_init": opt.GAUSSIAN_ROUNDNESS_INIT,
        "gaussian_axis_aligned": opt.GAUSSIAN_AXIS_ALIGNED,
        "background_color": _hex(opt.BACKGROUND_COLOR),
        "background_optimize": opt.BACKGROUND_OPTIMIZE,
        "background_image": opt.BACKGROUND_IMAGE or "",
        "background_image_fit": opt.BACKGROUND_IMAGE_FIT,
        "background_image_multiply": opt.BACKGROUND_IMAGE_MULTIPLY,
        "surface": parametric_surface.normalize_config(opt.SURFACE),
        "vae_model": opt.VAE_MODEL,
        "vae_precision": opt.VAE_PRECISION,
        "vae_decode_scale": opt.VAE_DECODE_SCALE,
        "vae_latent_bound": opt.VAE_LATENT_BOUND,
        "vae_latent_weight": opt.VAE_LATENT_WEIGHT,
        "conv_inplace": opt.CONV_INPLACE,
        "conv_input_res": list(opt.CONV_INPUT_RESOLUTION),
        "conv_layers": opt.CONV_LAYERS,
        "conv_channels": opt.CONV_CHANNELS,
        "conv_kernel": opt.CONV_KERNEL,
        "conv_fit_steps": opt.CONV_FIT_STEPS,
        "conv_norm": opt.CONV_NORM,
        "conv_activation": opt.CONV_ACTIVATION,
        "conv_residual_max": opt.CONV_RESIDUAL_MAX,
        "conv_weight_lr": opt.CONV_WEIGHT_LR,
        "rearrange_grid": list(opt.REARRANGE_GRID),
        "rearrange_permutation": opt.REARRANGE_PERMUTATION,
        "rearrange_init": opt.REARRANGE_INIT,
        "rearrange_init_std": opt.REARRANGE_INIT_STD,
        "rearrange_temp_start": opt.REARRANGE_TEMP_START,
        "rearrange_temp_end": opt.REARRANGE_TEMP_END,
        "rearrange_entropy_weight": opt.REARRANGE_ENTROPY_WEIGHT,
        "rearrange_hard_after": opt.REARRANGE_HARD_AFTER,
        "rearrange_lr": opt.REARRANGE_LR,
        "image_sim_weight": opt.IMAGE_SIM_WEIGHT if opt.IMAGE_SIM_TARGET is not None else 0.0,
        "match_targets": dict(opt.MATCH_TARGETS),
        "image_gray_weight": opt.IMAGE_GRAY_WEIGHT,
        "image_depth_weight": opt.IMAGE_DEPTH_WEIGHT,
        "image_edge_weight": opt.IMAGE_EDGE_WEIGHT,
        "image_color_stats_weight": opt.IMAGE_COLOR_STATS_WEIGHT,
        "image_edge_stats_weight": opt.IMAGE_EDGE_STATS_WEIGHT,
        "edge_stats_scales": opt.EDGE_STATS_SCALES,
        "image_lines_weight": opt.IMAGE_LINES_WEIGHT,
        "lines_blur": opt.LINES_BLUR,
        "edge_operator": opt.EDGE_OPERATOR,
        "depth_augment_views": opt.DEPTH_AUGMENT_VIEWS,
        "depth_normalization": opt.DEPTH_NORMALIZATION,
        "bidirectional": opt.BIDIRECTIONAL,
        "sequential_targets": opt.SEQUENTIAL_TARGETS,
        "sequential_steps": opt.SEQUENTIAL_STEPS,
        "sequential_cycles": opt.SEQUENTIAL_CYCLES,
        "stop_at_similarity": opt.STOP_AT_SIMILARITY,
        "diff_gain": opt.DIFF_GAIN,
        "tv_weight": opt.TV_WEIGHT,
        "symmetry_weight": opt.SYMMETRY_WEIGHT,
        "symmetry_axis": opt.SYMMETRY_AXIS,
        "blur_strength": opt.BLUR_STRENGTH,
        "blur_sigma": opt.BLUR_SIGMA,
        "median_weight": opt.MEDIAN_WEIGHT,
        "median_kernel": opt.MEDIAN_KERNEL,
        "transform_count": opt.RANDOM_TRANSFORM_COUNT,
        "crop_min_scale": opt.CROP_SCALE[0],
        "rotation": opt.ROTATION,
        "perspective": opt.PERSPECTIVE,
        "flip": opt.FLIP,
        "view_margin": opt.VIEW_MARGIN,
        "jitter": 0.2,
        "hue": 0.05,
        "noise": opt.TRANSFORM_NOISE,
        "include_untransformed": opt.INCLUDE_UNTRANSFORMED,
        "learning_rate": opt.LEARNING_RATE,
        "epochs": opt.EPOCHS,
        "seed": opt.SEED,
        "device": DEFAULT_DEVICE,
    }


def _hex(rgb) -> str:
    return "#" + "".join(f"{round(float(v) * 255):02x}" for v in rgb)


def _rgb(value) -> tuple[float, float, float]:
    """'#rrggbb' -> RGB in [0, 1]."""
    s = str(value or "").strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Bad colour: {value!r}")
    return tuple(int(s[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _size(value) -> tuple[int, int]:
    w, h = (int(v) for v in value)
    if not (8 <= w <= 4096 and 8 <= h <= 4096):
        raise ValueError(f"Bad size {w}x{h}")
    return w, h


def _choice(entry: dict, key: str, choices: list[str], default: str) -> str:
    value = entry.get(key) or default
    if value not in choices:
        raise ValueError(f"Bad {key}: {value!r} (one of {choices})")
    return value


def _templates(entry: dict) -> list[str] | None:
    """A CLIP entry's templates (blank lines dropped; none -> the default, None)."""
    templates = [t for t in (entry.get("templates") or []) if t.strip()]
    for t in templates:
        if t.count("{}") != 1:
            raise ValueError(f"Template needs exactly one {{}}: {t!r}")
    return templates or None


def _spec_from_json(entry: dict, custom: dict[str, dict]) -> 'opt.ModelSpec':
    """A page model entry (see _spec_to_json) -> ModelSpec, validated. `custom`: the
    custom models by id (custom_models())."""
    family = entry.get("family")
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Bad model family: {family!r}")
    kind, classes = family, IMAGENET_CLASSES
    if family in CUSTOM_FAMILIES:
        info = custom.get(entry.get("model"))
        if info is None:
            raise ValueError(f"No {family} model {entry.get('model')!r} "
                             f"(models folder: {differentiable.env_config.ROOT_MODEL_FOLDER})")
        if _custom_family(info["kind"]) != family:
            raise ValueError(f"Not a {family} model: {entry.get('model')!r}")
        kind, classes = info["kind"], info.get("classes")
        model_id = ModelID.custom(info["id"])
    else:
        model_id = ModelID.from_string(str(entry.get("model") or ""))
        if _family(model_id) != family:
            raise ValueError(f"Not a {family} model: {entry.get('model')!r}")
    name = model_id.arch

    def class_index(c):
        """A class index, or {"image": path} -> ImagePrompt."""
        if isinstance(c, dict):
            return _image_prompt(c)
        c = int(c)
        if not 0 <= c < len(classes):
            raise ValueError(f"{name}: bad class index {c}")
        return c

    # what a target end is: a prompt (CLIP), a class (classifiers), an image (DINOv2, autoencoders),
    # a question and answer (VLMs)
    parse = {"clip": _prompt_from_json, "imagenet": class_index, "classifier": class_index,
             "dino": _image_prompt, "autoencoder": _image_prompt, "vlm": _vlm_query}.get(kind)
    targets, directions = entry.get("targets") or [], entry.get("directions") or []
    if parse is None:   # scores: no targets
        targets, directions = [], []
    if parse is _image_prompt and not all(isinstance(e, dict) for t in [*targets, *directions] for e in t[:-1]):
        raise ValueError(f"{name}: no text encoder -- targets must be images")
    spec = opt.ModelSpec(
        model_id,
        targets=[(parse(t), float(w)) for t, w in targets],
        directions=[(parse(src), parse(t), float(w)) for src, t, w in directions],
        score=(_choice(entry, "score", VLM_SCORES, "logprob") if family == "vlm"
               else _choice(entry, "score", IMAGENET_SCORES, "logits")),
        tokens=_choice(entry, "tokens", DINO_TOKENS, "cls"),
        prompt=(entry.get("prompt") or "").strip() or "an image",
        goal=_choice(entry, "goal", SCORE_GOALS, "maximize"),
        value=float(entry.get("value") or 0),
        weight=float(entry.get("weight") if entry.get("weight") not in (None, "") else 1.0),
        precision=_choice(entry, "precision", CUSTOM_PRECISIONS, "bf16" if family == "vlm" else "fp32"),
        use_templates=bool(entry.get("use_templates", True)),
        templates=_templates(entry) if family == "clip" else None)
    if parse is class_index and any(isinstance(src, opt.ImagePrompt) != isinstance(t, opt.ImagePrompt)
                                    for src, t, _ in spec.directions):
        raise ValueError(f"{name}: a class direction joins two classes or two images, not one of each")
    return spec


def apply_config(cfg: dict) -> None:
    """Write a page config into optimization's module settings (validated first, so a
    bad config leaves the previous settings untouched)."""
    field_type = FieldType[cfg["field_type"]]
    entries = cfg.get("models") or []
    custom = {m["id"]: m for m in custom_models()} if any(e.get("family") in CUSTOM_FAMILIES for e in entries) else {}
    models = [_spec_from_json(entry, custom) for entry in entries]
    canvas = _size(cfg["canvas"])
    start_image = (cfg.get("start_image") or "").strip() or None
    if start_image is not None and not Path(start_image).is_file():
        raise ValueError(f"Start image not found: {start_image}")
    depth_normalization = cfg.get("depth_normalization") or "mean"
    if depth_normalization not in DEPTH_NORMALIZATIONS:
        raise ValueError(f"Bad depth normalization: {depth_normalization!r}")
    edge_operator = cfg.get("edge_operator") or "sobel"
    if edge_operator not in EDGE_OPERATORS:
        raise ValueError(f"Bad edge operator: {edge_operator!r}")
    symmetry_axis = cfg.get("symmetry_axis") or "left-right"
    if symmetry_axis not in SYMMETRY_AXES:
        raise ValueError(f"Bad symmetry axis: {symmetry_axis!r}")
    canvas_init = cfg.get("canvas_init") or "start"
    if canvas_init not in CANVAS_INITS:
        raise ValueError(f"Bad canvas init: {canvas_init!r}")
    start_image_fit = cfg.get("start_image_fit", "stretch")
    if start_image_fit not in IMAGE_FITS:
        raise ValueError(f"Bad start image fit: {start_image_fit!r}")
    shape_src = (cfg.get("shape_src") or "").strip()
    if field_type in (FieldType.SHAPES, FieldType.RELIEF) and not Path(shape_src).is_file():
        raise ValueError(f"Pattern not found: {shape_src}")
    if field_type in (FieldType.DISTORTION, FieldType.REARRANGE) and start_image is None:
        raise ValueError(f"{field_type.name} needs a start image")
    offset_mode = cfg.get("distortion_offset_mode") or "absolute"
    if offset_mode not in DISTORTION_OFFSET_MODES:
        raise ValueError(f"Bad distortion offset mode: {offset_mode!r}")
    edge_mode = cfg.get("distortion_edge_mode") or "reflection"
    if edge_mode not in DISTORTION_EDGE_MODES:
        raise ValueError(f"Bad distortion edge mode: {edge_mode!r}")
    d = default_config()
    vae_model = _choice(cfg, "vae_model", list(opt.VAE_MODELS), d["vae_model"])
    vae_precision = _choice(cfg, "vae_precision", VAE_PRECISIONS, d["vae_precision"])
    conv_norm = _choice(cfg, "conv_norm", CONV_NORMS, d["conv_norm"])
    conv_activation = _choice(cfg, "conv_activation", CONV_ACTIVATIONS, d["conv_activation"])
    rearrange_init = _choice(cfg, "rearrange_init", REARRANGE_INITS, d["rearrange_init"])
    patch_source = _choice(cfg, "patch_source", PATCH_SOURCES, d["patch_source"])
    shape_mode = _choice(cfg, "shape_mode", SHAPE_MODES, d["shape_mode"])
    shape_color_1 = _rgb(cfg.get("shape_color_1") or d["shape_color_1"])
    shape_color_2 = _rgb(cfg.get("shape_color_2") or d["shape_color_2"])
    opacity_budget = cfg.get("shape_opacity_budget")
    opacity_budget = None if opacity_budget in (None, "") else min(100.0, max(0.0, float(opacity_budget)))
    patch_images = [{"path": str(e.get("path") or "").strip(), "count": max(0, int(e.get("count") or 0))}
                    for e in cfg.get("patch_images") or []]
    if field_type == FieldType.PIXEL_PATCHES and patch_source == "images":
        for e in patch_images:
            if not Path(e["path"]).is_file():
                raise ValueError(f"Patch image not found: {e['path']}")
        if not any(e["count"] for e in patch_images):
            raise ValueError("Patches from images: add an image (with a count above 0)")
    lo, hi = sorted(float(v) for v in cfg.get("patch_scale_range") or d["patch_scale_range"])
    if not 0 < lo <= hi:
        raise ValueError(f"Bad patch scale range: {lo}..{hi}")
    background = _rgb(cfg.get("background_color") or d["background_color"])
    background_image = (cfg.get("background_image") or "").strip() or None
    if field_type in opt.BACKGROUND_IMAGE_FIELDS and background_image is not None and not Path(background_image).is_file():
        raise ValueError(f"Background image not found: {background_image}")
    background_image_fit = _choice(cfg, "background_image_fit", IMAGE_FITS, d["background_image_fit"])
    fg_color =_rgb(cfg.get("gaussian_fg_color") or d["gaussian_fg_color"])
    relief_color = _rgb(cfg.get("relief_color") or d["relief_color"])
    light_elevations = sorted(min(90.0, max(0.0, float(v)))
                              for v in cfg.get("light_random_elevation") or d["light_random_elevation"])
    surface = parametric_surface.normalize_config(cfg.get("surface"))
    levelset = opt.normalize_levelset(cfg.get("levelset"))
    function_stack = opt.normalize_function_stack(cfg.get("function_stack"))
    # Another field's run isn't stopped by a formula left half typed in this one.
    revolution = (surface_of_revolution.normalize_config(cfg.get("revolution")) if field_type == FieldType.REVOLUTION
                  else d["revolution"])
    mesh = (mesh_texture.normalize_config(cfg.get("mesh_texture")) if field_type == FieldType.MESH_TEXTURE
            else d["mesh_texture"])
    if field_type == FieldType.MESH_TEXTURE and not Path(mesh["path"]).is_file():
        raise ValueError(f"Mesh not found: {mesh['path'] or '(no .3dm chosen)'}")
    gaussians_3d = gaussian_splats_3d.normalize_config(cfg.get("gaussians_3d"))
    voxels = voxel_field.normalize_config(cfg.get("voxels"))
    solids = {kind: solid_fields.NORMALIZERS[kind](cfg.get(kind.lower())) for kind in solid_fields.NORMALIZERS}
    if field_type == FieldType.VOXELS and voxels["init"] == "mesh" and not Path(voxels["init_mesh"]).is_file():
        raise ValueError(f"Voxel init mesh not found: {voxels['init_mesh'] or '(no .3dm chosen)'}")
    if field_type.name in solids:
        solid_fields.validate(field_type.name, solids[field_type.name])
    coverage = cfg.get("pixel_coverage")
    coverage = None if coverage in (None, "") else min(1.0, max(0.0, float(coverage)))
    stop = cfg.get("stop_at_similarity")
    jitter, hue = float(cfg["jitter"]), float(cfg["hue"])

    opt.FIELD_TYPE = field_type
    opt.MODELS = models
    opt.CANVAS_RESOLUTION = canvas
    opt.START_IMAGE = start_image
    opt.START_IMAGE_FIT = start_image_fit
    opt.CANVAS_INIT = canvas_init
    opt.PIXEL_SRC_RESOLUTION = _size(cfg["pixel_src"])
    opt.PIXEL_PYRAMID_LEVELS = max(1, int(cfg["pyramid_levels"]))
    opt.PIXEL_PYRAMID_LR_DECAY = float(cfg.get("pyramid_lr_decay") or 1.0)
    noise_std = cfg.get("canvas_noise_std")
    opt.CANVAS_NOISE_STD = 0.4 if noise_std in (None, "") else max(0.0, float(noise_std))
    opt.PIXEL_CONTRAST = abs(float(cfg.get("pixel_contrast") or 0))
    opt.PIXEL_COVERAGE = coverage
    weight = cfg.get("pixel_coverage_weight")
    opt.PIXEL_COVERAGE_WEIGHT = 10.0 if weight in (None, "") else max(0.0, float(weight))
    opt.LEVELSET = levelset
    opt.FUNCTION_STACK = function_stack
    opt.REVOLUTION = revolution
    opt.MESH_TEXTURE = mesh
    opt.GAUSSIANS_3D = gaussians_3d
    opt.VOXELS = voxels
    opt.ASSEMBLY, opt.SOLIDS_ON_OFF, opt.EXTRUSIONS = (solids[k] for k in ("ASSEMBLY", "SOLIDS_ON_OFF", "EXTRUSIONS"))
    opt.DISTORTION_FIELD_RESOLUTION = tuple(max(2, int(v)) for v in cfg["distortion_res"])
    opt.DISTORTION_PYRAMID_LEVELS = max(1, int(cfg.get("distortion_levels") or 1))
    opt.DISTORTION_PYRAMID_LR_DECAY = float(cfg.get("distortion_lr_decay") or 1.0)
    opt.DISTORTION_OFFSET_MODE = offset_mode
    opt.DISTORTION_MAX = max(0.0, float(cfg.get("distortion_max") or 0))
    opt.DISTORTION_EDGE_MODE = edge_mode
    opt.DISTORTION_PREVENT_FOLDS = bool(cfg.get("distortion_prevent_folds", True))
    opt.DISTORTION_FOLD_WEIGHT = float(cfg.get("distortion_fold_weight") or 0)
    opt.SHAPE_SRC = shape_src
    opt.SHAPE_BORDER = int(cfg["shape_border"])
    opt.SHAPE_MODE = shape_mode
    opt.SHAPE_COLOR_1 = shape_color_1
    opt.SHAPE_COLOR_1_OPTIMIZE = bool(cfg.get("shape_color_1_optimize", d["shape_color_1_optimize"]))
    opt.SHAPE_COLOR_2 = shape_color_2
    opt.SHAPE_COLOR_2_OPTIMIZE = bool(cfg.get("shape_color_2_optimize", d["shape_color_2_optimize"]))
    opt.SHAPE_OPACITY_BUDGET = opacity_budget
    opt.PATCH_SOURCE = patch_source
    opt.PATCH_IMAGES = patch_images
    opt.PATCH_IMAGE_RESOLUTION = min(1024, max(8, int(cfg.get("patch_image_resolution") or d["patch_image_resolution"])))
    opt.PATCH_COUNT = max(1, int(cfg.get("patch_count") or d["patch_count"]))
    opt.PATCH_RESOLUTION = max(2, int(cfg.get("patch_resolution") or d["patch_resolution"]))
    opt.PATCH_IDENTICAL = bool(cfg.get("patch_identical", d["patch_identical"]))
    opt.PATCH_ROTATION = bool(cfg.get("patch_rotation", d["patch_rotation"]))
    opt.PATCH_SCALE_RANGE = (lo, hi)
    opt.PATCH_MONOCHROME = bool(cfg.get("patch_monochrome", d["patch_monochrome"]))
    opt.PATCH_OPAQUE = bool(cfg.get("patch_opaque", d["patch_opaque"]))
    opt.PATCH_TRANSFORM_LR = float(cfg.get("patch_transform_lr") or 1.0)
    opt.PATCH_RANDOMIZE_TRANSFORMS = bool(cfg.get("patch_randomize_transforms", d["patch_randomize_transforms"]))
    opt.GAUSSIAN_COUNT = max(1, int(cfg.get("gaussian_count") or d["gaussian_count"]))
    opt.GAUSSIAN_NEGATIVE_COUNT = max(0, int(cfg.get("gaussian_negative_count") or 0))
    opt.GAUSSIAN_MAX_SIZE = max(1e-3, float(cfg.get("gaussian_max_size") or d["gaussian_max_size"]))
    opt.GAUSSIAN_MONOCHROME = bool(cfg.get("gaussian_monochrome", d["gaussian_monochrome"]))
    spread = cfg.get("gaussian_spread")
    opt.GAUSSIAN_SPREAD = min(1.0, max(0.0, float(d["gaussian_spread"] if spread in (None, "") else spread)))
    opt.GAUSSIAN_CONTRAST = max(0.0, float(cfg.get("gaussian_contrast") or 0))
    opt.GAUSSIAN_FG_COLOR = fg_color
    opt.GAUSSIAN_FG_OPTIMIZE = bool(cfg.get("gaussian_fg_optimize", d["gaussian_fg_optimize"]))
    opt.GAUSSIAN_RECTILINEAR = bool(cfg.get("gaussian_rectilinear", d["gaussian_rectilinear"]))
    roundness = cfg.get("gaussian_roundness_init")
    opt.GAUSSIAN_ROUNDNESS_INIT = min(0.99, max(0.01, float(d["gaussian_roundness_init"] if roundness in (None, "") else roundness)))
    opt.GAUSSIAN_AXIS_ALIGNED = bool(cfg.get("gaussian_axis_aligned", d["gaussian_axis_aligned"]))
    opt.BACKGROUND_COLOR = background
    opt.BACKGROUND_OPTIMIZE = bool(cfg.get("background_optimize", d["background_optimize"]))
    opt.BACKGROUND_IMAGE = background_image
    opt.BACKGROUND_IMAGE_FIT = background_image_fit
    opt.BACKGROUND_IMAGE_MULTIPLY = bool(cfg.get("background_image_multiply", d["background_image_multiply"]))
    opt.SURFACE = surface
    opt.VAE_MODEL = vae_model
    opt.VAE_PRECISION = vae_precision
    opt.VAE_DECODE_SCALE = min(4.0, max(0.1, float(cfg.get("vae_decode_scale") or 1.0)))
    bound = cfg.get("vae_latent_bound")
    opt.VAE_LATENT_BOUND = d["vae_latent_bound"] if bound in (None, "") else max(0.1, float(bound))
    opt.VAE_LATENT_WEIGHT = max(0.0, float(cfg.get("vae_latent_weight") or 0))
    opt.CONV_INPLACE = bool(cfg.get("conv_inplace", d["conv_inplace"]))
    opt.CONV_INPUT_RESOLUTION = tuple(min(256, max(1, int(v))) for v in cfg.get("conv_input_res") or d["conv_input_res"])
    opt.CONV_LAYERS = min(10, max(1, int(cfg.get("conv_layers") or d["conv_layers"])))
    opt.CONV_CHANNELS = max(1, int(cfg.get("conv_channels") or d["conv_channels"]))
    opt.CONV_KERNEL = min(15, max(1, int(cfg.get("conv_kernel") or d["conv_kernel"])))
    fit_steps = cfg.get("conv_fit_steps")
    opt.CONV_FIT_STEPS = d["conv_fit_steps"] if fit_steps in (None, "") else max(0, int(fit_steps))
    opt.CONV_NORM = conv_norm
    opt.CONV_ACTIVATION = conv_activation
    opt.CONV_RESIDUAL_MAX = max(0.0, float(cfg.get("conv_residual_max") or 0))
    weight_lr = cfg.get("conv_weight_lr")
    opt.CONV_WEIGHT_LR = d["conv_weight_lr"] if weight_lr in (None, "") else max(0.0, float(weight_lr))

    def number(key, lo, hi=float("inf")):
        v = cfg.get(key)
        return min(hi, max(lo, float(d[key] if v in (None, "") else v)))
    opt.SHAPE_OPACITY_WEIGHT = number("shape_opacity_weight", 0.0)
    opt.SHAPE_HARDEN_STEPS = int(number("shape_harden_steps", 0))
    opt.SHAPE_HARDEN_WEIGHT = number("shape_harden_weight", 0.0)
    opt.REARRANGE_GRID = tuple(min(64, max(1, int(v))) for v in cfg.get("rearrange_grid") or d["rearrange_grid"])
    opt.REARRANGE_PERMUTATION = bool(cfg.get("rearrange_permutation", d["rearrange_permutation"]))
    opt.REARRANGE_INIT = rearrange_init
    opt.REARRANGE_INIT_STD = number("rearrange_init_std", 0.0)
    opt.REARRANGE_TEMP_START = number("rearrange_temp_start", 1e-3)
    opt.REARRANGE_TEMP_END = number("rearrange_temp_end", 1e-3)
    opt.REARRANGE_ENTROPY_WEIGHT = number("rearrange_entropy_weight", 0.0)
    opt.REARRANGE_HARD_AFTER = number("rearrange_hard_after", 0.0, 1.0)
    opt.REARRANGE_LR = number("rearrange_lr", 0.0)
    opt.RELIEF_COLOR = relief_color
    opt.RELIEF_COLOR_OPTIMIZE = bool(cfg.get("relief_color_optimize", d["relief_color_optimize"]))
    opt.RELIEF_DEPTH = number("relief_depth", 0.0)
    opt.RELIEF_SMOOTH = number("relief_smooth", 0.0)
    opt.LIGHT_COUNT = int(number("light_count", 1, 16))
    opt.LIGHT_AZIMUTH = number("light_azimuth", -360.0, 360.0)
    opt.LIGHT_ELEVATION = number("light_elevation", 0.0, 90.0)
    opt.LIGHT_RANDOM_COUNT = int(number("light_random_count", 0, 16))
    opt.LIGHT_RANDOM_ELEVATION = tuple(light_elevations)
    opt.LIGHT_AMBIENT = number("light_ambient", 0.0, 1.0)
    match_weights = [float(cfg.get(k) or 0) for k in
                     ("image_sim_weight", "image_gray_weight", "image_depth_weight", "image_edge_weight",
                      "image_color_stats_weight", "image_edge_stats_weight", "image_lines_weight")]
    match_targets = {}
    for term, path in (cfg.get("match_targets") or {}).items():
        if term not in MATCH_TERMS:
            raise ValueError(f"Bad match term: {term!r}")
        if path:
            match_targets[term] = _image_prompt({"image": path}).path
    # Every term compares against its override, else the start image.
    opt.IMAGE_SIM_TARGET = start_image
    opt.MATCH_TARGETS = match_targets
    (opt.IMAGE_SIM_WEIGHT, opt.IMAGE_GRAY_WEIGHT, opt.IMAGE_DEPTH_WEIGHT, opt.IMAGE_EDGE_WEIGHT,
     opt.IMAGE_COLOR_STATS_WEIGHT, opt.IMAGE_EDGE_STATS_WEIGHT, opt.IMAGE_LINES_WEIGHT) = match_weights
    blur = cfg.get("lines_blur")
    opt.LINES_BLUR = opt.LINES_BLUR if blur in (None, "") else max(0.0, float(blur))
    opt.EDGE_STATS_SCALES = max(1, int(cfg.get("edge_stats_scales") or 1))
    opt.EDGE_OPERATOR = edge_operator
    opt.DEPTH_AUGMENT_VIEWS = max(0, int(cfg.get("depth_augment_views") or 0))
    opt.DEPTH_NORMALIZATION = depth_normalization
    opt.BIDIRECTIONAL = bool(cfg["bidirectional"])
    opt.SEQUENTIAL_TARGETS = bool(cfg.get("sequential_targets", False))
    opt.SEQUENTIAL_STEPS = max(1, int(cfg.get("sequential_steps") or d["sequential_steps"]))
    opt.SEQUENTIAL_CYCLES = max(1, int(cfg.get("sequential_cycles") or d["sequential_cycles"]))
    opt.STOP_AT_SIMILARITY = None if stop in (None, "") else float(stop)
    opt.DIFF_GAIN = float(cfg["diff_gain"])
    opt.TV_WEIGHT = float(cfg["tv_weight"])
    opt.SYMMETRY_WEIGHT = float(cfg.get("symmetry_weight") or 0)
    opt.SYMMETRY_AXIS = symmetry_axis
    opt.BLUR_STRENGTH = float(cfg["blur_strength"])
    opt.BLUR_SIGMA = float(cfg["blur_sigma"])
    opt.MEDIAN_WEIGHT = min(1.0, max(0.0, float(cfg.get("median_weight") or 0)))
    kernel = max(3, int(cfg.get("median_kernel") or 3))
    opt.MEDIAN_KERNEL = kernel if kernel % 2 else kernel + 1   # odd: a centered window
    opt.RANDOM_TRANSFORM_COUNT = max(0, int(cfg["transform_count"]))
    opt.CROP_SCALE = (float(cfg["crop_min_scale"]), 1.0)
    opt.ROTATION = float(cfg["rotation"])
    opt.PERSPECTIVE = min(1.0, max(0.0, float(cfg.get("perspective") or 0)))
    opt.FLIP = bool(cfg["flip"])
    margin = cfg.get("view_margin")
    opt.VIEW_MARGIN = 0.05 if margin in (None, "") else max(0.0, float(margin))
    opt.COLOR_JITTER = transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=hue)
    opt.TRANSFORM_NOISE = float(cfg["noise"])
    opt.INCLUDE_UNTRANSFORMED = bool(cfg["include_untransformed"]) or opt.RANDOM_TRANSFORM_COUNT == 0
    opt.LEARNING_RATE = float(cfg["learning_rate"])
    opt.EPOCHS = max(1, int(cfg["epochs"]))
    opt.SEED = int(cfg["seed"])


# ── Config files ───────────────────────────────────────────────────────────────
# A saved config: {"schema", "saved", "config": the page config, "files": {path: {name,
# sha1, data (base64)}}} -- the files the config refers to (start image, target and match
# images, the .3dm pattern), embedded unless saved as paths only. A plain config object
# (no "config" key) loads too.

CONFIG_SCHEMA = "optimization.config/1"
_FILE_SUFFIXES = {*_IMAGE_EXTENSIONS, RHINO_EXT, *VOLUME_EXTS}


def _strings(value):
    """Every string in a nested config."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _replace_strings(value, mapping: dict[str, str]):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, dict):
        return {k: _replace_strings(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_strings(v, mapping) for v in value]
    return value


def _file_like(text: str) -> bool:
    return Path(text).suffix.lower() in _FILE_SUFFIXES


def pack_config(cfg: dict, embed: bool) -> dict:
    """A page config as a config file; embed: with the files it refers to, base64 encoded."""
    files = {}
    for path in sorted({t for t in _strings(cfg) if _file_like(t) and Path(t).is_file()} if embed else ()):
        data = Path(path).read_bytes()
        files[path] = {"name": Path(path).name, "sha1": hashlib.sha1(data).hexdigest(),
                       "data": base64.b64encode(data).decode("ascii")}
    return {"schema": CONFIG_SCHEMA, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg, "files": files}


def unpack_config(doc: dict) -> tuple[dict, list[str]]:
    """A config file -> (the page config, notes). An embedded file whose original path still
    holds the same bytes keeps that path; otherwise it is written to the uploads folder
    (as <sha1>_<name>) and the config points there. Notes name the referenced files that
    are missing."""
    if not isinstance(doc, dict):
        raise ValueError("Not a config file")
    cfg, files = (doc["config"], doc.get("files") or {}) if "config" in doc else (doc, {})
    if not isinstance(cfg, dict):
        raise ValueError("Not a config file")
    mapping = {}
    for original, entry in files.items():
        name = Path(str(entry.get("name") or original)).name
        if not _file_like(name):
            raise ValueError(f"Unsupported embedded file: {name!r}")
        data = base64.b64decode(entry["data"])
        sha = hashlib.sha1(data).hexdigest()
        if Path(original).is_file() and hashlib.sha1(Path(original).read_bytes()).hexdigest() == sha:
            continue
        UPLOAD_DIR.mkdir(exist_ok=True)
        target = UPLOAD_DIR / f"{sha[:10]}_{name}"
        if not target.exists():
            target.write_bytes(data)
        mapping[original] = str(target)
    cfg = _replace_strings(cfg, mapping)
    missing = sorted({t for t in _strings(cfg) if _file_like(t) and not Path(t).is_file()})
    notes = [f"restored {len(mapping)} embedded file(s) to {UPLOAD_DIR}"] if mapping else []
    notes += [f"missing: {p}" for p in missing]
    return cfg, notes


# ── Runner ─────────────────────────────────────────────────────────────────────

def loss_terms() -> list[dict]:
    """The run's non-model loss terms as run_pole's stats name them: {key, kind, weight}
    (weight None: a diagnostic, not a term of its own). "loss" is the total."""
    return [
        {"key": "match rgb", "kind": "match", "weight": opt.IMAGE_SIM_WEIGHT},
        {"key": "match gray", "kind": "match", "weight": opt.IMAGE_GRAY_WEIGHT},
        {"key": "match color stats", "kind": "match", "weight": opt.IMAGE_COLOR_STATS_WEIGHT},
        {"key": "match edges", "kind": "match", "weight": opt.IMAGE_EDGE_WEIGHT},
        {"key": "match edge stats", "kind": "match", "weight": opt.IMAGE_EDGE_STATS_WEIGHT},
        {"key": "match depth", "kind": "match", "weight": opt.IMAGE_DEPTH_WEIGHT},
        {"key": "match depth bare", "kind": "match", "weight": None},
        {"key": "match lines", "kind": "match", "weight": opt.IMAGE_LINES_WEIGHT},
        {"key": "tv", "kind": "regularization", "weight": opt.TV_WEIGHT},
        {"key": "sym", "kind": "regularization", "weight": opt.SYMMETRY_WEIGHT},
        {"key": "coverage", "kind": "field", "weight": opt.PIXEL_COVERAGE_WEIGHT},
        {"key": "folds", "kind": "field", "weight": opt.DISTORTION_FOLD_WEIGHT},
        {"key": "opacity", "kind": "field", "weight": None if opt.SHAPE_OPACITY_BUDGET is None else opt.SHAPE_OPACITY_WEIGHT},
        {"key": "hardening", "kind": "field", "weight": opt.SHAPE_HARDEN_WEIGHT},
        *({"key": f"layer {i + 1} coverage", "kind": "field", "weight": layer["coverage_weight"]}
          for i, layer in enumerate(opt.normalize_levelset(opt.LEVELSET)["layers"])),
        {"key": "loss", "kind": "total", "weight": None},
    ]


def _pole(sign: int) -> str:
    return opt.pole_name(sign)


def thumb_strip(images: torch.Tensor) -> np.ndarray:
    """(N, 3, H, W) images as one row of VIEW_THUMB thumbnails, BGR."""
    with torch.no_grad():
        thumbs = F.interpolate(images.detach(), size=(VIEW_THUMB, VIEW_THUMB),
                               mode="bilinear", antialias=True, align_corners=False)
    return np.hstack([opt.to_bgr(thumbs[i:i + 1]) for i in range(len(thumbs))])


class MapViews:
    """Depth / edge maps as BGR images. Start and current share one display range,
    taken from the target, so the pair is directly comparable."""

    def __init__(self, match: opt.ImageMatch):
        self.match = match
        if match.depth_target is not None:
            self.depth_lo, self.depth_hi = match.depth_target.min().item(), match.depth_target.max().item()
        # 99th percentile so a few strong edges don't wash the rest out.
        self.edge_hi = (max(torch.quantile(match.edges.flatten().float(), 0.99).item(), 1e-6)
                        if match.edges is not None else 1.0)
        self.lines_hi = max(match.lines.max().item(), 1e-6) if match.lines is not None else 1.0

    def depth(self, d: torch.Tensor) -> np.ndarray:
        x = ((d[0] - self.depth_lo) / max(self.depth_hi - self.depth_lo, 1e-6)).clamp(0, 1)
        return cv2.applyColorMap((x.cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    def edges(self, e: torch.Tensor) -> np.ndarray:
        x = (e[0, 0] / self.edge_hi).clamp(0, 1)
        return cv2.cvtColor((x.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def lines(self, acc: torch.Tensor) -> np.ndarray:
        """Line accumulator: offset across, angle 0..180 deg down, stretched to a square."""
        x = (acc[0] / self.lines_hi).clamp(0, 1).cpu().numpy()
        x = cv2.resize(x, (x.shape[1], x.shape[1]), interpolation=cv2.INTER_LINEAR)
        return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    def start_frames(self) -> dict[str, np.ndarray]:
        frames = {}
        if self.match.depth_target is not None:
            frames["depth_start"] = self.depth(self.match.depth_target)
        if self.match.edges is not None:
            frames["edges_start"] = self.edges(self.match.edges)
        if self.match.lines is not None:
            frames["lines_start"] = self.lines(self.match.lines)
        return frames

    def current_frames(self) -> dict[str, np.ndarray]:
        frames = {}
        if self.match.last_depth is not None:
            frames["depth_current"] = self.depth(self.match.last_depth)
        if self.match.last_edges is not None:
            frames["edges_current"] = self.edges(self.match.last_edges)
        if self.match.last_lines is not None:
            frames["lines_current"] = self.lines(self.match.last_lines)
        return frames


class StatusLog:
    """The server's warnings and errors since the last run started, for the page's Status tab.
    A repeated message (a warning raised every step) is one entry with a count."""
    MAX_ENTRIES = 500

    def __init__(self):
        self.lock = threading.Lock()
        self.version = 0  # changes with the contents: the page refetches only then
        self.clear_entries()

    def clear_entries(self) -> None:
        self.level = ""   # "", "warning", "error": the worst since the last clear
        self.entries: list[dict] = []
        self.seen: dict[tuple[str, str], dict] = {}

    def clear(self) -> None:
        with self.lock:
            self.clear_entries()
            self.version += 1

    def add(self, level: str, text: str) -> bool:
        """level: "warning" | "error". -> True if this message is new since the last clear."""
        with self.lock:
            self.version += 1
            self.level = "error" if "error" in (level, self.level) else "warning"
            entry = self.seen.get((level, text))
            if entry is not None:
                entry["count"] += 1
                return False
            entry = {"time": time.strftime("%H:%M:%S"), "level": level, "text": text, "count": 1}
            self.seen[(level, text)] = entry
            self.entries.append(entry)
            del self.entries[:-self.MAX_ENTRIES]
            return True

    def state(self, known_version: int) -> dict:
        """{version, level, entries}; entries None when the page has this version already."""
        with self.lock:
            changed = known_version != self.version
            return {"version": self.version, "level": self.level,
                    "entries": [dict(e) for e in self.entries] if changed else None}


status_log = StatusLog()


class StatusLogHandler(logging.Handler):
    """Log records at WARNING and above go to status_log. The web server's own only from ERROR:
    its warnings are about the server, not the run."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("werkzeug") and record.levelno < logging.ERROR:
            return
        status_log.add("error" if record.levelno >= logging.ERROR else "warning", self.format(record))


def capture_server_messages() -> None:
    """Send Python warnings and log records to status_log. Every warning is raised again in each
    run (Python's default shows one only once per process); the console still gets a message
    once per run. The categories Python hides by default stay hidden."""
    print_warning = warnings.showwarning

    def show_warning(message, category, filename, lineno, file=None, line=None):
        if status_log.add("warning", f"{message}  ({Path(filename).name}:{lineno})"):
            print_warning(message, category, filename, lineno, file, line)

    warnings.showwarning = show_warning
    warnings.simplefilter("always")
    for category in (DeprecationWarning, PendingDeprecationWarning, ImportWarning, ResourceWarning, BytesWarning):
        warnings.simplefilter("ignore", category)
    # A root handler replaces logging's fallback to the console, so keep one. At WARNING, as the
    # fallback was: below that, werkzeug still installs its own handler for the request lines.
    logging.basicConfig()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setLevel(logging.WARNING)
    root.addHandler(StatusLogHandler(logging.WARNING))


capture_server_messages()


class Runner:
    """One optimization at a time on a background thread; the page polls `snapshot`."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()  # set: the worker waits at the next step
        self._snapshot_job: dict | None = None  # a save_snapshot request, served by the worker
        self.show_gradient = False  # the page shows the canvas gradient: keep it (opt.CANVAS_GRAD), send it
        self.show_all_views = False  # the page shows every view of a 3D field (field.all_views): render them
        self._all_views_pending = False  # a fresh sheet is wanted at the worker's next chance (e.g. while paused)
        self._last_all_views = 0.0
        self.run_id = ""
        self.config = default_config()
        self.device = resolve_device(DEFAULT_DEVICE)
        self._reset()

    def _reset(self):
        self.status = "idle"
        self.message = ""
        self.pole = None
        self.step = 0
        self.epochs = opt.EPOCHS  # steps per pole (opt.pole_steps)
        self.active_target: int | None = None  # sequential mode: index into axes
        self.fps: float | None = None   # optimization steps / second, recent average
        self.stats: dict[str, float] = {}
        self.axes: list[dict] = []
        self.match_axes: list[dict] = []  # pixel-level match terms: trace columns after the axes
        self.traces: dict[str, list[list[float]]] = {"plus": [], "minus": []}
        # the loss_terms() in this run's stats, and their values per step, per pole
        self.terms: list[dict] = []
        self.term_traces: dict[str, list[list[float | None]]] = {"plus": [], "minus": []}
        self.view_axes: list[list[float]] = []
        self.finals: dict[str, dict] = {}
        self.result_folder = ""
        self.field_values: dict | None = None  # SURFACE, REVOLUTION: the current parameters, for the page
        self.frames: dict[str, bytes] = {}
        self.frame_version = 0
        self._last_frame = 0.0
        # for snapshots: the canvases (full resolution, detached) and the field being optimized
        self.start_canvas: torch.Tensor | None = None
        self.canvas_now: torch.Tensor | None = None
        self.pole_canvases: dict[str, torch.Tensor] = {}
        self.field = None

    # ── control ──

    def start(self, cfg: dict) -> str:
        self.stop()
        status_log.clear()
        with self.lock:
            try:
                apply_config(cfg)  # raises on a bad config, before anything is reset
            except Exception as exc:
                logging.error("Run not started: %s: %s", type(exc).__name__, exc)
                raise
            opt.CANVAS_GRAD = self.show_gradient
            self.config = cfg
            self.device = resolve_device(cfg.get("device"))
            self._reset()
            self.pause_event.clear()
            self.run_id = uuid.uuid4().hex[:8]
            self.status = "loading"
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self._run, args=(self.run_id, self.stop_event), daemon=True)
            self.thread.start()
            return self.run_id

    def stop(self):
        thread = self.thread
        if thread is not None and thread.is_alive():
            self.stop_event.set()
            thread.join(timeout=60)

    def set_show_gradient(self, show: bool) -> None:
        with self.lock:
            self.show_gradient = opt.CANVAS_GRAD = show
            if not show:
                self.frames.pop("gradient", None)

    def set_show_all_views(self, show: bool) -> None:
        """Every view of a 3D field as the "all_views" frame, while shown: the worker renders it with the
        other frames (or, paused, at once); with no run going, it is rendered here, from the last field."""
        with self.lock:
            self.show_all_views = show
            if not show:
                self.frames.pop("all_views", None)
                self.frame_version += 1
                return
            running = self.thread is not None and self.thread.is_alive()
            self._all_views_pending = running
        if not running:
            self._render_all_views()

    def _render_all_views(self) -> None:
        """The current field's all_views() as the "all_views" frame (outside the lock: it renders)."""
        self._all_views_pending = False
        field = self.field
        if field is None or not hasattr(field, "all_views"):
            return
        try:
            with torch.no_grad():
                bgr = opt.to_bgr(field.all_views())
        except Exception:  # a display, not the run: logged, not raised
            logging.exception("all views failed")
            return
        with self.lock:
            if self.show_all_views:
                self._set_frame("all_views", bgr)

    def set_paused(self, paused: bool) -> bool:
        if paused:
            self.pause_event.set()
        else:
            self.pause_event.clear()
        return self.pause_event.is_set()

    # ── snapshots ──

    def save_snapshot(self, embed: bool) -> dict:
        """Save the current run's outputs and its config (see _write_snapshot) into a new
        folder in env_config.DEFAULT_RESULTS_FOLDER -> {"folder", "files"}. While the run
        goes on, the worker saves at its next step (or while paused), so the field is
        never read mid-update."""
        job = {"embed": embed, "done": threading.Event(), "result": None, "error": None}
        with self.lock:
            if self.canvas_now is None:
                raise ValueError("Nothing to save yet: start a run")
            self._snapshot_job = job
        while not job["done"].wait(0.2):
            if self.thread is None or not self.thread.is_alive():
                self._serve_snapshot()  # no worker (any more): save here
        if job["error"] is not None:
            raise job["error"]
        return job["result"]

    def _serve_snapshot(self):
        with self.lock:
            job, self._snapshot_job = self._snapshot_job, None
        if job is None:
            return
        try:
            job["result"] = self._write_snapshot(job["embed"])
        except Exception as exc:
            logging.exception("snapshot failed")
            job["error"] = exc
        finally:
            job["done"].set()

    def _write_snapshot(self, embed: bool) -> dict:
        """config.json (a config file: images embedded unless `embed` is off), state.json
        (step, stats, axes, traces), canvas.png (the current canvas), start.png, plus.png /
        minus.png (finished poles), the page's current views and maps, the composite once
        the run is done, and the field's own outputs (field.snapshot: the pattern with its
        colours and the masks for SHAPES, the displacement for DISTORTION, ...)."""
        with self.lock:
            pole, step = self.pole, self.step
            state = {"run_id": self.run_id, "status": self.status, "pole": pole, "step": step,
                     "stats": self.stats, "axes": self.axes, "finals": self.finals, "traces": self.traces,
                     "terms": self.terms, "term_traces": self.term_traces,
                     "result_folder": self.result_folder}
            canvases = {"canvas": self.canvas_now, "start": self.start_canvas,
                        **{p: c for p, c in self.pole_canvases.items()}}
            frames = {name: data for name, data in self.frames.items()
                      if name not in ("canvas", "start", "plus", "minus")}
            field, cfg = self.field, self.config
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tag = f"{'+' if pole == 'plus' else '-'}{step}" if pole else "start"
        folder = Path(env_config.DEFAULT_RESULTS_FOLDER) / f"{stamp}_{opt.FIELD_TYPE.name.lower()}_{opt.run_slug()}_{tag}"
        folder.mkdir(parents=True, exist_ok=True)
        written = []
        (folder / "config.json").write_text(json.dumps(pack_config(cfg, embed), indent=1))
        (folder / "state.json").write_text(json.dumps(state, indent=1, default=str))
        written += ["config.json", "state.json"]
        for name, canvas in canvases.items():
            if canvas is not None:
                cv2.imwrite(str(folder / f"{name}.png"), opt.to_bgr(canvas))
                written.append(f"{name}.png")
        for name, data in frames.items():   # views, depth / edge / line maps, displacement, composite
            (folder / f"{name}.jpg").write_bytes(data)
            written.append(f"{name}.jpg")
        if field is not None and hasattr(field, "snapshot"):
            written += field.snapshot(folder)
        return {"folder": str(folder), "files": written}

    # ── worker ──

    def _set_frame(self, name: str, bgr: np.ndarray, quality: int = 90):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            self.frames[name] = buf.tobytes()
            self.frame_version += 1

    def _run(self, run_id: str, stop_event: threading.Event):
        log, error = None, ""
        device = self.device
        try:
            central = central_for(device)
            log = opt.start_log(device, {"config.json": json.dumps(pack_config(self.config, False), indent=1)})
            names = ", ".join(spec.short_name() for spec in opt.MODELS) or "the image matching terms"
            self.message = f"loading {names} ..."
            objective = opt.make_objective(central, device, log)
            maps = MapViews(objective.match) if objective.match is not None else None
            with self.lock:
                self.axes = objective.axes()
                self.epochs = opt.pole_steps(objective)
                self.match_axes = objective.match.axes() if objective.match is not None else []
                for name, bgr in (maps.start_frames() if maps else {}).items():
                    self._set_frame(name, bgr)

            # Step times for the fps estimate: from one step's callback to the next, less
            # the time spent paused; restarted with each pole.
            timing = {"sign": None, "last": None, "paused": 0.0, "dts": collections.deque(maxlen=FPS_WINDOW)}

            def on_step(sign, step, canvas, views, stats):
                if stop_event.is_set():
                    return False
                pole = _pole(sign)
                now = time.monotonic()
                if timing["sign"] != sign:
                    timing.update(sign=sign, last=None)
                    timing["dts"].clear()
                if timing["last"] is not None:
                    timing["dts"].append(now - timing["last"] - timing["paused"])
                timing["last"], timing["paused"] = now, 0.0
                dts = timing["dts"]
                with self.lock:
                    self.status, self.message = "running", ""
                    self.pole, self.step, self.stats = pole, step, stats
                    self.active_target = objective.active
                    if not self.terms:
                        self.terms = [t for t in loss_terms() if t["key"] in stats]
                    self.term_traces[pole].append([None if stats.get(t["key"]) is None
                                                   else round(float(stats[t["key"]]), 6) for t in self.terms])
                    self.fps = len(dts) / sum(dts) if dts and sum(dts) > 0 else None
                    self.canvas_now, self.field = canvas.detach(), objective.field
                    if self.start_canvas is None:
                        self.start_canvas = canvas.detach()
                    if step == 0 and "start" not in self.frames:
                        self._set_frame("start", opt.to_bgr(canvas))
                    if objective.last_axes is not None:
                        errors = [stats.get(opt.ImageMatch.stat_key(a["term"]), 0.0) for a in self.match_axes]
                        self.traces[pole].append([round(float(v), 5) for v in [*objective.last_axes, *errors]])
                        self.view_axes = np.round(objective.last_view_axes, 5).tolist()
                if now - self._last_frame >= FRAME_INTERVAL:
                    self._last_frame = now
                    strip = thumb_strip(views[:8])
                    with self.lock:
                        self._set_frame("canvas", opt.to_bgr(canvas))
                        self._set_frame("views", strip, 80)
                        for name, bgr in (maps.current_frames() if maps else {}).items():
                            self._set_frame(name, bgr)
                        if isinstance(getattr(objective, "field", None), opt.DistortionField):
                            with torch.no_grad():
                                self._set_frame("displacement", opt.flow_to_bgr(objective.field.offsets()))
                        if self.show_gradient and objective.last_canvas_grad is not None:
                            self._set_frame("gradient", opt.grad_to_bgr(objective.last_canvas_grad))
                        if isinstance(getattr(objective, "field", None),
                                      (opt.SurfaceField, opt.RevolutionField, opt.MeshTextureField,
                                       opt.Gaussians3DField, opt.VolumeFieldBase)):
                            self.field_values = objective.field.values()
                        if isinstance(getattr(objective, "field", None), opt.Gaussians3DField) \
                                and objective.field.last_views is not None:
                            self._set_frame("g3d_views", thumb_strip(objective.field.last_views))
                        if isinstance(getattr(objective, "field", None), opt.VolumeFieldBase):
                            for name, image in objective.field.previews().items():
                                self._set_frame(name, opt.to_bgr(image))
                            if objective.field.last_views is not None:
                                self._set_frame("vox_views", thumb_strip(objective.field.last_views))
                        if isinstance(getattr(objective, "field", None), opt.MeshTextureField):
                            for name, image in objective.field.previews().items():
                                self._set_frame(name, opt.to_bgr(image))
                            if objective.field.last_views is not None:
                                self._set_frame("mesh_views", thumb_strip(objective.field.last_views))
                        if isinstance(getattr(objective, "field", None), opt.RevolutionField):
                            for name, image in objective.field.previews().items():
                                self._set_frame(name, opt.to_bgr(image))
                            self._set_frame("rev_angles", thumb_strip(objective.field.last_views))
                        if isinstance(getattr(objective, "field", None), opt.PatchField) and opt.PATCH_IDENTICAL \
                                and not objective.field.from_images:
                            self._set_frame("patch", opt.to_bgr(objective.field.patch_image()))
                        if isinstance(getattr(objective, "field", None), opt.VAEField):
                            self._set_frame("latent", opt.to_bgr(objective.field.latent_image()))
                        if isinstance(getattr(objective, "field", None), opt.RearrangeField):
                            self._set_frame("assignment", opt.to_bgr(objective.field.assignment_image()))
                        if isinstance(getattr(objective, "field", None), opt.ReliefField):
                            for name, image in objective.field.previews().items():
                                self._set_frame(name, opt.to_bgr(image))
                            self._set_frame("lights", thumb_strip(objective.field.last_lit))
                    if self.show_all_views and (self._all_views_pending or now - self._last_all_views >= ALL_VIEWS_INTERVAL):
                        self._last_all_views = now
                        self._render_all_views()
                self._serve_snapshot()
                paused_at = time.monotonic()
                while self.pause_event.is_set() and not stop_event.is_set():
                    with self.lock:
                        self.status = "paused"
                    if self._all_views_pending:
                        self._render_all_views()
                    self._serve_snapshot()
                    time.sleep(0.05)
                timing["paused"] = time.monotonic() - paused_at
                return not stop_event.is_set()

            start, poles = None, {}
            for sign in ((1, -1) if opt.BIDIRECTIONAL else (1,)):
                out = opt.run_pole(sign, objective, device, log.wrap(on_step))
                if out is None:
                    break
                start, poles[sign] = out
                with self.lock:
                    r = poles[sign]
                    self.finals[_pole(sign)] = {"steps": r["steps"], "reached_budget": r["reached_budget"],
                                                "similarity_to_start": r["similarity_to_start"],
                                                "stats": r["stats"]}
                    self._set_frame(_pole(sign), opt.to_bgr(r["canvas"]))
                    self.pole_canvases[_pole(sign)] = r["canvas"]

            if poles:
                folder = opt.save_experiment(start, poles)
                composite = cv2.imread(str(folder / "composite.png"))
                with self.lock:
                    self.result_folder = str(folder)
                    self._set_frame("composite", composite)
                log.event(f"saved -> {folder}")
            with self.lock:
                self.status = "stopped" if stop_event.is_set() else "done"
        except Exception as exc:  # surfaced on the page
            logging.exception("Run failed: %s: %s", type(exc).__name__, exc)
            with self.lock:
                self.status, self.message = "error", f"{type(exc).__name__}: {exc}"
            error = traceback.format_exc()
        finally:
            if log is not None:
                log.close("error:\n" + error if self.status == "error" else self.status)
            self._serve_snapshot()

    # ── state for the page ──

    def snapshot(self, since: dict[str, int]) -> dict:
        with self.lock:
            return {
                "run_id": self.run_id, "status": self.status, "message": self.message,
                "field_type": opt.FIELD_TYPE.name,
                "paused": self.pause_event.is_set(),
                "pole": self.pole, "step": self.step, "epochs": self.epochs, "fps": self.fps,
                "active_target": self.active_target,
                "bidirectional": opt.BIDIRECTIONAL, "device": str(self.device),
                "stop_at_similarity": opt.STOP_AT_SIMILARITY,
                "stats": self.stats, "axes": self.axes, "match_axes": self.match_axes,
                "trace_offsets": {k: since.get(k, 0) for k in self.traces},
                "traces": {k: v[since.get(k, 0):] for k, v in self.traces.items()},
                "trace_lengths": {k: len(v) for k, v in self.traces.items()},
                "terms": self.terms,
                "term_offsets": {k: since.get(f"t{k}", 0) for k in self.term_traces},
                "term_traces": {k: v[since.get(f"t{k}", 0):] for k, v in self.term_traces.items()},
                "views": self.view_axes, "finals": self.finals,
                "result_folder": self.result_folder, "field_values": self.field_values,
                "frames": sorted(self.frames), "frame_version": self.frame_version,
            }


runner = Runner()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/options")
def options():
    patterns = sorted(PATTERNS_DIR.glob(f"*{RHINO_EXT}"), key=lambda p: p.name.lower())
    uploads = sorted(UPLOAD_DIR.glob(f"*{RHINO_EXT}"), key=lambda p: p.name.lower()) if UPLOAD_DIR.is_dir() else []
    # ASSEMBLY's choices: the .3dm files and the dvolume fields (.npz / .npy) of both folders.
    volumes = [*patterns, *uploads, *sorted((f for folder in (PATTERNS_DIR, UPLOAD_DIR) if folder.is_dir()
                                             for f in folder.iterdir() if f.suffix.lower() in VOLUME_EXTS
                                             and "_sdf_" not in f.name), key=lambda f: f.name.lower())]
    return jsonify(field_types=[f.name for f in FieldType], models=MODEL_IDS, model_families=MODEL_FAMILIES,
                   imagenet_models=IMAGENET_MODEL_IDS, dino_models=DINO_MODEL_IDS,
                   vlm_models=VLM_MODEL_IDS, vlm_scores=VLM_SCORES,
                   custom_models=custom_models(), score_goals=SCORE_GOALS, custom_precisions=CUSTOM_PRECISIONS, imagenet_scores=IMAGENET_SCORES, dino_tokens=DINO_TOKENS, imagenet_classes=IMAGENET_CLASSES,
                   image_fits=IMAGE_FITS, canvas_inits=CANVAS_INITS, symmetry_axes=SYMMETRY_AXES, edge_operators=EDGE_OPERATORS, depth_normalizations=DEPTH_NORMALIZATIONS,
                   distortion_offset_modes=DISTORTION_OFFSET_MODES, distortion_edge_modes=DISTORTION_EDGE_MODES,
                   vae_models=[{"id": name, "repo": repo + (f"/{sub}" if sub else "")}
                               for name, (repo, sub, _) in opt.VAE_MODELS.items()],
                   function_kinds=opt.FUNCTION_KINDS, function_waves=opt.FUNCTION_WAVES,
                   vae_precisions=VAE_PRECISIONS, conv_norms=CONV_NORMS, conv_activations=CONV_ACTIVATIONS,
                   surface=parametric_surface.schema(), revolution=surface_of_revolution.schema(),
                   mesh_texture=mesh_texture.schema(), gaussians_3d=gaussian_splats_3d.schema(),
                   voxels=voxel_field.schema(), solids=solid_fields.schema(),
                   volumes=[{"name": v.name, "path": str(v)} for v in volumes],
                   defaults=default_config(), text_templates=opt.TEXT_TEMPLATES, presets=PRESETS, config=runner.config, run_id=runner.run_id,
                   patterns=[{"name": p.name, "path": str(p)} for p in [*patterns, *uploads]],
                   devices=DEVICES)


@app.post("/api/start")
def start():
    try:
        run_id = runner.start(request.get_json(force=True))
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify(error=f"{type(exc).__name__}: {exc}"), 400
    return jsonify(run_id=run_id)


@app.post("/api/stop")
def stop():
    runner.stop()
    return jsonify(ok=True)


@app.post("/api/pause")
def pause():
    """{"paused": bool} (default: toggle) -> {"paused"}"""
    body = request.get_json(silent=True) or {}
    return jsonify(paused=runner.set_paused(bool(body.get("paused", not runner.pause_event.is_set()))))


@app.post("/api/display")
def display():
    """What the page shows that costs the run something: {"gradient": bool, "all_views": bool}"""
    body = request.get_json(silent=True) or {}
    runner.set_show_gradient(bool(body.get("gradient")))
    runner.set_show_all_views(bool(body.get("all_views")))
    return jsonify(gradient=runner.show_gradient, all_views=runner.show_all_views)


@app.post("/api/snapshot")
def snapshot():
    """Save the current run's outputs and config: {"embed": bool} -> {"folder", "files"}"""
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(runner.save_snapshot(bool(body.get("embed", True))))
    except Exception as exc:
        return jsonify(error=f"{type(exc).__name__}: {exc}"), 400


@app.post("/api/config/pack")
def config_pack():
    """{"config", "embed"} -> a config file (see pack_config)"""
    body = request.get_json(force=True)
    if not isinstance(body.get("config"), dict):
        return jsonify(error="No config"), 400
    return jsonify(pack_config(body["config"], bool(body.get("embed", True))))


@app.post("/api/config/unpack")
def config_unpack():
    """A config file -> {"config", "notes"} (see unpack_config)"""
    try:
        cfg, notes = unpack_config(request.get_json(force=True))
    except (KeyError, ValueError, TypeError, json.JSONDecodeError, base64.binascii.Error) as exc:
        return jsonify(error=f"{type(exc).__name__}: {exc}"), 400
    return jsonify(config=cfg, notes=notes)


@app.get("/api/state")
def state():
    since = {k: int(request.args.get(k, 0)) for k in ("plus", "minus", "tplus", "tminus")}
    if request.args.get("run") != runner.run_id:
        since = {}
    return jsonify(**runner.snapshot(since), status_log=status_log.state(int(request.args.get("log", -1))))


@app.get("/api/frame/<name>.jpg")
def frame(name: str):
    with runner.lock:
        data = runner.frames.get(name)
    if data is None:
        abort(404)
    return send_file(io.BytesIO(data), mimetype="image/jpeg", max_age=0)


def _jpeg(bgr: np.ndarray):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        abort(500)
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg", max_age=0)


@app.get("/api/preview/image")
def preview_image():
    """Thumbnail of an image file (start image picker), framed onto a w x h canvas by
    `fit` exactly as the optimization will see it."""
    path = Path(request.args.get("path", ""))
    if path.suffix.lower() not in _IMAGE_EXTENSIONS or not path.is_file():
        abort(404)
    fit = request.args.get("fit", "stretch")
    if fit not in IMAGE_FITS:
        abort(400)
    try:
        frame = (max(8, int(request.args.get("w", 256))), max(8, int(request.args.get("h", 256))))
        size = int(request.args.get("size", 256))
        img = Image.open(path).convert("RGB")
    except (ValueError, OSError):
        abort(400)
    scale = size / max(frame)
    thumb_size = (max(1, round(frame[0] * scale)), max(1, round(frame[1] * scale)))
    # Frame at the thumbnail's own size (same aspect as `frame`): padding / cropping a tiny
    # frame first and upsampling would blur it.
    thumb = opt.fit_image(img, thumb_size, fit, thumb_size)
    return _jpeg(cv2.cvtColor(np.asarray(thumb), cv2.COLOR_RGB2BGR))


@app.get("/api/preview/pattern")
def preview_pattern():
    """Outlines of a .3dm's closed curves as they will sit on the canvas."""
    path = Path(request.args.get("path", ""))
    if path.suffix.lower() != RHINO_EXT or not path.is_file():
        abort(404)
    w, h = int(request.args.get("w", 256)), int(request.args.get("h", 256))
    border = int(request.args.get("border", 0))
    scale = 256 / max(w, h)
    try:
        pattern = Pattern.from_rhino(path, img_size=(max(8, round(w * scale)), max(8, round(h * scale))),
                                     img_border=round(border * scale))
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    return _jpeg(cv2.cvtColor(pattern.render_outlines(), cv2.COLOR_RGB2BGR))


@app.post("/api/preview/surface")
def preview_surface():
    """{"surface": a parametric_surface config, "canvas": [w, h], "device", "background_image",
    "background_image_fit", "background_image_multiply"} -> JPEG contact sheet of its cameras'
    un-jittered views, as the run's canvas will start (at most SURFACE_PREVIEW_MAX px)."""
    body = request.get_json(force=True)
    try:
        w, h = _size(body.get("canvas") or [512, 512])
        scale = min(1.0, SURFACE_PREVIEW_MAX / max(w, h))
        size = (max(8, round(w * scale)), max(8, round(h * scale)))
        path = (body.get("background_image") or "").strip()
        fit = _choice(body, "background_image_fit", IMAGE_FITS, opt.BACKGROUND_IMAGE_FIT)
        if path and not Path(path).is_file():
            raise ValueError(f"Background image not found: {path}")
        load_image = (lambda tile: opt.frame_background(path, tile, fit)) if path else None
        sheet = parametric_surface.preview(body.get("surface"), size, resolve_device(body.get("device")),
                                           load_image, bool(body.get("background_image_multiply")))
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify(error=f"{type(exc).__name__}: {exc}"), 400
    return _jpeg(opt.to_bgr(sheet))


@app.post("/api/preview/revolution")
def preview_revolution():
    """{"revolution": a surface_of_revolution config, "canvas": [w, h], "start_image", "start_image_fit",
    "canvas_init", "background_color", "background_image", "background_image_fit",
    "background_image_multiply"} -> JPEG of the surface as a run starts (noise shows as mid grey)."""
    body = request.get_json(force=True)
    try:
        cfg = surface_of_revolution.normalize_config(body.get("revolution"))
        w, h = _size(body.get("canvas") or [512, 512])
        scale = min(1.0, REVOLUTION_PREVIEW_MAX / max(w, h))
        size = (max(8, round(w * scale)), max(8, round(h * scale)))
        cfg["softness"] = max(0.05, cfg["softness"] * scale)
        path = (body.get("background_image") or "").strip()
        fit = _choice(body, "background_image_fit", IMAGE_FITS, opt.BACKGROUND_IMAGE_FIT)
        if path and not Path(path).is_file():
            raise ValueError(f"Background image not found: {path}")
        image = opt.frame_background(path, size, fit) if path else None
        color = torch.tensor(_rgb(body.get("background_color") or "#ffffff"), dtype=torch.float32)
        background = parametric_surface.paint_background(color, image, size, bool(body.get("background_image_multiply")))
        texture = None
        if cfg["texture"]["enabled"]:
            start = (body.get("start_image") or "").strip()
            init = _choice(body, "canvas_init", CANVAS_INITS, "start")
            if start and not Path(start).is_file():
                raise ValueError(f"Start image not found: {start}")
            texture = opt.texture_start(cfg["texture"], start or None, "gray" if init == "noise" else init,
                                        _choice(body, "start_image_fit", IMAGE_FITS, "stretch"))
        sheet = surface_of_revolution.preview(cfg, size, parametric_surface.hex_color(cfg["color"], "cpu"),
                                              background, texture)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        return jsonify(error=f"{exc}"), 400
    return _jpeg(opt.to_bgr(sheet))


@app.post("/api/preview/mesh")
def preview_mesh():
    """{"mesh_texture": a mesh_texture config, "canvas": [w, h], "start_image", "start_image_fit",
    "canvas_init", "background_color", "background_image", "background_image_fit",
    "background_image_multiply"} -> JPEG sheet of every view as a run starts (the canvas view
    outlined; noise shows as mid grey), the mesh's description in the X-Mesh header."""
    body = request.get_json(force=True)
    try:
        cfg = mesh_texture.normalize_config(body.get("mesh_texture"))
        w, h = _size(body.get("canvas") or [512, 512])
        scale = min(1.0, MESH_PREVIEW_MAX / max(w, h))
        size = (max(8, round(w * scale)), max(8, round(h * scale)))
        path = (body.get("background_image") or "").strip()
        fit = _choice(body, "background_image_fit", IMAGE_FITS, opt.BACKGROUND_IMAGE_FIT)
        if path and not Path(path).is_file():
            raise ValueError(f"Background image not found: {path}")
        color = torch.tensor(_rgb(body.get("background_color") or "#ffffff"), dtype=torch.float32)
        multiply = bool(body.get("background_image_multiply"))

        def background(tile):
            image = opt.frame_background(path, tile, fit) if path else None
            return parametric_surface.paint_background(color, image, tile, multiply)

        start = (body.get("start_image") or "").strip()
        init = _choice(body, "canvas_init", CANVAS_INITS, "start")
        if start and not Path(start).is_file():
            raise ValueError(f"Start image not found: {start}")
        tw, th = cfg["texture"]["resolution"]
        # The preview's texture at most 1024 px: it only has to look like the start.
        shrink = min(1.0, 1024 / max(tw, th))
        texture = opt.texture_start({"resolution": [max(8, round(tw * shrink)), max(8, round(th * shrink))],
                                     "monochrome": mesh_texture.channels(cfg) == 1},
                                    start or None, "gray" if init == "noise" else init,
                                    _choice(body, "start_image_fit", IMAGE_FITS, "stretch"))
        sheet, described = mesh_texture.preview(cfg, size, texture, parametric_surface.hex_color(cfg["color"], "cpu"),
                                                 background)
    except (KeyError, ValueError, TypeError, OSError, RuntimeError) as exc:
        return jsonify(error=f"{exc}"), 400
    response = _jpeg(opt.to_bgr(sheet))
    response.headers["X-Mesh"] = described
    return response


def _scene_inputs(body: dict, size: tuple[int, int], gray: bool):
    """A 3D preview's background and start, from its request body (see preview_gaussians_3d):
    (background(tile) -> (1, 3, h, w), the (1, C, H, W) start image framed on a canvas of `size` or
    None, the canvas init, the noise std, the seed)."""
    path = (body.get("background_image") or "").strip()
    fit = _choice(body, "background_image_fit", IMAGE_FITS, opt.BACKGROUND_IMAGE_FIT)
    if path and not Path(path).is_file():
        raise ValueError(f"Background image not found: {path}")
    color = torch.tensor(_rgb(body.get("background_color") or "#ffffff"), dtype=torch.float32)
    multiply = bool(body.get("background_image_multiply"))

    def background(tile):
        image = opt.frame_background(path, tile, fit) if path else None
        return parametric_surface.paint_background(color, image, tile, multiply)

    start_path = (body.get("start_image") or "").strip()
    init = _choice(body, "canvas_init", CANVAS_INITS, "start")
    if start_path and not Path(start_path).is_file():
        raise ValueError(f"Start image not found: {start_path}")
    start = None
    if init == "start" and start_path:
        image = Image.open(start_path).convert("L" if gray else "RGB")
        fitted = opt.fit_image(image, size, _choice(body, "start_image_fit", IMAGE_FITS, "stretch"), size)
        start = transforms.functional.to_tensor(fitted)[None]
    noise = body.get("canvas_noise_std")
    return (background, start, init, 0.4 if noise in (None, "") else max(0.0, float(noise)),
            int(body.get("seed") or 0))


def _preview_size(body: dict, limit: int) -> tuple[int, int]:
    w, h = _size(body.get("canvas") or [512, 512])
    scale = min(1.0, limit / max(w, h))
    return max(8, round(w * scale)), max(8, round(h * scale))


@app.post("/api/preview/gaussians3d")
def preview_gaussians_3d():
    """{"gaussians_3d": a gaussian_splats_3d config, "canvas": [w, h], "seed", "start_image",
    "start_image_fit", "canvas_init", "canvas_noise_std", "background_color", "background_image",
    "background_image_fit", "background_image_multiply"} -> JPEG sheet of every camera's view of the
    scene as a run with that seed starts (the canvas view outlined)."""
    body = request.get_json(force=True)
    try:
        cfg = gaussian_splats_3d.normalize_config(body.get("gaussians_3d"))
        size = _preview_size(body, GAUSSIANS_3D_PREVIEW_MAX)
        background, start, init, noise, seed = _scene_inputs(body, size, cfg["monochrome"])
        sheet = gaussian_splats_3d.preview(cfg, size, background, start, init, noise, seed)
    except (KeyError, ValueError, TypeError, OSError, RuntimeError) as exc:
        return jsonify(error=f"{exc}"), 400
    return _jpeg(opt.to_bgr(sheet))


@app.post("/api/preview/voxels")
def preview_voxels():
    """{"voxels": a voxel_field config, and the rest as preview_gaussians_3d's} -> JPEG sheet of every
    camera's view of the volume as a run with that seed starts (the canvas view outlined)."""
    body = request.get_json(force=True)
    try:
        cfg = voxel_field.normalize_config(body.get("voxels"))
        size = _preview_size(body, VOXELS_PREVIEW_MAX)
        background, start, init, noise, seed = _scene_inputs(body, size, cfg["mode"] == "monochrome")
        with torch.no_grad():
            sheet = voxel_field.preview(cfg, size, background, start, init, noise, seed)
    except (KeyError, ValueError, TypeError, OSError, RuntimeError) as exc:
        return jsonify(error=f"{exc}"), 400
    return _jpeg(opt.to_bgr(sheet))


@app.post("/api/preview/solid/<kind>")
def preview_solid(kind: str):
    """{<kind lowercased>: an ASSEMBLY / SOLIDS_ON_OFF / EXTRUSIONS config, and the rest as
    preview_gaussians_3d's} -> JPEG sheet of every camera's view as a run with that seed starts. A mesh
    voxelized for the first time takes a while (then dvolume's cache beside it is read)."""
    kind = kind.upper()
    if kind not in solid_fields.NORMALIZERS:
        abort(404)
    body = request.get_json(force=True)
    try:
        size = _preview_size(body, VOXELS_PREVIEW_MAX)
        background, _, _, _, seed = _scene_inputs(body, size, False)
        with torch.no_grad():
            sheet = solid_fields.preview(kind, body.get(kind.lower()) or {}, size, background, seed)
    except (KeyError, ValueError, TypeError, OSError, RuntimeError) as exc:
        return jsonify(error=f"{exc}"), 400
    return _jpeg(opt.to_bgr(sheet))


@app.post("/api/revolution/formula")
def revolution_formula():
    """{"formula", "parameters": {name: {value, optimize}}} -> {"parameters"}, the formula's free
    parameters (see surface_of_revolution.reconcile_parameters), or {"error"}."""
    body = request.get_json(force=True)
    try:
        formula = surface_of_revolution.parse_formula(body.get("formula"))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(parameters=surface_of_revolution.reconcile_parameters(formula, body.get("parameters")))


@app.post("/api/upload")
def upload():
    """Save an uploaded start image, .3dm pattern or dvolume field (.npz / .npy); returns its path on disk."""
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(error="No file"), 400
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _IMAGE_EXTENSIONS and suffix != RHINO_EXT and suffix not in VOLUME_EXTS:
        return jsonify(error=f"Unsupported file type {suffix}"), 400
    UPLOAD_DIR.mkdir(exist_ok=True)
    path = UPLOAD_DIR / Path(file.filename).name
    file.save(path)
    return jsonify(path=str(path))


@app.get("/api/random_image")
def random_image():
    """A random image from a folder (or from the folder of a given image path)."""
    folder = Path(request.args.get("path", ""))
    if folder.is_file():
        folder = folder.parent
    if not folder.is_dir():
        return jsonify(error=f"Not a folder: {folder}"), 400
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS]
    if not images:
        return jsonify(error=f"No images in {folder}"), 400
    return jsonify(path=str(random.choice(images)))


class _QuietPolls(logging.Filter):
    """Drop request-log lines for the polls and frames."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/api/state" not in msg and "/api/frame/" not in msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP optimization explorer")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    logging.getLogger("werkzeug").addFilter(_QuietPolls())
    print(f"Devices: {', '.join(d['id'] for d in DEVICES)} | http://localhost:{args.port}")
    app.run(debug=False, port=args.port, threaded=True)
