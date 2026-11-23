"""CLIP-guided image optimization: not to make a good picture, but to watch what
CLIP pulls an image towards -- its built-in biases and aesthetics. MODELS lists the models
the image is optimized for, all at once, each with its own targets (ModelSpec): CLIP (text /
image targets), DINOv2 (image targets), a torchvision ImageNet classifier (ImageNet-1k
classes, scored by their logits or softmax probabilities), or a custom model (ModelID.custom,
shared.custom_models.differentiable): a classifier trained in custom_models (its own
classes), an autoencoder (image targets in its latent space, like DINOv2), or a RankNet /
ready-made aesthetic score (maximize, minimize, or reach a value). Their losses add up.

One optimizable tensor (the "field") is rendered to a canvas, the canvas is seen by
CLIP through a batch of random augmentations, and the field follows the gradient of
weighted text / image similarities. FIELD_TYPE picks what the field is:

  PIXELS_COLOR  RGB image at PIXEL_SRC_RESOLUTION, resampled to the canvas (optionally through
                a contrast sigmoid, PIXEL_CONTRAST)
  PIXELS_GRAY   same, single channel (optionally pulled to a mean brightness, PIXEL_COVERAGE)
  SHAPES        the closed curves in SHAPE_SRC, composited in file order over the background; each
                optimizes (SHAPE_MODE) its RGB colour, a mix of two colours, or its opacity
  DISTORTION    a low-res displacement field warping START_IMAGE via grid_sample
  PIXEL_PATCHES PATCH_COUNT small RGBA (or gray + alpha) images, each placed by its own
                optimizable affine transform and composited over BACKGROUND_COLOR; or
                (PATCH_SOURCE "images") fixed images, only their transforms optimized
  GAUSSIANS     GAUSSIAN_COUNT anisotropic Gaussian splats over BACKGROUND_COLOR; with
                GAUSSIAN_CONTRAST > 0 each splat gets crisp edges and keeps its own colour,
                or, with GAUSSIAN_MONOCHROME too, their summed density is thresholded into
                one crisp shape in GAUSSIAN_FG_COLOR
  SURFACE       a parametric 3-D surface (parametric_surface.py: a base shape plus a stack of
                deformations, see SURFACE), seen from several cameras through a differentiable
                renderer; the cameras' and lights' jitter replace the 2-D augmentations
  LEVELSET      a stack of monochrome layers (LEVELSET), each its own resolution: resized to the
                canvas and through its contrast sigmoid, it is the alpha mask of the layer's RGBA
                colour; composited over a background in order. Colours and contrasts are optimized
                too, or frozen, per layer
  VAE           the latent of a pretrained image autoencoder (VAE_MODEL, e.g. Stable Diffusion's),
                decoded to the canvas: the optimization moves through what the VAE can draw
  CONVOLUTIONS  a small CNN built on the fly, its weights optimized: a stack of stride-2 transposed
                convolutions growing an optimizable input tensor to the image, or (CONV_INPLACE) a
                stack of same-size convolutions over the fixed start image
  REARRANGE     the start image's tiles (REARRANGE_GRID), rearranged: each destination tile is a
                soft blend of the sources (softmax / Sinkhorn of optimized logits) annealed to one
  FUNCTION_STACK a stack of curves (FUNCTION_STACK) over BACKGROUND_COLOR: a horizontal line or a
                circle, moved by an offset and bent by superimposed sine / triangle / square waves
                whose amplitudes, phases and frequencies are optimized; filled or stroked, sigmoid-edged
  RELIEF        one height per closed curve in SHAPE_SRC: the composited heights are a displacement map
                RELIEF_DEPTH px deep, lit from several directions (LIGHT_COUNT fixed lights, plus
                LIGHT_RANDOM_COUNT random ones redrawn every step). Every light's image is scored; the
                first light's is the canvas
  REVOLUTION    a surface of revolution (surface_of_revolution.py, see REVOLUTION): a half profile of
                points, waves or a formula, optionally with an optimized texture, seen by an
                orthographic camera in closed form; a turn of the surface only rotates its uvs
  MESH_TEXTURE  the texture of a Rhino mesh (mesh_texture.py, see MESH_TEXTURE): the mesh is
                prerendered once from fixed cameras into float G-buffers (mask, world xyz, normals,
                uv); every step the texture is sampled through them and lit, differentiably. The
                texture is RGB albedo, gray albedo, or a bump map over a material colour
  GAUSSIANS_3D  a scene of 3D Gaussian splats (gaussian_splats_3d.py, see GAUSSIANS_3D): centres,
                scales, rotations, opacities and colours, seen by orbiting cameras and composited
                front to back per pixel as in 3D Gaussian splatting; the canvas is one camera
  VOXELS        a voxel volume (voxel_field.py, see VOXELS): density and colour, each a Laplacian
                pyramid of 3D tensors, rendered by dvolume's volume renderer from orbiting cameras;
                exported as dvolume fields and a marching cubes mesh
  ASSEMBLY      clones of solids (solid_fields.py, see ASSEMBLY: groups of a .3dm / .npz and a count),
                each placed by an optimized 3D affine transformation, composited by sum or max
  SOLIDS_ON_OFF the meshes of one .3dm voxelized in place (SOLIDS_ON_OFF), each with an optimized
                density multiplier pushed towards on / off and a coverage
  EXTRUSIONS    closed-curve patterns on the XY, YZ, ZX planes (EXTRUSIONS), each shape extruded through
                the volume with its own multiplier, negative ones subtracting

Against adversarial high-frequency noise: pixel fields can be a Laplacian pyramid
(PIXEL_PYRAMID_LEVELS), there is a total-variation penalty (TV_WEIGHT), a small blur
of the field every step (BLUR_STRENGTH / BLUR_SIGMA), and the augmentations themselves
(crops, rotations, jitter, noise) only reward features that survive them.

Bias experiments:
  - every CLIP text prompt is an ensemble over its model's templates (and over paraphrases, when a
    prompt is given as a list), so a direction is a concept, not one sentence's quirks
  - BIDIRECTIONAL runs the optimization twice from the same start, with the text /
    direction weights flipped, and saves a [- pole | start | + pole] composite with
    amplified difference maps below -- the contrast between poles is the finding
  - STOP_AT_SIMILARITY ends each run once CLIP(canvas) . CLIP(start) drops to a fixed
    value, so edits are compared at equal size rather than equal step count

Preview: OpenCV window. Esc stops, S saves the current canvas.

Every run also gets a log folder in OUTPUT_FOLDER/logs (run_log.py): its settings, and
the gpu's memory, load, temperature and power every few seconds, kept on disk line by line.

Run:  uv run python src/engine/optimization.py
"""
import json
import math
import re
import sys
import time
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field as dc_field, replace
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torchvision import transforms

from pattern import Pattern, save_colored_rhino
import parametric_surface
import gaussian_splats_3d
import mesh_texture
import solid_fields
import voxel_field
import surface_of_revolution
from surface_of_revolution import wave_function
from run_log import RunLog
from shared.models.frozen import cap_gpu_memory
from shared.models.model_central import ModelCentral
from shared.models.embedding_models import INPUT_SIZE
from shared.models.model_types import ModelID, CLIPArch, CLIPPretrained, ImageNetArch, DINOV2Arch, VLMArch


class FieldType(Enum):
    PIXELS_COLOR = 1
    PIXELS_GRAY = 2
    SHAPES = 3
    DISTORTION = 4
    SURFACE = 5
    LEVELSET = 6
    PIXEL_PATCHES = 7
    GAUSSIANS = 8
    VAE = 9
    CONVOLUTIONS = 10
    REARRANGE = 11
    FUNCTION_STACK = 12
    RELIEF = 13
    REVOLUTION = 14
    MESH_TEXTURE = 15
    GAUSSIANS_3D = 16
    VOXELS = 17
    ASSEMBLY = 18
    SOLIDS_ON_OFF = 19
    EXTRUSIONS = 20

# ── Models ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ImagePrompt:
    """An image file whose embedding stands in for a text prompt or a class."""
    path: str


@dataclass(frozen=True)
class VLMQuery:
    """A question about the image and an answer, for a vision-language model: its target
    is the answer's log-probability (shared.models.vlm)."""
    question: str
    answer: str


Prompt = str | list[str] | ImagePrompt | VLMQuery


@dataclass
class ModelSpec:
    """One model of the objective with its own targets, by ModelID family:
      clip       targets / directions: text prompts (a list = paraphrases, averaged) or ImagePrompts
      dino       targets / directions: ImagePrompts (no text encoder)
      imagenet   targets / directions: ImageNet-1k class indices or ImagePrompts
      custom     ModelID.custom(id), id from shared.custom_models.differentiable.list_models()
                 ("folder/<name>" in env_config's models folder, "card/<file>.ranknet.json",
                 "aesthetic/<scorer>"), by its kind:
          classifier   targets / directions: its class indices or ImagePrompts
          autoencoder  targets / directions: ImagePrompts (cosine in its latent space)
          ranknet / aesthetic   no targets: the score itself, to `goal`
      vlm        ModelID.vlm(VLMArch...): targets / directions: VLMQuery(question, answer), scored
                 by log p(answer | image, question) (`score`: "logprob" the answer's, "mean" per token).
                 A direction rewards logp(target) - logp(source): VLMQuery(q, "No") -> VLMQuery(q,
                 "Yes") is the log-odds of a yes / no question.
    A target is (prompt | class | ImagePrompt, weight). A direction (source, target, weight)
    rewards the change of the embedding from the start lining up with E(target) - E(source)
    (StyleGAN-NADA style); between two classes it rewards score[target] - score[source]. An
    image direction joins two images, a class direction two classes."""
    model_id: ModelID
    targets: list = dc_field(default_factory=list)
    directions: list = dc_field(default_factory=list)
    score: str = "logits"      # classifiers: "logits" (raw, unbounded) | "softmax" (probability, 0..1;
                               # saturates -- its gradient vanishes once a class wins or is hopeless).
                               # VLMs: "logprob" (the answer's) | "mean" (per answer token)
    tokens: str = "cls"        # DINOv2: the "cls" token (global semantics) or all "patch" tokens, each
                               # normalized and flattened (similarity = mean per-patch cosine: keeps the
                               # layout, so it is sensitive to the augmentations' shifts)
    prompt: str = "an image"   # PickScore / HPSv2: the score is how well the image fits this text
    goal: str = "maximize"     # scores: "maximize" | "minimize" | "value" (bring the score to `value`).
                               # BIDIRECTIONAL's - pole swaps maximize and minimize; a value is the same on both.
    value: float = 0.0
    weight: float = 1.0        # scores: the weight of the score term
    precision: str = "fp32"    # CLIP, custom models, VLMs: "fp32" | "bf16" (the backbone / image tower in bf16:
                               # about half the gpu memory, scores barely change but the gradients of
                               # large models can differ a lot; VLMs: the whole model)
    use_templates: bool = True  # CLIP: embed each text prompt through `templates` and average; off: as written
    templates: list | None = None  # CLIP: "{}" = the prompt; None -> TEXT_TEMPLATES

    def text_templates(self) -> list[str]:
        """CLIP: the templates a text prompt is embedded through (["{}"] = as written)."""
        return (self.templates or TEXT_TEMPLATES) if self.use_templates else ["{}"]

    def central_id(self) -> ModelID:
        """The id of this spec's model in ModelCentral: a CLIP model in bf16 is a separate one."""
        if self.model_id.is_clip() and self.precision != "fp32":
            return replace(self.model_id, extra=self.precision)
        return self.model_id

    def short_name(self) -> str:
        """For labels: the CLIP / DINOv2 / ImageNet architecture, a custom model's name."""
        if self.model_id.is_custom():
            return Path(self.model_id.arch).name.removesuffix(".ranknet.json")
        if self.model_id.is_vlm():
            return self.model_id.arch.split("/")[-1]   # the repo's name, without its owner
        return self.model_id.arch


CANVAS_RESOLUTION = (512, 512)  # (width, height)

FIELD_TYPE = FieldType.PIXELS_COLOR

# The models the image is optimized for, all at once: their losses add up (an empty list leaves
# only the image matching terms below). The first one also measures the edit size (sim->start,
# STOP_AT_SIMILARITY). See ModelSpec for the targets each family takes.
MODELS = [
    ModelSpec(ModelID.clip(CLIPArch.VITB32, CLIPPretrained.DATACOMP_XL_S13B_B90K),
              directions=[("a woman", "a feminine woman", 1.0)]),
              # targets=[("a beautiful woman", -1.0)], directions=[("a chair", ["a feminine chair", "a girly chair"], 1.0)]
    # ModelSpec(ModelID.dino(DINOV2Arch.VITB14), targets=[(ImagePrompt(r"C:\path\to\image.jpg"), 1.0)]),
    # ModelSpec(ModelID.vlm(VLMArch.QWEN3_VL_2B), precision="bf16",
    #           directions=[(VLMQuery("Is this person a woman? Answer yes or no.", "No"),
    #                        VLMQuery("Is this person a woman? Answer yes or no.", "Yes"), 1.0)]),
    # ModelSpec(ModelID.imagenet(ImageNetArch.RESNET50, "IMAGENET1K_V2"), targets=[(309, 1.0)]),  # bee
    # ModelSpec(ModelID.imagenet(ImageNetArch.RESNET50, "IMAGENET1K_V2"), directions=[(281, 291, 1.0)]),  # tabby -> lion
    # ModelSpec(ModelID.custom("aesthetic/laion_ava"), goal="maximize"),
    # ModelSpec(ModelID.custom("aesthetic/pickscore"), prompt="a drawing of a cat", precision="bf16"),
]

START_IMAGE = r"D:\Dropbox\Projects\DiffDrawing\datasets\art_rng_3000\00852.jpg"  # None -> mid grey
CANVAS_INIT = "start"  # what the field starts as: "start" (START_IMAGE) | "gray" (solid mid-grey) | "noise"
                       # (uniform random, SEED). START_IMAGE stays the match target either way.
                       # PIXELS_*, SHAPES and LEVELSET only -- DISTORTION always warps START_IMAGE.
                       # LEVELSET "start": layer i is the level set luminance > (i + 1) / (layers + 1).
CANVAS_NOISE_STD = 0.4  # "noise": std in logit space around mid-grey (0.4 -> pixels ~0.4..0.6, 1.5 -> ~0.05..0.95)
START_IMAGE_FIT = "letterbox"  # onto the canvas: "stretch" | "letterbox" (mid-grey bars) | "crop" (centre)

#pixels src
PIXEL_SRC_RESOLUTION = (256, 256)  # (width, height)
PIXEL_PYRAMID_LEVELS = 4           # 1 = plain pixels; >1 = Laplacian pyramid, biased to low frequencies
PIXEL_PYRAMID_LR_DECAY = 1.0       # learning rate x this per level towards the finest (finest gets LR x decay^(levels-1)).
                                   # 1 = same LR everywhere (Adam then lets the many fine pixels dominate);
                                   # ~0.5 = coarse-to-fine, which resists adversarial fine-scale noise when
                                   # generating from gray / noise.
PIXEL_CONTRAST = 0.0               # 0 = off; else the canvas is contrast_curve(image, this): a sigmoid of
                                   # (c - 0.5) * this per channel, rescaled so 0 and 1 stay put -- ~10+
                                   # pushes every pixel towards black or white
PIXEL_COVERAGE = None              # PIXELS_GRAY: the mean brightness to pull the canvas to (None = off).
                                   # 1 = all white, 0 = all black, 0.5 = mid grey -- or, with a high
                                   # PIXEL_CONTRAST, half the pixels black and half white
PIXEL_COVERAGE_WEIGHT = 10.0       # x (mean - PIXEL_COVERAGE)^2

#levelset src
# Layers composited over `background` in list order (later on top). A layer is a (w, h) tensor in
# [0, 1] (the sigmoid of its logits), resized to the canvas, through contrast_curve(., contrast)
# (0 = linear) and times the colour's alpha: the alpha mask of its colour. optimize_color /
# optimize_contrast: optimized (a contrast in log space, so it stays > 0) or frozen. coverage: the
# share of the canvas the layer's mask should cover, 0..1 (None = free), pulled to by
# coverage_weight x (mean mask - coverage)^2.
# See normalize_levelset.
LEVELSET = {
    "background": "#000000", "optimize_background": False,
    "layers": [
        {"resolution": [32, 32], "color": "#555555", "alpha": 1.0, "optimize_color": True,
         "contrast": 10.0, "optimize_contrast": False},
        {"resolution": [64, 64], "color": "#aaaaaa", "alpha": 1.0, "optimize_color": True,
         "contrast": 10.0, "optimize_contrast": False},
        {"resolution": [128, 128], "color": "#ffffff", "alpha": 1.0, "optimize_color": True,
         "contrast": 10.0, "optimize_contrast": False},
    ],
}

#distortion src
# The displacement is a pyramid of fields (finest = DISTORTION_FIELD_RESOLUTION, each next level
# half that), each bicubically upsampled, bounded by tanh to its own cap, and summed. Offsets
# are isotropic: one unit is the same distance in x and y.
DISTORTION_FIELD_RESOLUTION = (32, 32)  # (width, height) of the finest level
DISTORTION_PYRAMID_LEVELS = 1           # 1 = one field; more = coarser levels for large-scale warps
DISTORTION_PYRAMID_LR_DECAY = 1.0       # LR x this per level towards the finest (like PIXEL_PYRAMID_LR_DECAY):
                                        # < 1 favours large, smooth distortion over fine wiggles
DISTORTION_OFFSET_MODE = "absolute"     # per-level cap: "absolute" (DISTORTION_MAX % of the canvas' long
                                        # side, same at every level) | "resolution" (DISTORTION_MAX x that
                                        # level's grid step, so coarse levels may move further; > 1 allowed)
DISTORTION_MAX = 5.0
DISTORTION_EDGE_MODE = "reflection"     # what is pulled in from outside the image: "reflection" | "border" | "gray"
DISTORTION_PREVENT_FOLDS = True         # penalize the warp folding over itself (Jacobian determinant < margin)
DISTORTION_FOLD_WEIGHT = 10.0
DISTORTION_FOLD_MARGIN = 0.2            # the determinant is kept above this (1 = no local compression)

#shape src
SHAPE_SRC = str(Path(__file__).resolve().parent.parent / "patterns" / "trigs_4x4.3dm")
SHAPE_BORDER = 0                        # px around the pattern's bbox on the canvas
SHAPE_MODE = "color"                    # what each shape optimizes: "color": its RGB colour | "monochrome": one amount
                                        # in (0, 1) mixing SHAPE_COLOR_1 (0) to SHAPE_COLOR_2 (1) | "masking": its
                                        # opacity in (0, 1), SHAPE_COLOR_1 blended over the background
SHAPE_COLOR_1 = (0.0, 0.0, 0.0)         # RGB in [0, 1]: monochrome's first colour, masking's only one
SHAPE_COLOR_1_OPTIMIZE = False          # True: SHAPE_COLOR_1 is optimized too
SHAPE_COLOR_2 = (1.0, 1.0, 1.0)         # RGB in [0, 1], monochrome only
SHAPE_COLOR_2_OPTIMIZE = False          # True: SHAPE_COLOR_2 is optimized too
SHAPE_OPACITY_BUDGET = None             # masking: the % of the canvas to cover, 0..100: the per-pixel opacity
                                        # (the largest of the shapes there), summed, / the canvas' pixels (None = free)
SHAPE_OPACITY_WEIGHT = 10.0             # x (mean opacity - budget)^2, opacity and budget as fractions
SHAPE_HARDEN_STEPS = 0                  # masking: forces shapes to almost fully opaque or fully transparent (0 = off).
                                        # Counted from the run's start: no hardening for the first half of these
                                        # steps, rising over the second half to its full weight at this step, then
                                        # held there. The run also ignores STOP_AT_SIMILARITY, or it could end before
                                        # they harden
SHAPE_HARDEN_WEIGHT = 10.0              # x ramp x mean((1 - pole x logit / SHAPE_HARDEN_LOGIT)+^2).
                                        # Poles: the nearest, or per the budget (ShapeField.harden_poles)
SHAPE_HARDEN_LOGIT = 4.6                # the logit counted as hard: opacity 0.01 or 0.99

#relief src (the shapes are SHAPE_SRC / SHAPE_BORDER's; the background material is BACKGROUND_COLOR / _OPTIMIZE / _IMAGE)
RELIEF_COLOR = (1.0, 1.0, 1.0)          # RGB in [0, 1]: the shapes' material
RELIEF_COLOR_OPTIMIZE = False           # True: RELIEF_COLOR is optimized too
RELIEF_DEPTH = 8.0                      # px: height of a shape at value 1 (the background is at 0)
RELIEF_SMOOTH = 1.5                     # px: Gaussian sigma on the height map before the normals -- the bevel
                                        # width of the walls (0 = the masks' 1-px antialiased edges)
LIGHT_COUNT = 4                         # fixed lights at evenly spaced azimuths, starting at LIGHT_AZIMUTH (>= 1)
LIGHT_AZIMUTH = 135.0                   # deg, counter-clockwise from +x on screen: 135 = from the top left
LIGHT_ELEVATION = 45.0                  # deg above the canvas plane, fixed lights
LIGHT_RANDOM_COUNT = 2                  # extra lights at random azimuths, redrawn every step
LIGHT_RANDOM_ELEVATION = (20.0, 70.0)   # deg, range of the random lights' elevation
LIGHT_AMBIENT = 0.2                     # shade = ambient + (1 - ambient) * max(0, n . l)

#patch src
# PATCH_COUNT m x m images with alpha, composited in order (later on top). Each has its own
# position, scale and (with PATCH_ROTATION) rotation, all optimized along with the pixels.
PATCH_SOURCE = "pixels"                 # "pixels": PATCH_COUNT optimizable images | "images": PATCH_IMAGES, their
                                        # pixels fixed, only the transforms optimized (the pixel-only settings --
                                        # count, resolution, identical, monochrome, randomize -- don't apply)
PATCH_IMAGES: list[dict] = []           # "images": [{"path": ..., "count": n}], each image placed n times. Each is
                                        # fitted into a PATCH_IMAGE_RESOLUTION square, padded transparent (a PNG's
                                        # alpha is kept), so a patch's side is its image's long side
PATCH_IMAGE_RESOLUTION = 128
PATCH_COUNT = 16
PATCH_RESOLUTION = 32                   # m: each patch is m x m pixels
PATCH_IDENTICAL = False                 # True: one m x m image shared by every patch (only placements differ)
PATCH_ROTATION = True                   # False: patches stay axis-aligned
PATCH_SCALE_RANGE = (0.1, 0.3)          # patch side, as a fraction of the canvas' short side
PATCH_MONOCHROME = False                # gray + alpha instead of RGB + alpha
PATCH_OPAQUE = False                    # True: alpha frozen at 1, every patch fully opaque (colours only);
                                        # "images": the images' own alpha ignored (their padding stays clear)
PATCH_TRANSFORM_LR = 1.0                # LR x this for position / scale / rotation
PATCH_RANDOMIZE_TRANSFORMS = False      # True: transforms are not optimized but redrawn at random every
                                        # frame -- the pixels must work wherever the patches land

#gaussian src
GAUSSIAN_COUNT = 64
GAUSSIAN_NEGATIVE_COUNT = 0             # extra splats in the background colour, composited last: they cut voids
GAUSSIAN_MAX_SIZE = 0.15                # largest std of a splat along either axis, fraction of the canvas' short side
GAUSSIAN_MONOCHROME = False             # gray splat colours (with GAUSSIAN_CONTRAST > 0: one shape, see there)
GAUSSIAN_SPREAD = 1.0                   # initial scatter of the centres: 0 = all at the canvas centre, 1 = uniform
                                        # over the canvas (to 90% of the way to its edges)
GAUSSIAN_CONTRAST = 0.0                 # 0 = off: soft splats alpha-composited in their own colours. > 0 (~20 = crisp):
                                        # each splat's alpha goes through contrast_curve(alpha, contrast), still
                                        # composited in its own colour. With GAUSSIAN_MONOCHROME instead: the summed
                                        # density goes through sigmoid(contrast * (density - 0.5)), a clean shape
                                        # mask painted GAUSSIAN_FG_COLOR over the background
GAUSSIAN_FG_COLOR = (0.0, 0.0, 0.0)     # RGB in [0, 1], contrast + monochrome only
GAUSSIAN_FG_OPTIMIZE = False            # True: GAUSSIAN_FG_COLOR is optimized too (the background follows BACKGROUND_OPTIMIZE)
GAUSSIAN_RECTILINEAR = False           # True: each splat also optimizes a roundness in (0, 1); its density is
                                        # exp(-(|u|^p + |v|^p) / 2) with p = 2 at roundness 1 (the plain
                                        # Gaussian) up to GAUSSIAN_MAX_POWER at 0 (a flat block, sharp corners)
GAUSSIAN_MAX_POWER = 32.0               # p at roundness 0: edges about 2 * size / p wide
GAUSSIAN_ROUNDNESS_INIT = 0.5           # starting roundness of every splat, in (0, 1)
GAUSSIAN_AXIS_ALIGNED = False           # True: splat angles fixed at 0 (edges parallel to the canvas')

#function stack src
# Curves composited over the background in list order (later on top). kind "line": the height of a
# horizontal line through the canvas centre; kind "circle": the radius of a circle of radius
# FUNCTION_BASE_RADIUS at the centre. Either way the curve is
#   offset + sum_k amplitude_k * wave(2 pi * frequency_k * t + phase_k),   k = 1 .. components
# with t in 0..1 across the canvas' width or around the circle. frequency: the first wave's starting
# frequency (wave k starts at k * frequency), in cycles across the canvas / around the circle; a
# circle only closes at whole numbers, elsewhere it has a seam at its left. Lengths are fractions of
# the canvas' short side; a positive offset moves a line up and grows a circle. amplitude: the first
# wave's starting amplitude (wave k starts at amplitude / k). Amplitudes and phases are always
# optimized, the offset, frequencies and RGBA colour per optimize_offset / optimize_frequency /
# optimize_color. filled: everything below the line / inside
# the circle; else a stroke `thickness` wide. softness: the sigmoid edges' scale, in canvas
# pixels. shape_lr: offsets' and amplitudes' learning rate, as a multiple of LEARNING_RATE (they move
# in canvas units, phases in radians). See normalize_function_stack.
FUNCTION_STACK = {
    "kind": "line", "softness": 1.0, "shape_lr": 0.5,
    "functions": [
        {"wave": "sine", "offset": 0.25, "components": 4, "frequency": 1.0, "amplitude": 0.06, "color": "#aaaaaa"},
        {"wave": "triangle", "offset": 0.0, "components": 4, "frequency": 2.0, "amplitude": 0.05, "color": "#555555"},
        {"wave": "square", "offset": -0.15, "components": 2, "frequency": 3.0, "amplitude": 0.04, "color": "#000000"},
    ],
}
FUNCTION_KINDS = ["line", "circle"]
FUNCTION_WAVES = surface_of_revolution.WAVES
FUNCTION_BASE_RADIUS = 0.25     # the circle at offset 0, fraction of the canvas' short side

#background (PIXEL_PATCHES, GAUSSIANS, FUNCTION_STACK, RELIEF, SHAPES: the canvas' base layer, under the shapes;
# REVOLUTION, MESH_TEXTURE, GAUSSIANS_3D and the volume fields: behind the object)
BACKGROUND_COLOR = (1.0, 1.0, 1.0)      # RGB in [0, 1]; luminance for monochrome fields
BACKGROUND_OPTIMIZE = False             # True: optimized from BACKGROUND_COLOR (fg colours have their own switches:
                                        # GAUSSIAN_FG_OPTIMIZE, SHAPE_COLOR_1_OPTIMIZE, SHAPE_COLOR_2_OPTIMIZE)
#background image (BACKGROUND_IMAGE_FIELDS; LEVELSET and SURFACE keep their own background colour settings)
# An image in place of the solid background, over that colour: letterbox bars and an image's transparent
# parts show the colour.
BACKGROUND_IMAGE: str | None = None     # a path; None = the solid colour alone
BACKGROUND_IMAGE_FIT = "crop"           # framing on the canvas: "letterbox" | "crop" (centre crop) | "stretch"
BACKGROUND_IMAGE_MULTIPLY = False       # True: the image x the background colour
BACKGROUND_IMAGE_FIELDS = (FieldType.SHAPES, FieldType.SURFACE, FieldType.LEVELSET, FieldType.PIXEL_PATCHES,
                           FieldType.GAUSSIANS, FieldType.FUNCTION_STACK, FieldType.RELIEF, FieldType.REVOLUTION,
                           FieldType.MESH_TEXTURE, FieldType.GAUSSIANS_3D, FieldType.VOXELS, FieldType.ASSEMBLY,
                           FieldType.SOLIDS_ON_OFF, FieldType.EXTRUSIONS)
VOLUME_FIELDS = (FieldType.VOXELS, FieldType.ASSEMBLY, FieldType.SOLIDS_ON_OFF, FieldType.EXTRUSIONS)
#revolution src
# A surface_of_revolution config (see normalize_config there): the half profile (mode "points" |
# "waves" | "parametric"), the texture (off, or mapped around the surface: monochrome or colour, starts
# per CANVAS_INIT), shading, the Rhino export's mesh. The canvas is one view; with a texture, the
# models see texture.views angles a step, RANDOM_TRANSFORM_COUNT split between them.
REVOLUTION = surface_of_revolution.default_config()
#mesh texture src
# A mesh_texture config (see normalize_config there): the .3dm (its meshes, or its Breps' render
# meshes), the mode ("color" | "monochrome" | "bump"), where the uvs come from (the mesh's own, or a
# planar / cylindrical / spherical projection), the texture (resolution, pyramid; starts per
# CANVAS_INIT), the cameras (count, how many the models see per step, orbit, fov, framing, which
# one is the canvas), the light and the supersampling. The canvas is one view; the models see
# views.per_step of them a step, at random, RANDOM_TRANSFORM_COUNT split between them.
MESH_TEXTURE = mesh_texture.default_config()
#3d gaussians src
# A gaussian_splats_3d config (see normalize_config there): how many Gaussians, colour or monochrome,
# where and how big they start (a ball of `spread` scene radii, starting colours per CANVAS_INIT: the
# start image seen from the canvas camera), their size bound and starting opacity, the learning
# rates of the centres and of the shapes, the cameras (as MESH_TEXTURE's, around a unit sphere) and
# how much the models' cameras are jittered. The canvas is one view; the models see views.per_step
# of them a step, at random, RANDOM_TRANSFORM_COUNT split between them.
GAUSSIANS_3D = gaussian_splats_3d.default_config()
#voxels src
# A voxel_field config (see normalize_config there): the volume's resolution and pyramid, the mode
# ("color" | "monochrome" | "density": a material colour), how the density starts (a ball, smooth
# noise, empty, or a .3dm's meshes; the colour per CANVAS_INIT: the start image seen from the canvas
# camera), the renderer (depth samples, opacity, depth jitter, render scale), the light, a 3D total
# variation weight, the cameras (as MESH_TEXTURE's, around a unit sphere) and their jitter, and the
# exported isosurface's level. Rendered by dvolume (shared.dvolume).
VOXELS = voxel_field.default_config()
#solid fields src (solid_fields.py; rendered as VOXELS, with the same rendering settings in each config)
# ASSEMBLY: groups of solids ({path: .3dm / .npz / .npy, count, color, scale}), each clone placed by an
#   optimized affine transformation (translation, rotation, scale), composited by "max" or "add".
# SOLIDS_ON_OFF: every mesh of one .3dm voxelized in place, an optimized multiplier each, pushed towards
#   on / off (harden_weight) and a share of the volume on (coverage %, coverage_weight).
# EXTRUSIONS: closed-curve patterns on the XY, YZ, ZX planes (YZ, ZX empty: the XY one), each shape
#   extruded through the volume with a multiplier in -1..1 (negative = subtract) or 0..1.
ASSEMBLY = solid_fields.normalize_assembly({})
SOLIDS_ON_OFF = solid_fields.normalize_on_off({})
EXTRUSIONS = solid_fields.normalize_extrusions({})
#surface src
# A parametric_surface config (see normalize_config there): base ("plane" | "sphere" | "ellipsoid",
# the ellipsoid's radii optimizable), deformation (a stack of plane waves: angle, frequency,
# amplitude, phase each), which parameters to optimize, uv spans, smooth / faceted shading, the
# cameras (distance, azimuth, elevation, fov, jitter each), lighting ("shared" | "camera" |
# "random"), material colours and the renderer. The canvas is a contact sheet of every camera's
# view; the models see each camera, jittered, at CANVAS_RESOLUTION (plus TRANSFORM_NOISE).
SURFACE = parametric_surface.default_config()

#vae src
# The field is a VAE's latent, scaled to about unit variance (the diffusion models' convention:
# (z - shift_factor) x scaling_factor). It starts as the encoding of the start image ("start"),
# of mid grey ("gray"), or mid grey's plus CANVAS_NOISE_STD x N(0, 1) ("noise"); render() decodes it.
# VAE_MODELS: name -> (Hugging Face repo, subfolder, diffusers class), all public. Any diffusers
# AutoencoderKL / AutoencoderTiny / AutoencoderDC repo can be added.
VAE_MODELS = {
    "sd-vae-ft-mse (SD 1.x, 4ch f8)": ("stabilityai/sd-vae-ft-mse", None, "AutoencoderKL"),
    "sd-vae-ft-ema (SD 1.x, 4ch f8)": ("stabilityai/sd-vae-ft-ema", None, "AutoencoderKL"),
    "sdxl-vae (SDXL, 4ch f8)": ("madebyollin/sdxl-vae-fp16-fix", None, "AutoencoderKL"),
    "flux2-vae (FLUX.2, 32ch f8)": ("black-forest-labs/FLUX.2-klein-4B", "vae", "AutoencoderKLFlux2"),
    "vae-kl-f8-d16 (ostris, 16ch f8)": ("ostris/vae-kl-f8-d16", None, "AutoencoderKL"),
    "dc-ae-f32c32 (Sana, 32ch f32)": ("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", None, "AutoencoderDC"),
    "taesd (tiny SD 1.x, 4ch f8)": ("madebyollin/taesd", None, "AutoencoderTiny"),
    "taesdxl (tiny SDXL, 4ch f8)": ("madebyollin/taesdxl", None, "AutoencoderTiny"),
    "taef1 (tiny FLUX.1, 16ch f8)": ("madebyollin/taef1", None, "AutoencoderTiny"),
}
VAE_MODEL = "sd-vae-ft-mse (SD 1.x, 4ch f8)"
VAE_PRECISION = "fp32"          # "fp32" | "bf16" (about half the memory and time; no fp16: the
                                # original SDXL VAE overflows in it)
VAE_DECODE_SCALE = 1.0          # decoded at the canvas x this (rounded to the VAE's factor), then resized
                                # to the canvas: < 1 is faster and lighter, the latent smaller
VAE_LATENT_BOUND = 3.0          # |z| beyond this (unit-variance units) is penalized: far outside the prior
VAE_LATENT_WEIGHT = 1.0         # the decoder draws saturated garbage. x mean(relu(|z| - bound)^2)

#convolutions src
# Generative (CONV_INPLACE off): an optimizable (CONV_CHANNELS, h, w) input at CONV_INPUT_RESOLUTION,
# through CONV_LAYERS stride-2 transposed convolutions, each doubling the resolution and halving the
# channels (at least 8; LeakyReLU between), the last to RGB; sigmoid, resized to the canvas.
# CANVAS_INIT "start" first fits the net to the start image (CONV_FIT_STEPS of plain MSE); "gray" /
# "noise" start from the random weights.
# Inplace (CONV_INPLACE on): CANVAS_INIT's image (start / gray / noise) at the canvas size stays fixed;
# CONV_LAYERS same-size convolutions (CONV_CHANNELS wide, LeakyReLU between, reflected edges) map it
# to a residual added to its logits. The last layer starts at zero, so the canvas starts as the image.
# Either way the weights (and the generative input) are what is optimized.
CONV_INPLACE = False
CONV_INPUT_RESOLUTION = (6, 6)  # (width, height), generative only: output = this x 2**CONV_LAYERS
CONV_LAYERS = 6
CONV_CHANNELS = 128             # generative: the input's and first layer's channels; inplace: every layer's
CONV_KERNEL = 4                 # kernel size (generative: >= 2; inplace: any, padded 'same')
CONV_FIT_STEPS = 300            # generative with CANVAS_INIT "start": Adam steps fitting the start image
CONV_NORM = "group"             # after every hidden conv: "none" | "instance" | "group" (GroupNorm, 8 groups).
                                # Without one, the activations can grow without limit as the weights move:
                                # inplace at LR 0.03 blew the residual up to ~4e4 logits (a solid colour) in
                                # 1000 steps, and a deep random generative net outputs nearly flat grey
CONV_ACTIVATION = "leaky_relu"  # between layers: "leaky_relu" | "relu" | "silu"
CONV_RESIDUAL_MAX = 2.0         # inplace: the residual is soft-clipped to +- this (logits, tanh); 0 = unbounded
CONV_WEIGHT_LR = 1.0            # LR x this for the weights (Adam moves every weight ~LR per step, so a
                                # whole net drifts far faster than a pixel field at the same LR)

#rearrange src
# START_IMAGE cut into REARRANGE_GRID tiles (K of them); destination tile d shows sum_s P[d, s] x
# source tile s. P comes from optimized logits L (K x K): softmax(L / T) per row, or with
# REARRANGE_PERMUTATION Sinkhorn-normalized so every source is also used once in total (a soft,
# doubly-stochastic permutation). T anneals geometrically from REARRANGE_TEMP_START to
# REARRANGE_TEMP_END over the run, so P sharpens towards one-hot. From REARRANGE_HARD_AFTER x the
# run, the canvas shows the nearest hard assignment (Hungarian for permutations, argmax otherwise)
# with the soft P's gradient (straight-through), so the result is an exact rearrangement.
REARRANGE_GRID = (8, 8)           # (columns, rows); the tiles are the canvas / this (the image is fitted
                                  # to a whole number of tiles, the result resized back to the canvas)
REARRANGE_PERMUTATION = True      # False: each destination picks any source (tiles may repeat or drop out)
REARRANGE_INIT = "random"         # "random": every tile starts as a near-uniform blend of all tiles |
                                  # "identity": starts as the start image (each tile in its own place)
REARRANGE_INIT_STD = 0.1          # "random": logits ~ this x N(0, 1) (SEED)
REARRANGE_TEMP_START = 1.0
REARRANGE_TEMP_END = 0.05
REARRANGE_ENTROPY_WEIGHT = 0.0    # x P's mean row entropy (0..1, 0 = one-hot), ramped in over the run
REARRANGE_HARD_AFTER = 0.9        # fraction of the run; 1 = soft to the end
REARRANGE_SINKHORN_ITERS = 20
REARRANGE_LR = 10.0               # LR x this for the logits

#Loss
# Matching the target image (None = no target). Each weight at 0 turns its term off.
IMAGE_SIM_TARGET = START_IMAGE
# Per-term target overrides, {term: image path}; a term without one matches IMAGE_SIM_TARGET.
# Terms: "rgb" "gray" "depth" "edges" "color_stats" "edge_stats" "lines".
MATCH_TARGETS: dict[str, str] = {}
IMAGE_SIM_WEIGHT = 1.0      # rgb: MSE of the RGB values, bare canvas vs target (for an embedding match,
                            # add the image as a text / image target instead)
IMAGE_GRAY_WEIGHT = 0.0     # gray: MSE of luminance, bare canvas vs target
IMAGE_DEPTH_WEIGHT = 0.0    # depth: MSE of scale/shift-normalized monocular depth (DEPTH_MODEL)
IMAGE_EDGE_WEIGHT = 0.0     # edges: MSE of edge maps of the luminance (EDGE_OPERATOR)
IMAGE_COLOR_STATS_WEIGHT = 0.0  # color stats: sliced Wasserstein distance between the pixel color
                                # distributions (palette, spread, cross-channel mix; ignores layout)
COLOR_STATS_PROJECTIONS = 32    # random RGB directions per step for the color stats term
IMAGE_EDGE_STATS_WEIGHT = 0.0   # edge stats: sliced Wasserstein distance between the distributions of
                                # luminance gradients as orientation (mod 180 deg) x strength -- same
                                # edge density and dominant orientations as the target, anywhere
EDGE_STATS_SCALES = 3           # pyramid levels (full, 1/2, 1/4, ...): fine texture to coarse contours
EDGE_STATS_PROJECTIONS = 16     # evenly spaced (randomly rotated) directions per scale per step
IMAGE_LINES_WEIGHT = 0.0        # lines: MSE between differentiable Hough (Radon) accumulators -- straight
                                # lines at the same angle and offset, scored by how continuously they run
LINES_RESOLUTION = 128          # long side the line transform runs at
LINES_ANGLES = 90               # line angle bins over 180 deg
LINES_ORIENTATION_SHARPNESS = 4 # a pixel votes for a line by |g| * cos^2(gradient - line normal)^this
LINES_BLUR = 1.5                # Gaussian over (angle, offset) bins: tolerance to small shifts / tilts
EDGE_OPERATOR = "sobel"     # "sobel" (gradient magnitude) | "laplacian" (|second derivative|)
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"  # transformers depth-estimation model id
DEPTH_INPUT_SIZE = 266      # long side fed to the depth model (rounded to its patch multiple); 518 = native
# Against adversarial depth matches (a pattern that fools this one network at this one size
# into reading the target depth): score the depth term over the bare canvas plus this many
# randomly transformed copies, each compared to the target under the *same* transform.
DEPTH_AUGMENT_VIEWS = 4     # 0 = bare canvas only
# Per-map normalization before the depth MSE (monocular depth is only defined up to scale + shift):
# "mean" (subtract mean, / std) | "mad" (subtract median, / mean |d - median|; MiDaS, robust to
# outliers) | "none" (raw relative inverse depth; its range varies per image -- ~0.7..4.9 for the
# default portrait with Depth-Anything-V2-Small -- so the term also penalizes range differences).
DEPTH_NORMALIZATION = "mean"
DEPTH_AUG_ROTATION = 8.0    # degrees, +/-
DEPTH_AUG_TRANSLATE = 0.04  # fraction of the canvas, +/-
DEPTH_AUG_SCALE = (0.85, 1.15)
DEPTH_AUG_NOISE = 0.03      # pixel noise std on the canvas copies (not the target)

# The default templates of a CLIP model (ModelSpec.templates): every text prompt is embedded
# through each and averaged. A prompt may also be a list of paraphrases, averaged the same way.
# ModelSpec.use_templates = False embeds its prompts as written (paraphrases still averaged).
TEXT_TEMPLATES = [
    "{}",
    "a photo of {}",
    "a painting of {}",
    "a picture of {}",
    "an image of {}",
    "a drawing of {}",
    "a close-up of {}",
    "an artwork depicting {}",
]

# Keep d(loss)/d(canvas) of every step in Objective.last_canvas_grad (for previews): the
# backward pass computes it anyway, keeping it costs one canvas of memory. Read every step.
CANVAS_GRAD = False

# Directions start from the field's first render, averaged over START_DIRECTION_VIEWS views.
START_DIRECTION_VIEWS = 32

#experiment
BIDIRECTIONAL = False       # also run with every model's target / direction weights negated
STOP_AT_SIMILARITY = None   # stop once cos(canvas, start) in the first model's embedding <= this (None = run all EPOCHS)
DIFF_GAIN = 4.0             # amplification of the difference maps in the composite
SEQUENTIAL_TARGETS = False  # one target / direction at a time, in order, each continuing from the last
                            # state (same field and optimizer): an animation through the targets.
                            # Image / match terms and regularizers stay on throughout; the
                            # STOP_AT_SIMILARITY budget and EPOCHS are not used.
SEQUENTIAL_STEPS = 200      # steps per target in sequential mode
SEQUENTIAL_CYCLES = 1       # passes through the target list in sequential mode

#regularization
TV_WEIGHT = 0.05          # total variation of the canvas (distortion: of the field)
SYMMETRY_WEIGHT = 0.0     # MSE between the canvas and its mirror image (0 = off)
SYMMETRY_AXIS = "left-right"  # "left-right" (mirror about the vertical axis) | "top-bottom" | "both"
BLUR_STRENGTH = 0.0       # per step, field <- lerp(field, blur(field), strength) (0 = never). Keep 0 with
                          # STOP_AT_SIMILARITY: the blur spends edit budget regardless of the prompt,
                          # more so on the slower pole, which confounds the comparison.
                          # Otherwise ~0.02;
                          # continuous diffusion -- a full blur every N steps undoes N steps of progress
BLUR_SIGMA = 0.75         # in field pixels
MEDIAN_WEIGHT = 0.0       # per step, field <- lerp(field, median(field), weight) (0 = never): removes speckle
                          # and thin strokes while keeping edges, unlike the Gaussian blur
MEDIAN_KERNEL = 3         # odd window size, in field pixels

#transforms (each copy draws its own). A copy's crop, flip, rotation and perspective are one
#homography, sampled once from the canvas straight to the models' input size (see augment).
RANDOM_TRANSFORM_COUNT = 8
CROP_SCALE = (0.5, 1.0)   # area of the canvas a copy covers, as a fraction (before rotation's fit)
CROP_RATIO = (0.8, 1.25)  # its width / height in canvas pixels (log-uniform)
ROTATION = 10             # degrees, +-
PERSPECTIVE = 0.0         # corners pulled inward by up to this fraction of the half-size (0 = off), half the copies
FLIP = False              # horizontal, half the copies
VIEW_MARGIN = 0.05        # how far past the canvas edge (fraction of its size) a copy may reach into grey;
                          # a copy that does not fit (e.g. full-size and rotated) zooms in instead.
                          # 0 = never any fill; a little lets border pixels away from the view's edge
COLOR_JITTER = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
TRANSFORM_NOISE = 0.03    # gaussian pixel noise std added after the transforms
INCLUDE_UNTRANSFORMED = True  # also score the bare canvas

#optimization
LEARNING_RATE = 0.005     # Adam; pixel/shape fields live in logit space, distortion in atanh space.
                          # Low on purpose: at 0.02 the 0.85 budget is spent in ~5 steps of Adam noise.
EPOCHS = 1000             # max steps per run (per pole)
SEED = 0

#output
PREVIEW_SIZE = 512
PREVIEW_EVERY = 1
OUTPUT_FOLDER = Path(__file__).resolve().parents[2] / "results"


# ── Helpers ────────────────────────────────────────────────────────────────────

def fit_image(img: Image.Image, size: tuple[int, int], fit: str | None = None,
              frame: tuple[int, int] | None = None) -> Image.Image:
    """Resize to size (width, height) by `fit` (default START_IMAGE_FIT). The framing
    (padding / crop) is decided against `frame`'s aspect (default CANVAS_RESOLUTION),
    then resampled to `size`, so a field stored at another resolution
    (PIXEL_SRC_RESOLUTION) frames identically."""
    fit = fit or START_IMAGE_FIT
    frame = frame or CANVAS_RESOLUTION
    if fit == "letterbox":
        fill = 128 if img.mode == "L" else (128, 128, 128)
        img = ImageOps.pad(img, frame, Image.LANCZOS, color=fill)
    elif fit == "crop":
        img = ImageOps.fit(img, frame, Image.LANCZOS)
    return img.resize(size, Image.LANCZOS)


def load_image(path: str | None, size: tuple[int, int], gray: bool = False) -> torch.Tensor:
    """(1, C, H, W) in [0, 1]; size is (width, height), see fit_image. None -> mid grey."""
    channels = 1 if gray else 3
    if path is None:
        return torch.full((1, channels, size[1], size[0]), 0.5)
    img = fit_image(Image.open(path).convert("L" if gray else "RGB"), size)
    return TF.to_tensor(img).unsqueeze(0)


def load_rgba(path: str, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(4, m, m) in [0, 1]: the image (its own alpha, else opaque) fitted into an m x m square
    at its aspect, centred, the rest transparent; and (1, m, m) 1 over the image's rectangle."""
    img = Image.open(path).convert("RGBA")
    scale = m / max(img.size)
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    square = Image.new("RGBA", (m, m), (0, 0, 0, 0))
    x, y = (m - img.width) // 2, (m - img.height) // 2
    square.paste(img, (x, y))
    rect = torch.zeros(1, m, m)
    rect[:, y:y + img.height, x:x + img.width] = 1
    return TF.to_tensor(square), rect


def frame_background(path: str, size: tuple[int, int], fit: str) -> torch.Tensor:
    """(1, 4, H, W) in [0, 1], straight colours + alpha: the image framed on a canvas of size
    (width, height) by `fit`; letterbox bars are transparent."""
    img = Image.open(path).convert("RGBA")
    if fit == "letterbox":
        img = ImageOps.pad(img, size, Image.LANCZOS, color=(0, 0, 0, 0))
    elif fit == "crop":
        img = ImageOps.fit(img, size, Image.LANCZOS)
    else:
        img = img.resize(size, Image.LANCZOS)
    return TF.to_tensor(img).unsqueeze(0)


def background_image(size: tuple[int, int], gray: bool, device) -> torch.Tensor | None:
    """(1, C + 1, H, W) BACKGROUND_IMAGE framed for a canvas of size (width, height), for
    backdrop(); its luminance when gray. None without an image."""
    if BACKGROUND_IMAGE is None:
        return None
    rgba = frame_background(BACKGROUND_IMAGE, size, BACKGROUND_IMAGE_FIT).to(device)
    return torch.cat([luminance(rgba[:, :3]), rgba[:, 3:]], 1) if gray else rgba


def backdrop(color: torch.Tensor, image: torch.Tensor | None, size: tuple[int, int] | None = None) -> torch.Tensor:
    """(1, C, H, W) a field's background: its colour (C,), and background_image() if any."""
    return parametric_surface.paint_background(color, image, size or CANVAS_RESOLUTION, BACKGROUND_IMAGE_MULTIPLY)


def load_image_unframed(path: str, size: tuple[int, int]) -> torch.Tensor:
    """(1, 3, h, w) in [0, 1]: the whole image at its own aspect, scaled to about the pixel
    count of size (width, height) -- no letterbox bars or crop. For the layout-free terms
    (color / edge stats), where padding would join the distribution and a crop drop part of it."""
    img = Image.open(path).convert("RGB")
    scale = math.sqrt(size[0] * size[1] / (img.width * img.height))
    img = img.resize((max(8, round(img.width * scale)), max(8, round(img.height * scale))), Image.LANCZOS)
    return TF.to_tensor(img).unsqueeze(0)


def initial_image(size: tuple[int, int], gray: bool = False) -> torch.Tensor:
    """(1, C, H, W) starting image for a pixel field, per CANVAS_INIT. Noise is drawn
    from torch's CPU generator, so it follows SEED (same noise for both poles)."""
    if CANVAS_INIT == "gray":
        return torch.full((1, 1 if gray else 3, size[1], size[0]), 0.5)
    if CANVAS_INIT == "noise":
        # Noise in logit space around mid-grey. (Uniform [0, 1] noise put ~11% of the
        # field where the sigmoid is nearly flat -- logits up to +-7 -- freezing it as noise.)
        return torch.sigmoid(CANVAS_NOISE_STD * torch.randn(1, 1 if gray else 3, size[1], size[0]))
    return load_image(START_IMAGE, size, gray)


def logit(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def resize(x: torch.Tensor, size: tuple[int, int], mode: str = "bicubic") -> torch.Tensor:
    """Resize to size (width, height), bicubic (or `mode`: "bilinear"); antialiased when shrinking."""
    if x.shape[-1] == size[0] and x.shape[-2] == size[1]:
        return x
    return F.interpolate(x, size=(size[1], size[0]), mode=mode, align_corners=False, antialias=True)


def contrast_curve(x: torch.Tensor, k: float | torch.Tensor) -> torch.Tensor:
    """sigmoid((x - 0.5) * k), rescaled so that 0 -> 0 and 1 -> 1: towards the identity as k -> 0,
    a hard threshold at 0.5 for large k. k: a number (0 = x unchanged; the sign is ignored) or a
    differentiable 0-d tensor > 0. (sigmoid(z) = (1 + tanh(z / 2)) / 2, so the rescaled curve is
    1/2 + tanh((x - 0.5) k / 2) / (2 tanh(k / 4)).)"""
    if not torch.is_tensor(k):
        if k == 0:
            return x
        k = abs(k)
        return 0.5 + 0.5 * torch.tanh((x - 0.5) * (k / 2)) / math.tanh(k / 4)
    k = k.clamp_min(1e-3)
    return 0.5 + 0.5 * torch.tanh((x - 0.5) * (k / 2)) / torch.tanh(k / 4)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return ((x[..., 1:, :] - x[..., :-1, :]).abs().mean()
            + (x[..., :, 1:] - x[..., :, :-1]).abs().mean())


def symmetry_error(canvas: torch.Tensor) -> torch.Tensor:
    """MSE between the canvas and its mirror image(s) per SYMMETRY_AXIS; 0 = symmetric."""
    dims = {"left-right": [(-1,)], "top-bottom": [(-2,)], "both": [(-1,), (-2,)]}[SYMMETRY_AXIS]
    return sum(F.mse_loss(canvas, canvas.flip(d)) for d in dims) / len(dims)


_LUMA = (0.299, 0.587, 0.114)
_SOBEL_X = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
_LAPLACIAN = ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))


def luminance(img: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) Rec.601 luma of a (B, 3, H, W) image."""
    return (img * img.new_tensor(_LUMA).view(1, 3, 1, 1)).sum(1, keepdim=True)


def edge_map(img: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) edges of the luminance: Sobel gradient magnitude or |Laplacian|."""
    lum = F.pad(luminance(img), (1, 1, 1, 1), mode="replicate")
    if EDGE_OPERATOR == "laplacian":
        return F.conv2d(lum, img.new_tensor(_LAPLACIAN).view(1, 1, 3, 3)).abs()
    kx = img.new_tensor(_SOBEL_X).view(1, 1, 3, 3)
    gx, gy = F.conv2d(lum, kx), F.conv2d(lum, kx.transpose(-1, -2))
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def pixel_samples(img: torch.Tensor) -> torch.Tensor:
    """(C, B*H*W): every pixel of a (B, C, H, W) map as one C-dim sample."""
    return img.permute(1, 0, 2, 3).reshape(img.shape[1], -1)


def sliced_wasserstein(a: torch.Tensor, b: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Sliced Wasserstein-2 (squared) between the sample sets a (C, N) and b (C, M) along the
    K unit dirs (K, C): each slice is a 1-D optimal transport, i.e. sorted vs sorted."""
    a, b = (dirs @ a).sort(dim=1).values, (dirs @ b).sort(dim=1).values
    if a.shape[1] != b.shape[1]:  # different sample counts: compare at matching quantiles
        b = F.interpolate(b[:, None], size=a.shape[1], mode="linear", align_corners=True)[:, 0]
    return F.mse_loss(a, b)


def color_stats_error(img: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sliced Wasserstein between the color distributions of img and target: the RGB axes
    plus COLOR_STATS_PROJECTIONS random directions."""
    dirs = F.normalize(torch.cat([torch.eye(3, device=img.device),
                                  torch.randn(COLOR_STATS_PROJECTIONS, 3, device=img.device)]), dim=1)
    return sliced_wasserstein(pixel_samples(img), pixel_samples(target), dirs)


def edge_orientation_maps(img: torch.Tensor) -> list[torch.Tensor]:
    """Per pyramid level (EDGE_STATS_SCALES), (B, 2, H, W) Sobel gradients of the luminance in
    doubled-angle form: length |g|, angle 2*theta. Opposite gradients (dark->light vs
    light->dark across the same edge) land on the same vector, so the 2-D distribution is
    edge strength x orientation mod 180 deg; flat areas pile up at the origin (density)."""
    lum = luminance(img)
    kx = img.new_tensor(_SOBEL_X).view(1, 1, 3, 3)
    maps = []
    for level in range(EDGE_STATS_SCALES):
        if level:
            lum = F.avg_pool2d(lum, 2)
        padded = F.pad(lum, (1, 1, 1, 1), mode="replicate")
        gx, gy = F.conv2d(padded, kx), F.conv2d(padded, kx.transpose(-1, -2))
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        maps.append(torch.cat([gx * gx - gy * gy, 2 * gx * gy], dim=1) / mag)
    return maps


def edge_stats_error(maps: list[torch.Tensor], target_maps: list[torch.Tensor]) -> torch.Tensor:
    """Mean over scales of the sliced Wasserstein between doubled-angle gradient distributions
    (see edge_orientation_maps), along EDGE_STATS_PROJECTIONS evenly spaced directions."""
    k = EDGE_STATS_PROJECTIONS
    err = 0.0
    for a, b in zip(maps, target_maps):
        angles = (torch.arange(k, device=a.device) + torch.rand(1, device=a.device)) * (torch.pi / k)
        dirs = torch.stack([angles.cos(), angles.sin()], dim=1)
        err = err + sliced_wasserstein(pixel_samples(a), pixel_samples(b), dirs)
    return err / len(maps)


_LINE_GRIDS: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _line_grid(h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cached for an h x w map: the grid_sample grid (A, n, n, 2) that walks, for each of the
    A = LINES_ANGLES normal angles theta and each offset rho (px from the centre), along the
    line x cos(theta) + y sin(theta) = rho; theta (A,); and each line's length inside the
    map (A, n), floored at a quarter of the short side so corner stubs don't blow up."""
    key = (h, w, LINES_ANGLES, str(device))
    if key not in _LINE_GRIDS:
        r = math.hypot(h, w) / 2
        n = math.ceil(2 * r)
        steps = torch.linspace(-r, r, n, device=device)
        theta = torch.arange(LINES_ANGLES, device=device) * (math.pi / LINES_ANGLES)
        cos, sin = theta.cos().view(-1, 1, 1), theta.sin().view(-1, 1, 1)
        rho, t = steps.view(1, -1, 1), steps.view(1, 1, -1)
        x, y = rho * cos - t * sin, rho * sin + t * cos
        grid = torch.stack([x / (w / 2), y / (h / 2)], dim=-1)
        ones = torch.ones(LINES_ANGLES, 1, h, w, device=device)
        length = F.grid_sample(ones, grid, align_corners=False).sum(-1)[:, 0]
        _LINE_GRIDS[key] = (grid, theta, length.clamp(min=0.25 * min(h, w)))
    return _LINE_GRIDS[key]


def line_transform(img: torch.Tensor) -> torch.Tensor:
    """(B, A, n) differentiable Hough / Radon accumulator of a (B, 3, H, W) image at
    LINES_RESOLUTION: rows are line normal angles (0..180 deg), columns offsets from the
    centre. Each pixel votes with its Sobel gradient magnitude, gated by how well the gradient
    lines up with the line's normal (an edge running along the line); a cell is the mean
    vote along its line, so a long continuous line peaks and scattered texture stays low."""
    lum = luminance(img)
    scale = LINES_RESOLUTION / max(lum.shape[-2:])
    if scale < 1:
        lum = F.interpolate(lum, scale_factor=scale, mode="bilinear", antialias=True, align_corners=False)
    h, w = lum.shape[-2:]
    padded = F.pad(lum, (1, 1, 1, 1), mode="replicate")
    kx = img.new_tensor(_SOBEL_X).view(1, 1, 3, 3)
    gx, gy = F.conv2d(padded, kx), F.conv2d(padded, kx.transpose(-1, -2))
    mag2 = gx * gx + gy * gy + 1e-8
    grid, theta, length = _line_grid(h, w, img.device)
    # cos^2(phi - theta) = (1 + cos 2phi cos 2theta + sin 2phi sin 2theta) / 2
    cos2t, sin2t = (2 * theta).cos().view(1, -1, 1, 1), (2 * theta).sin().view(1, -1, 1, 1)
    align = (1 + ((gx * gx - gy * gy) * cos2t + 2 * gx * gy * sin2t) / mag2) / 2
    votes = mag2.sqrt() * align.clamp(0, 1) ** LINES_ORIENTATION_SHARPNESS  # (B, A, h, w)
    b, a = votes.shape[:2]
    acc = F.grid_sample(votes.reshape(b * a, 1, h, w), grid.repeat(b, 1, 1, 1), align_corners=False).sum(-1)
    return acc.view(b, a, -1) / length


def blur_lines(acc: torch.Tensor) -> torch.Tensor:
    """Gaussian blur (LINES_BLUR bins) of a (B, A, n) line accumulator. The angle axis wraps:
    the line at theta + 180 deg is the one at theta with the offset negated."""
    if LINES_BLUR <= 0:
        return acc
    k = max(1, math.ceil(3 * LINES_BLUR))
    g = torch.exp(-0.5 * (torch.arange(-k, k + 1, device=acc.device, dtype=acc.dtype) / LINES_BLUR) ** 2)
    g = g / g.sum()
    x = torch.cat([acc[:, -k:].flip(-1), acc, acc[:, :k].flip(-1)], dim=1)[:, None]  # (B, 1, A + 2k, n)
    x = F.conv2d(x, g.view(1, 1, -1, 1))                                               # angle (wrapped)
    x = F.conv2d(x, g.view(1, 1, 1, -1), padding=(0, k))                               # offset
    return x[:, 0]


class DepthEstimator:
    """Frozen monocular depth model (DEPTH_MODEL through transformers), differentiable
    w.r.t. its input image. Loaded models are cached per name across runs, and moved
    to the device a run asks for.
    (No torch hub MiDaS fallback: MiDaS_small pulls a second hub repo whose
    trust prompt blocks on stdin, which hangs the web server.)"""

    _cache: dict[str, 'DepthEstimator'] = {}
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    @classmethod
    def get(cls, name: str, device) -> 'DepthEstimator':
        est = cls._cache.get(name)
        if est is None:
            est = cls._cache[name] = cls(name, device)
        elif est.device != torch.device(device):
            est.model.to(device)
            est.device = torch.device(device)
        return est

    def __init__(self, name: str, device):
        try:
            from transformers import AutoModelForDepthEstimation
            model = AutoModelForDepthEstimation.from_pretrained(name)
        except Exception as exc:
            raise RuntimeError(f"Depth model {name!r} could not be loaded ({type(exc).__name__}: {exc}). "
                               "It needs `transformers` and the model in the Hugging Face cache or online.") from exc
        self.name = name
        self.multiple = 14  # ViT patch size: inputs must be a multiple
        self.device = torch.device(device)
        self.model = model.to(device).eval().requires_grad_(False)
        print(f"[depth] {self.name}")

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """(B, h, w) relative depth (inverse depth / disparity, larger = nearer) of a
        (B, 3, H, W) image in [0, 1], at roughly DEPTH_INPUT_SIZE on the long side."""
        h, w = img.shape[-2:]
        scale = DEPTH_INPUT_SIZE / max(h, w)
        m = self.multiple
        size = (max(m, round(h * scale / m) * m), max(m, round(w * scale / m) * m))
        x = F.interpolate(img, size=size, mode="bilinear", align_corners=False, antialias=True)
        x = (x - x.new_tensor(self._MEAN).view(1, 3, 1, 1)) / x.new_tensor(self._STD).view(1, 3, 1, 1)
        return self.model(pixel_values=x).predicted_depth


def normalize_depth(d: torch.Tensor) -> torch.Tensor:
    """Remove the scale and shift monocular depth is only defined up to, per map, by
    DEPTH_NORMALIZATION (see there). d is (B, h, w)."""
    if DEPTH_NORMALIZATION == "none":
        return d
    if DEPTH_NORMALIZATION == "mad":
        center = d.flatten(-2).median(dim=-1).values[..., None, None]
        d = d - center
        return d / d.abs().mean(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    d = d - d.mean(dim=(-1, -2), keepdim=True)
    return d / d.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)


def match_target(term: str) -> str | None:
    """The image a match term compares against: its MATCH_TARGETS override, else IMAGE_SIM_TARGET."""
    return MATCH_TARGETS.get(term) or IMAGE_SIM_TARGET


def match_weights() -> dict[str, float]:
    """Weight per pixel-level match term, 0 where the term has no target image."""
    weights = {"rgb": IMAGE_SIM_WEIGHT, "gray": IMAGE_GRAY_WEIGHT, "depth": IMAGE_DEPTH_WEIGHT, "edges": IMAGE_EDGE_WEIGHT,
               "color_stats": IMAGE_COLOR_STATS_WEIGHT, "edge_stats": IMAGE_EDGE_STATS_WEIGHT,
               "lines": IMAGE_LINES_WEIGHT}
    return {term: w if match_target(term) is not None else 0.0 for term, w in weights.items()}


UNFRAMED_TERMS = ("color_stats", "edge_stats")  # distributions: compared against the target image as is


class ImageMatch:
    """Pixel-level matching of the bare canvas to each term's target image (match_target):
    rgb, gray, depth, edge, color / edge stats and line terms. Target maps are computed
    once; None when every term is off."""

    @classmethod
    def create(cls, device) -> 'ImageMatch | None':
        return cls(device) if any(match_weights().values()) else None

    def __init__(self, device):
        self.w = match_weights()
        loaded: dict[tuple[str, bool], torch.Tensor] = {}  # by (path, framed): terms often share one image
        self.targets = {}
        for term, w in self.w.items():
            if w:
                # the stats terms ignore layout: their target is the whole image, unframed
                key = (match_target(term), term not in UNFRAMED_TERMS)
                if key not in loaded:
                    loaded[key] = (load_image(key[0], CANVAS_RESOLUTION) if key[1]
                                   else load_image_unframed(key[0], CANVAS_RESOLUTION)).to(device)
                self.targets[term] = loaded[key]
        self.depth = DepthEstimator.get(DEPTH_MODEL, device) if self.w["depth"] else None
        on = lambda term, fn: fn(self.targets[term]) if self.w[term] else None
        with torch.no_grad():
            self.gray = on("gray", luminance)
            self.edges = on("edges", edge_map)
            self.edge_orientations = on("edge_stats", edge_orientation_maps)
            self.lines = on("lines", line_transform)
            self.lines_blurred = blur_lines(self.lines) if self.lines is not None else None
            self.depth_target = on("depth", lambda t: normalize_depth(self.depth(t)))
        # The canvas' maps from the last __call__ (detached), for previews.
        self.last_edges: torch.Tensor | None = None
        self.last_lines: torch.Tensor | None = None
        self.last_depth: torch.Tensor | None = None

    @staticmethod
    def stat_key(term: str) -> str:
        """The stats entry holding a term's error."""
        return "match " + term.replace("_", " ")

    def axes(self) -> list[dict]:
        """One entry per active term (its error is stats[stat_key(term)]), for the page."""
        return [{"kind": "match", "term": term, "label": term.replace("_", " "),
                 "image": match_target(term), "weight": w}
                for term, w in self.w.items() if w]

    def __call__(self, canvas: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        loss = canvas.new_zeros(())
        stats = {}
        if self.w["rgb"]:
            err = F.mse_loss(canvas, self.targets["rgb"])
            loss = loss + self.w["rgb"] * err
            stats["match rgb"] = err.item()
        if self.w["gray"]:
            err = F.mse_loss(luminance(canvas), self.gray)
            loss = loss + self.w["gray"] * err
            stats["match gray"] = err.item()
        if self.w["edges"]:
            edges = edge_map(canvas)
            err = F.mse_loss(edges, self.edges)
            loss = loss + self.w["edges"] * err
            stats["match edges"] = err.item()
            self.last_edges = edges.detach()
        if self.w["color_stats"]:
            err = color_stats_error(canvas, self.targets["color_stats"])
            loss = loss + self.w["color_stats"] * err
            stats["match color stats"] = err.item()
        if self.w["edge_stats"]:
            err = edge_stats_error(edge_orientation_maps(canvas), self.edge_orientations)
            loss = loss + self.w["edge_stats"] * err
            stats["match edge stats"] = err.item()
        if self.w["lines"]:
            lines = line_transform(canvas)
            err = F.mse_loss(blur_lines(lines), self.lines_blurred)
            loss = loss + self.w["lines"] * err
            stats["match lines"] = err.item()
            self.last_lines = lines.detach()
        if self.depth is not None:
            err, bare_err, bare_depth = self._depth_error(canvas)
            loss = loss + self.w["depth"] * err
            stats["match depth"] = err.item()
            if DEPTH_AUGMENT_VIEWS:
                # Much lower than "match depth" = the bare canvas fools the network again.
                stats["match depth bare"] = bare_err.item()
            self.last_depth = bare_depth.detach()
        return loss, stats

    def _depth_error(self, canvas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(mean error over all views, bare-canvas error, bare-canvas depth). View 0 is the
        bare canvas vs the cached target depth; views 1.. apply one random affine to the
        canvas (plus noise) and the target alike, the target depth recomputed per view."""
        size = canvas.shape[-1]
        canvas_views, target_views = [], []
        for _ in range(DEPTH_AUGMENT_VIEWS):
            params = dict(
                angle=float(torch.empty(1).uniform_(-DEPTH_AUG_ROTATION, DEPTH_AUG_ROTATION)),
                translate=[round(float(torch.empty(1).uniform_(-1, 1)) * DEPTH_AUG_TRANSLATE * size) for _ in range(2)],
                scale=float(torch.empty(1).uniform_(*DEPTH_AUG_SCALE)), shear=[0.0])
            warp = lambda x: TF.affine(x, interpolation=TF.InterpolationMode.BILINEAR, fill=0.5, **params)
            view = warp(canvas)
            canvas_views.append((view + DEPTH_AUG_NOISE * torch.randn_like(view)).clamp(0, 1))
            target_views.append(warp(self.targets["depth"]))
        depths = normalize_depth(self.depth(torch.cat([canvas, *canvas_views])))  # (1 + V, h, w)
        errors = [F.mse_loss(depths[:1], self.depth_target)]
        if canvas_views:
            with torch.no_grad():
                target_depths = normalize_depth(self.depth(torch.cat(target_views)))
            errors += [F.mse_loss(depths[i + 1 : i + 2], target_depths[i : i + 1]) for i in range(len(canvas_views))]
        return torch.stack(errors).mean(), errors[0], depths[:1]


def fit_kernel(kernel: int, param: torch.Tensor, what: str) -> tuple[int, int]:
    """(width, height) of an odd kernel on a (..., H, W) tensor: reflected edges need its radius
    below the tensor's side, so it shrinks to fit a small tensor (LEVELSET layers, patches, coarse
    pyramid levels), with a warning. (1, 1): the filter can't apply."""
    h, w = param.shape[-2:]
    kw, kh = min(kernel, 2 * w - 1), min(kernel, 2 * h - 1)
    if (kw, kh) != (kernel, kernel):
        result = "not applied" if kw == kh == 1 else f"using {kw}x{kh}"
        warnings.warn(f"{what} kernel {kernel}x{kernel} is too large for a {w}x{h} tensor: {result}", stacklevel=3)
    return kw, kh


def gaussian_blur_(param: torch.Tensor, sigma: float, strength: float = 1.0) -> None:
    """In-place blend of a (1, C, H, W) parameter towards its blur, outside autograd."""
    kw, kh = fit_kernel(max(3, int(2 * round(3 * sigma) + 1)), param, "Blur")
    if kw == kh == 1:
        return
    with torch.no_grad():
        param.lerp_(TF.gaussian_blur(param, [kw, kh], [sigma, sigma]), strength)


def median_blur_(param: torch.Tensor, kernel: int, weight: float = 1.0) -> None:
    """In-place blend of a (1, C, H, W) parameter towards its median filter (kernel x kernel
    window, reflected edges), outside autograd. On the float tensor itself: no 8-bit round trip."""
    kw, kh = fit_kernel(kernel, param, "Median")
    if kw == kh == 1:
        return
    with torch.no_grad():
        padded = F.pad(param, (kw // 2, kw // 2, kh // 2, kh // 2), mode="reflect")
        windows = padded.unfold(2, kh, 1).unfold(3, kw, 1)  # (1, C, H, W, kh, kw)
        param.lerp_(windows.flatten(-2).median(dim=-1).values, weight)


# ── Fields: optimizable tensor -> (1, 3, H, W) canvas in [0, 1] ────────────────

class PixelField:
    """Sigmoid of a Laplacian pyramid: level i has resolution src / 2**i, all levels are
    upsampled to src and summed. The finest level starts as logit(START_IMAGE), the rest
    at zero; because coarse levels get the same learning rate over far fewer pixels, the
    optimization favours broad, low-frequency changes."""

    def __init__(self, gray: bool, device):
        self.gray = gray
        start = initial_image(PIXEL_SRC_RESOLUTION, gray).to(device)
        w, h = PIXEL_SRC_RESOLUTION
        self.levels = []
        for i in range(PIXEL_PYRAMID_LEVELS):
            lw, lh = max(1, w >> i), max(1, h >> i)
            init = logit(start) if i == 0 else torch.zeros(1, start.shape[1], lh, lw, device=device)
            self.levels.append(init.clone().requires_grad_(True))

    def params(self) -> list[dict]:
        """Adam param groups, one per level: coarser levels get relatively more LR."""
        last = len(self.levels) - 1
        return [{"params": [level], "lr": LEARNING_RATE * PIXEL_PYRAMID_LR_DECAY ** (last - i)}
                for i, level in enumerate(self.levels)]

    def blurrable(self) -> list[torch.Tensor]:
        return self.levels[:1]

    def image(self) -> torch.Tensor:
        """(1, C, H, W) the field on the canvas, before PIXEL_CONTRAST."""
        x = sum(resize(level, PIXEL_SRC_RESOLUTION) for level in self.levels)
        return torch.sigmoid(resize(x, CANVAS_RESOLUTION))

    def render(self) -> torch.Tensor:
        img = contrast_curve(self.image(), PIXEL_CONTRAST)
        return img.expand(-1, 3, -1, -1) if self.gray else img

    def regularized(self) -> torch.Tensor:
        return self.render()

    def penalty(self, canvas: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """PIXELS_GRAY with PIXEL_COVERAGE: pulls the canvas' mean brightness to it."""
        if not self.gray or PIXEL_COVERAGE is None:
            return canvas.new_zeros(()), {}
        mean = canvas.mean()
        return PIXEL_COVERAGE_WEIGHT * (mean - PIXEL_COVERAGE) ** 2, {"coverage": mean.item()}

    def tensors(self) -> dict[str, torch.Tensor]:
        """The canvas-sized image before the contrast sigmoid ("rgb" / "gray") and, with
        PIXEL_CONTRAST on, after it ("rgb_sigmoid" / "gray_sigmoid"), detached."""
        name = "gray" if self.gray else "rgb"
        with torch.no_grad():
            img = self.image()
            out = {name: img}
            if PIXEL_CONTRAST:
                out[f"{name}_sigmoid"] = contrast_curve(img, PIXEL_CONTRAST)
        return out

    def snapshot(self, folder: Path) -> list[str]:
        """The field's own outputs into folder (the canvas is saved anyway): the image at
        PIXEL_SRC_RESOLUTION, before resampling to the canvas, and tensors() (save_tensors).
        -> the file names."""
        with torch.no_grad():
            img = torch.sigmoid(sum(resize(level, PIXEL_SRC_RESOLUTION) for level in self.levels))
        cv2.imwrite(str(folder / "field.png"), to_bgr(img.expand(-1, 3, -1, -1)))
        return ["field.png", *save_tensors(folder, self.tensors())]


class ShapeField:
    """The shapes composited in the order the curves appear in the .3dm (later on top), over a
    base layer: BACKGROUND_COLOR (per BACKGROUND_OPTIMIZE) and BACKGROUND_IMAGE. What a shape
    optimizes is self.colors, logits, per SHAPE_MODE: "color" (S, 3) its RGB; "monochrome" (S,)
    an amount mixing SHAPE_COLOR_1 (0) to SHAPE_COLOR_2 (1); "masking" (S,) its opacity: the
    largest under each pixel (in any order) is how much SHAPE_COLOR_1 covers the base layer
    there. Masking also has penalty(): an opacity budget and hardening."""

    def __init__(self, device):
        self.src = SHAPE_SRC
        pattern = Pattern.from_rhino(SHAPE_SRC, img_size=CANVAS_RESOLUTION, img_border=SHAPE_BORDER)
        self.masks = torch.from_numpy(np.stack(pattern.masks)).to(device)[:, None, None]  # (S, 1, 1, H, W)
        n = self.masks.shape[0]
        print(f"[shapes] {n} closed curves from {Path(SHAPE_SRC).name}")
        self.size = (self.masks.shape[-1], self.masks.shape[-2])
        self.image = background_image(self.size, False, device)
        self.mode = SHAPE_MODE
        self.bg = solid_color(BACKGROUND_COLOR, False, device)
        self.color1 = solid_color(SHAPE_COLOR_1, False, device).requires_grad_(
            SHAPE_COLOR_1_OPTIMIZE and self.mode != "color")
        self.color2 = solid_color(SHAPE_COLOR_2, False, device).requires_grad_(
            SHAPE_COLOR_2_OPTIMIZE and self.mode == "monochrome")
        self.step = 0
        self.harden_steps = SHAPE_HARDEN_STEPS if self.mode == "masking" and SHAPE_HARDEN_WEIGHT > 0 else 0
        self.last_opacity = None

        m = self.masks[:, 0, 0]  # (S, H, W)
        under = lambda img: (m[:, None] * img).sum((-1, -2)) / m.sum((-1, -2))[:, None].clamp_min(1e-6)  # (S, C)
        from_start = CANVAS_INIT == "start" and START_IMAGE is not None
        if from_start:
            # Each shape starts as the mean colour of START_IMAGE under it.
            colors = under(load_image(START_IMAGE, CANVAS_RESOLUTION).to(device)[0])
        if self.mode == "color":
            if not from_start:
                colors = (torch.sigmoid(CANVAS_NOISE_STD * torch.randn(n, 3)).to(device) if CANVAS_INIT == "noise"
                          else torch.full((n, 3), 0.5, device=device))
            self.colors = logit(colors).requires_grad_(True)
        elif from_start:
            # The start colour projected onto the line between the shape's two ends.
            low = (under(backdrop(unit(self.bg), self.image, self.size).detach()[0]) if self.mode == "masking"
                   else self.color1.detach())
            high = (self.color1 if self.mode == "masking" else self.color2).detach()
            axis = high - low
            amounts = ((colors - low) * axis).sum(-1) / (axis * axis).sum(-1).clamp_min(1e-6)
            self.colors = logit(amounts.clamp(0.02, 0.98)).requires_grad_(True)
        elif CANVAS_INIT == "noise":
            self.colors = (CANVAS_NOISE_STD * torch.randn(n)).to(device).requires_grad_(True)
        else:
            self.colors = torch.zeros(n, device=device, requires_grad=True)

    def params(self) -> list[dict]:
        """Adam param groups, self.colors first (hardening changes its learning rate)."""
        others = [c for c in (self.bg, self.color1, self.color2) if c.requires_grad]
        return [{"params": [self.colors], "lr": LEARNING_RATE},
                *([{"params": others, "lr": LEARNING_RATE}] if others else [])]

    def blurrable(self) -> list[torch.Tensor]:
        return []

    def shape_colors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(S, 3) the colour of every shape (masking: its opacity as gray, 0 = transparent, 1 =
        opaque; the canvas is composited in render), (3,) the background's."""
        bg = unit(self.bg)
        amounts = torch.sigmoid(self.colors)
        if self.mode == "color":
            return amounts, bg
        if self.mode == "masking":
            return amounts[:, None].expand(-1, 3), bg
        low, high = unit(self.color1), unit(self.color2)
        return low + (high - low) * amounts[:, None], bg

    def render(self) -> torch.Tensor:
        img = backdrop(unit(self.bg), self.image, self.size)
        if self.mode == "masking":
            # The largest opacity per pixel, not one over the other: neighbours' masks both reach ~1
            # along their shared edge, so stacking them would reinforce into lines and nodes.
            alpha = (torch.sigmoid(self.colors).view(-1, 1, 1, 1, 1) * self.masks).amax(0)  # (1, 1, H, W)
            self.last_opacity = alpha
            return img * (1 - alpha) + unit(self.color1).view(1, 3, 1, 1) * alpha
        colors, _ = self.shape_colors()
        for s in range(self.masks.shape[0]):
            alpha = self.masks[s]
            img = img * (1 - alpha) + colors[s].view(1, 3, 1, 1) * alpha
        return img

    def regularized(self) -> torch.Tensor:
        return self.render()

    def harden_ramp(self) -> float:
        """0 for the first half of harden_steps, up to 1 at their end, then 1."""
        if not self.harden_steps:
            return 0.0
        half = self.harden_steps / 2
        return min(1.0, max(0.0, (self.step - half) / half))

    def harden_poles(self) -> torch.Tensor:
        """(S,) +1 / -1: the pole hardening pushes each shape to. Without a budget, its nearest.
        With one, the nearest would lose it (under a low budget most shapes sit below 0.5, and
        all go to 0): the most opaque shapes that together cover the budget go to 1, the rest
        to 0. Measured as render composites them: neighbours' masks overlap along their edges."""
        logits = self.colors.detach()
        if SHAPE_OPACITY_BUDGET is None:
            return torch.where(logits >= 0, 1.0, -1.0)
        poles = -torch.ones_like(logits)
        opaque = torch.zeros_like(self.masks[0, 0, 0])
        covered = 0.0
        for s in logits.argsort(descending=True):
            opaque = torch.maximum(opaque, self.masks[s, 0, 0])
            now = opaque.mean().item()
            if (covered + now) / 2 > SHAPE_OPACITY_BUDGET / 100:  # past its midpoint: the closest total
                break
            poles[s], covered = 1.0, now
        return poles

    def penalty(self, canvas: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Masking only. SHAPE_OPACITY_WEIGHT x (mean opacity of the last render - budget)^2, stat
        "opacity" in %; and SHAPE_HARDEN_WEIGHT x harden_ramp() x how far the opacities are from
        their poles, stat "hardening" (0 = all there, 1 = all at 0.5, more = on the wrong side)."""
        loss = canvas.new_zeros(())
        if self.mode != "masking":
            return loss, {}
        opacity = self.last_opacity.mean()
        stats = {"opacity": opacity.item() * 100}
        if SHAPE_OPACITY_BUDGET is not None:
            loss = loss + SHAPE_OPACITY_WEIGHT * (opacity - SHAPE_OPACITY_BUDGET / 100) ** 2
        if self.harden_steps:
            soft = (1 - self.harden_poles() * self.colors / SHAPE_HARDEN_LOGIT).clamp_min(0).square().mean()
            loss = loss + SHAPE_HARDEN_WEIGHT * self.harden_ramp() * soft
            stats["hardening"] = soft.item()
        return loss, stats

    def before_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Adam moves a logit by about its learning rate per step, whatever the weight, so
        during the ramp the opacities get one that can reach SHAPE_HARDEN_LOGIT in time (twice
        over: its first steps are a weak push). Back to LEARNING_RATE once it is at full weight,
        so hard shapes are not shaken by a high rate. Then the next step is counted."""
        ramping = 0 < self.harden_ramp() < 1
        needed = 4 * SHAPE_HARDEN_LOGIT / max(1, self.harden_steps)  # over half of the steps
        optimizer.param_groups[0]["lr"] = max(LEARNING_RATE, needed) if ramping else LEARNING_RATE
        self.step += 1

    def snapshot(self, folder: Path) -> list[str]:
        """The pattern with every curve in its colour (pattern_colored.3dm), the colours
        (shapes.json: RGB 0..255 per shape in file order, and the background) and each
        shape's mask on the canvas (masks/mask_<i>.png, antialiased). Monochrome adds its two
        colours and each shape's amount (0 = color 1, 1 = color 2). Masking: the colours are each
        shape's opacity as gray (0 = transparent, 255 = opaque); adds its one colour and the
        opacities. -> the file names."""
        to_rgb = lambda c: (unit(c) * 255).round().cpu().numpy().astype(int).tolist()
        with torch.no_grad():
            colors, bg = ((c * 255).round().cpu().numpy().astype(int).tolist() for c in self.shape_colors())
            amounts = [round(a, 4) for a in torch.sigmoid(self.colors).cpu().tolist()] if self.mode != "color" else None
            extra = ({"mode": self.mode} if self.mode == "color" else
                     {"mode": self.mode, "color_1": to_rgb(self.color1), "color_2": to_rgb(self.color2), "amounts": amounts}
                     if self.mode == "monochrome" else
                     {"mode": self.mode, "color": to_rgb(self.color1), "opacities": amounts})
            masks = (self.masks[:, 0, 0] * 255).round().byte().cpu().numpy()
        save_colored_rhino(self.src, folder / "pattern_colored.3dm", colors)
        (folder / "shapes.json").write_text(json.dumps(
            {"source": str(self.src), "background": bg, "colors": colors, **extra}, indent=1))
        (folder / "masks").mkdir(exist_ok=True)
        for i, mask in enumerate(masks):
            cv2.imwrite(str(folder / "masks" / f"mask_{i:04d}.png"), mask)
        return ["pattern_colored.3dm", "shapes.json", f"masks/ ({len(masks)})"]


def light_direction(azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
    """(N, 3) unit vectors towards the lights, in image coordinates (x right, y down, z out
    of the screen), from degrees: azimuth counter-clockwise from +x as seen on screen."""
    az, el = torch.deg2rad(azimuth), torch.deg2rad(elevation)
    return torch.stack([el.cos() * az.cos(), -el.cos() * az.sin(), el.sin()], dim=-1)


class ReliefField:
    """A scalar height per shape of SHAPE_SRC (the background at 0), composited in file order
    (later on top). Smoothed by RELIEF_SMOOTH and scaled to RELIEF_DEPTH px, the heights are a
    displacement map whose Sobel derivatives give the normals n = normalize(-dz/dx, -dz/dy, 1).
    Each light l shades the materials (the backdrop / RELIEF_COLOR, blended by coverage) by
    LIGHT_AMBIENT + (1 - LIGHT_AMBIENT) max(0, n . l).

    render() is the first fixed light's image; views() the batch the models score -- each light's
    image (LIGHT_COUNT fixed, LIGHT_RANDOM_COUNT random), augmented separately. It shades the
    surface of the last render(). Masks are fixed, so a composite is linear in the heights: each
    pixel keeps its top K (shape, weight) pairs, K = the most shapes any pixel sees -- exact,
    and cheap however many shapes there are."""

    def __init__(self, device):
        self.src = SHAPE_SRC
        pattern = Pattern.from_rhino(SHAPE_SRC, img_size=CANVAS_RESOLUTION, img_border=SHAPE_BORDER)
        self.pixels_per_unit = pattern.pixels_per_unit
        masks = torch.from_numpy(np.stack(pattern.masks)).float()  # (S, H, W), CPU
        n, h, w = masks.shape
        self.size = (w, h)
        weights = torch.empty_like(masks)
        uncovered = torch.ones(h, w)
        for s in reversed(range(n)):
            weights[s] = masks[s] * uncovered
            uncovered = uncovered * (1 - masks[s])
        k = max(1, int((weights > 0).sum(0).max()))
        top = weights.topk(k, dim=0)
        self.index = top.indices.to(device)                  # (K, H, W)
        self.weight = top.values.to(device)[..., None]       # (K, H, W, 1)
        self.coverage = (1 - uncovered).to(device)[None, None]  # (1, 1, H, W): how much any shape covers
        print(f"[relief] {n} closed curves from {Path(SHAPE_SRC).name}, <= {k} per pixel")

        if CANVAS_INIT == "start" and START_IMAGE is not None:
            # Each shape starts as the mean luminance of START_IMAGE over its visible area.
            start = luminance(load_image(START_IMAGE, CANVAS_RESOLUTION))[0, 0]
            heights = (weights * start).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1e-6)
        elif CANVAS_INIT == "noise":
            heights = torch.sigmoid(CANVAS_NOISE_STD * torch.randn(n))
        else:
            heights = torch.full((n,), 0.5)
        self.heights = logit(heights.clamp(0.02, 0.98).to(device))[:, None].requires_grad_(True)  # (S, 1)
        self.image = background_image(self.size, False, device)
        self.bg = solid_color(BACKGROUND_COLOR, False, device)
        self.material = torch.tensor(RELIEF_COLOR, dtype=torch.float32, device=device).requires_grad_(RELIEF_COLOR_OPTIMIZE)
        count = max(1, LIGHT_COUNT)
        azimuths = LIGHT_AZIMUTH + 360.0 * torch.arange(count) / count
        self.fixed_lights = light_direction(azimuths, torch.full_like(azimuths, LIGHT_ELEVATION)).to(device)
        self.device = device
        # From the last render(), in its graph:
        self.last_height: torch.Tensor | None = None
        self.last_normals: torch.Tensor | None = None
        self.last_albedo: torch.Tensor | None = None
        self.last_fixed: torch.Tensor | None = None  # (LIGHT_COUNT, 3, H, W): image 0 is the canvas itself
        self.last_lit: torch.Tensor | None = None    # every light of the last render() / views(), detached

    def params(self) -> list[torch.Tensor]:
        return [self.heights] + [c for c in (self.bg, self.material) if c.requires_grad]

    def blurrable(self) -> list[torch.Tensor]:
        return []

    def height(self) -> torch.Tensor:
        """(1, 1, H, W) composited heights in [0, 1], before smoothing."""
        return (torch.sigmoid(self.heights)[self.index] * self.weight).sum(0).permute(2, 0, 1)[None]

    def normals(self, height: torch.Tensor) -> torch.Tensor:
        """(1, 3, H, W) unit surface normals of a (1, 1, H, W) height map."""
        if RELIEF_SMOOTH > 0:
            k = max(3, int(2 * math.ceil(3 * RELIEF_SMOOTH) + 1))
            height = TF.gaussian_blur(height, [k, k], [RELIEF_SMOOTH, RELIEF_SMOOTH])
        padded = F.pad(height, (1, 1, 1, 1), mode="replicate")
        kx = height.new_tensor(_SOBEL_X).view(1, 1, 3, 3)
        gx, gy = F.conv2d(padded, kx), F.conv2d(padded, kx.transpose(-1, -2))
        scale = RELIEF_DEPTH / 8  # Sobel's weights sum to 8 per unit slope
        return F.normalize(torch.cat([-scale * gx, -scale * gy, torch.ones_like(gx)], dim=1), dim=1)

    def shade(self, lights: torch.Tensor) -> torch.Tensor:
        """(L, 3, H, W): the last render()'s surface under lights (L, 3)."""
        diffuse = torch.einsum("lc,chw->lhw", lights, self.last_normals[0]).clamp_min(0)[:, None]
        return self.last_albedo * (LIGHT_AMBIENT + (1 - LIGHT_AMBIENT) * diffuse)

    def render(self) -> torch.Tensor:
        self.last_height = self.height()
        self.last_normals = self.normals(self.last_height)
        backdrop_img = backdrop(unit(self.bg), self.image, self.size)
        self.last_albedo = backdrop_img + (unit(self.material).view(1, 3, 1, 1) - backdrop_img) * self.coverage
        fixed = self.shade(self.fixed_lights)
        canvas = fixed[:1]
        self.last_fixed = torch.cat([canvas, fixed[1:]])  # through `canvas`: its gradient sees the models too
        self.last_lit = self.last_fixed.detach()
        return canvas

    def fixed_renders(self) -> torch.Tensor:
        """The fixed lights' images of the last render(): what the image match terms compare."""
        return self.last_fixed

    def views(self) -> torch.Tensor:
        images = self.last_fixed
        if LIGHT_RANDOM_COUNT > 0:
            n = LIGHT_RANDOM_COUNT
            lo, hi = LIGHT_RANDOM_ELEVATION
            lights = light_direction(torch.rand(n) * 360, lo + (hi - lo) * torch.rand(n)).to(self.device)
            images = torch.cat([images, self.shade(lights)])
        self.last_lit = images.detach()
        return torch.cat([augment(image[None]) for image in images])

    def regularized(self) -> torch.Tensor:
        # The height map of the last render: a lit image is never symmetric or flat where the surface is.
        return self.last_height

    def previews(self) -> dict[str, torch.Tensor]:
        """(1, 3, H, W) images for the page, by name."""
        with torch.no_grad():
            height = self.height()
            return {"height": height.expand(-1, 3, -1, -1), "normals": self.normals(height) * 0.5 + 0.5}

    def tensors(self) -> dict[str, torch.Tensor]:
        return self.previews() | {"height": self.height().detach()}

    def values(self) -> dict:
        """The current heights (per shape, in file order) and material colours."""
        with torch.no_grad():
            return {"heights": [round(h, 4) for h in torch.sigmoid(self.heights)[:, 0].tolist()],
                    "background": _to_hex(unit(self.bg).tolist()), "material": _to_hex(unit(self.material).tolist())}

    def snapshot(self, folder: Path) -> list[str]:
        """pattern_colored.3dm: the pattern with every curve gray by its height, and a layer
        "extrusions" of the curves as capped meshes in the same grays, their heights the render's
        (RELIEF_DEPTH px in model units); the heights and colours (relief.json) and the maps
        (height.npy / .png, normals.npy / .png). -> the file names."""
        with torch.no_grad():
            heights = torch.sigmoid(self.heights)[:, 0].cpu().numpy()
        grays = (heights * 255).round().astype(int).tolist()
        depth = RELIEF_DEPTH / self.pixels_per_unit
        save_colored_rhino(self.src, folder / "pattern_colored.3dm", [[g, g, g] for g in grays],
                           extrusion_heights=(heights * depth).tolist())
        (folder / "relief.json").write_text(json.dumps(
            {"source": str(self.src), "depth": RELIEF_DEPTH, "depth_model_units": depth, **self.values()}, indent=1))
        return ["pattern_colored.3dm", "relief.json", *save_tensors(folder, self.tensors())]


class DistortionField:
    """START_IMAGE resampled through identity + offset. The offset (in pixels, isotropic) is a
    sum over DISTORTION_PYRAMID_LEVELS fields, level i at DISTORTION_FIELD_RESOLUTION / 2**i,
    each bicubically upsampled to the canvas and bounded by tanh to its cap (see
    DISTORTION_OFFSET_MODE)."""

    def __init__(self, device):
        self.image = load_image(START_IMAGE, CANVAS_RESOLUTION).to(device)
        cw, ch = CANVAS_RESOLUTION
        w, h = DISTORTION_FIELD_RESOLUTION
        self.levels, caps = [], []
        for i in range(DISTORTION_PYRAMID_LEVELS):
            lw, lh = max(2, w >> i), max(2, h >> i)
            self.levels.append(torch.zeros(1, 2, lh, lw, device=device, requires_grad=True))
            if DISTORTION_OFFSET_MODE == "resolution":
                caps.append((DISTORTION_MAX * (cw - 1) / (lw - 1), DISTORTION_MAX * (ch - 1) / (lh - 1)))
            else:
                caps.append((DISTORTION_MAX / 100 * max(cw, ch),) * 2)
        self.caps = torch.tensor(caps, device=device).view(-1, 1, 2, 1, 1)  # (L, 1, 2, 1, 1) px, (x, y)
        ys = torch.linspace(-1, 1, ch, device=device)
        xs = torch.linspace(-1, 1, cw, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.identity = torch.stack([gx, gy], dim=-1)[None]  # (1, H, W, 2), grid_sample's (x, y)
        # px -> grid_sample's normalized units (align_corners=True: -1..1 spans size - 1 px)
        self.to_grid = torch.tensor([2 / (cw - 1), 2 / (ch - 1)], device=device).view(1, 2, 1, 1)
        self.long_side = max(cw, ch)

    def params(self) -> list[dict]:
        """Adam param groups, one per level: coarser levels get relatively more LR."""
        last = len(self.levels) - 1
        return [{"params": [level], "lr": LEARNING_RATE * DISTORTION_PYRAMID_LR_DECAY ** (last - i)}
                for i, level in enumerate(self.levels)]

    def blurrable(self) -> list[torch.Tensor]:
        return self.levels[:1]

    def offsets(self) -> torch.Tensor:
        """(1, 2, H, W) displacement in canvas pixels, (x, y)."""
        return sum(cap * torch.tanh(resize(level, CANVAS_RESOLUTION)) for cap, level in zip(self.caps, self.levels))

    def render(self) -> torch.Tensor:
        grid = self.identity + (self.offsets() * self.to_grid).permute(0, 2, 3, 1)
        if DISTORTION_EDGE_MODE == "gray":
            return F.grid_sample(self.image - 0.5, grid, mode="bilinear", padding_mode="zeros", align_corners=True) + 0.5
        return F.grid_sample(self.image, grid, mode="bilinear", padding_mode=DISTORTION_EDGE_MODE, align_corners=True)

    def regularized(self) -> torch.Tensor:
        # In units of half the long side, so TV_WEIGHT keeps its scale across canvas sizes.
        return self.offsets() * (2 / self.long_side)

    def penalty(self, canvas: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Fold-over penalty (DISTORTION_PREVENT_FOLDS): the mapping p -> p + d(p) folds where
        det(I + grad d) <= 0; pushes the determinant above DISTORTION_FOLD_MARGIN. The fraction
        of folded pixels is reported either way."""
        d = self.offsets()[0]                          # (2, H, W) px
        ddx = d[:, :-1, 1:] - d[:, :-1, :-1]           # d/dx of (dx, dy)
        ddy = d[:, 1:, :-1] - d[:, :-1, :-1]           # d/dy of (dx, dy)
        det = (1 + ddx[0]) * (1 + ddy[1]) - ddx[1] * ddy[0]
        stats = {"folds": (det <= 0).float().mean().item()}
        if not DISTORTION_PREVENT_FOLDS:
            return d.new_zeros(()), stats
        return DISTORTION_FOLD_WEIGHT * F.relu(DISTORTION_FOLD_MARGIN - det).pow(2).mean(), stats

    def snapshot(self, folder: Path) -> list[str]:
        """The displacement: displacement.png (hue = direction, see flow_to_bgr) and
        offsets.npy ((2, H, W) float32, canvas pixels, x then y). -> the file names."""
        with torch.no_grad():
            offsets = self.offsets()
        cv2.imwrite(str(folder / "displacement.png"), flow_to_bgr(offsets))
        np.save(folder / "offsets.npy", offsets[0].float().cpu().numpy())
        return ["displacement.png", "offsets.npy"]


class SurfaceField:
    """A parametric surface (SURFACE, see parametric_surface): render() is the contact sheet
    of the cameras' un-jittered views at CANVAS_RESOLUTION, views() the batch the models
    score -- every camera (render.views_per_camera times), jittered, with its light, plus
    TRANSFORM_NOISE. Parameters get LEARNING_RATE x their own scale (parametric_surface.Param.lr)."""

    def __init__(self, device):
        self.cfg = parametric_surface.normalize_config(SURFACE)
        self.device = device
        self.model = parametric_surface.SurfaceModel(self.cfg, device)
        self.renderer = parametric_surface.make_renderer(self.cfg, device)
        if BACKGROUND_IMAGE is not None:
            self.renderer.set_background_image(lambda size: background_image(size, False, device),
                                               BACKGROUND_IMAGE_MULTIPLY)
        self.fixed_views = parametric_surface.make_views(self.cfg, False, device)
        self.last_canvas: torch.Tensor | None = None
        groups = self.model.param_groups()
        print(f"[surface] {self.cfg['base']['type']} + {len(self.cfg['deformation']['items'])} "
              f"{self.cfg['deformation']['type']}, {len(self.cfg['cameras'])} cameras, "
              f"{sum(t.numel() for t, _ in groups)} optimized values")
        if not groups and not self.renderer.background.requires_grad:
            raise ValueError("SURFACE: nothing to optimize (no parameter is set to optimize)")

    def params(self) -> list[dict]:
        background = [{"params": [self.renderer.background]}] if self.renderer.background.requires_grad else []
        return [*({"params": [t], "lr": LEARNING_RATE * scale} for t, scale in self.model.param_groups()), *background]

    def blurrable(self) -> list[torch.Tensor]:
        return []

    def render(self) -> torch.Tensor:
        self.last_canvas = parametric_surface.render_sheet(self.model, self.renderer, self.fixed_views,
                                                           CANVAS_RESOLUTION)
        return self.last_canvas

    def views(self) -> torch.Tensor:
        views = parametric_surface.make_views(self.cfg, True, self.device)
        batch = self.renderer.render(self.model, views, CANVAS_RESOLUTION)
        if TRANSFORM_NOISE > 0:
            batch = batch + TRANSFORM_NOISE * torch.randn_like(batch)
        return batch.clamp(0, 1)

    def regularized(self) -> torch.Tensor:
        # The sheet run_pole just rendered (TV of the shading), not a second render.
        return self.last_canvas if self.last_canvas is not None else self.render()

    def values(self) -> dict:
        """The current parameters, in SURFACE's format."""
        return self.model.values()

    def snapshot(self, folder: Path) -> list[str]:
        """surface.json (the config with the current parameters) and surface.obj (the mesh)."""
        cfg = json.loads(json.dumps(self.cfg))
        values = self.values()
        cfg["base"].update(values["base"])
        cfg["deformation"].update(values["deformation"])
        cfg["material"]["background"] = _to_hex(self.renderer.background_color().tolist())
        (folder / "surface.json").write_text(json.dumps(cfg, indent=1))
        self.model.write_obj(folder / "surface.obj")
        return ["surface.json", "surface.obj"]


def texture_start(cfg: dict, path: str | None, init: str, fit: str | None = None) -> torch.Tensor:
    """(1, C, H, W) a REVOLUTION texture's start, per a CANVAS_INIT value: the image framed for the
    texture's own aspect, not the canvas', or mid grey; or noise as initial_image's."""
    w, h = cfg["resolution"]
    gray = cfg["monochrome"]
    if init == "noise":
        return torch.sigmoid(CANVAS_NOISE_STD * torch.randn(1, 1 if gray else 3, h, w))
    if init != "start" or path is None:
        return torch.full((1, 1 if gray else 3, h, w), 0.5)
    return TF.to_tensor(fit_image(Image.open(path).convert("L" if gray else "RGB"), (w, h), fit, (w, h))).unsqueeze(0)


class RevolutionField:
    """A surface of revolution (REVOLUTION, see surface_of_revolution), its profile's parameters and
    optionally its texture optimized. The texture is the sigmoid of a Laplacian pyramid, as PixelField,
    wrapping around at the seam. render() is the view at canvas_angle and evaluates the geometry and
    texture once for the step; views() reuses them: the geometry is the same from every angle, only
    the uvs turn. Without a texture every angle looks the same, so the models just see the canvas,
    augmented."""

    def __init__(self, device):
        self.cfg = surface_of_revolution.normalize_config(REVOLUTION)
        self.model = surface_of_revolution.RevolutionModel(self.cfg, CANVAS_RESOLUTION, device)
        self.profile = self.model.profile
        self.device = device
        texture = self.cfg["texture"]
        self.textured = texture["enabled"]
        self.levels = []
        if self.textured:
            start = texture_start(texture, START_IMAGE, CANVAS_INIT).to(device)
            w, h = texture["resolution"]
            self.texture_size = (w, h)
            for i in range(texture["levels"]):
                init = (logit(start) if i == 0
                        else torch.zeros(1, start.shape[1], max(1, h >> i), max(4, w >> i), device=device))
                self.levels.append(init.clone().requires_grad_(True))
        self.color = parametric_surface.hex_color(self.cfg["color"], device).clone().requires_grad_(
            self.cfg["optimize_color"] and not self.textured)
        self.bg = solid_color(BACKGROUND_COLOR, False, device)
        self.image = background_image(CANVAS_RESOLUTION, False, device)
        # From the last render(), in its graph:
        self.last_geometry = None
        self.last_texture: torch.Tensor | None = None
        self.last_background: torch.Tensor | None = None
        self.last_canvas: torch.Tensor | None = None
        self.last_views: torch.Tensor | None = None  # the angles of the last views(), detached
        if not self.params():
            raise ValueError("REVOLUTION: nothing to optimize (profile frozen, no texture, colours frozen)")
        print(f"[revolution] {self.cfg['mode']} profile" + (f", texture {w}x{h}" if self.textured else ""))

    def params(self) -> list[dict]:
        last = len(self.levels) - 1
        lr_decay = self.cfg["texture"]["lr_decay"]
        groups = [{"params": [level], "lr": LEARNING_RATE * lr_decay ** (last - i)} for i, level in enumerate(self.levels)]
        if self.profile.tensors:
            groups.append({"params": self.profile.tensors, "lr": LEARNING_RATE * self.cfg["shape_lr"]})
        colors = [c for c in (self.color, self.bg) if c.requires_grad]
        return groups + ([{"params": colors, "lr": LEARNING_RATE}] if colors else [])

    def blurrable(self) -> list[torch.Tensor]:
        return self.levels[:1]

    def texture(self) -> torch.Tensor:
        """(1, C, H, W) in [0, 1], at the texture's resolution."""
        w, h = self.texture_size
        total = 0
        for level in self.levels:
            if level.shape[-2:] == (h, w):
                total = total + level
                continue
            # Upsampled with neighbours from around the seam, so that it stays closed.
            margin = round(2 * w / level.shape[-1])
            wrapped = F.pad(level, (2, 2, 0, 0), mode="circular")
            up = F.interpolate(wrapped, size=(h, w + 2 * margin), mode="bicubic", align_corners=False)
            total = total + up[..., margin:margin + w]
        return torch.sigmoid(total)

    def view(self, angle: float) -> torch.Tensor:
        """(1, 3, H, W) the last render()'s surface turned by `angle` degrees."""
        geometry = self.last_geometry
        if self.textured:
            albedo = surface_of_revolution.sample_texture(self.last_texture, geometry, angle).expand(-1, 3, -1, -1)
        else:
            albedo = unit(self.color).view(1, 3, 1, 1)
        return surface_of_revolution.compose(geometry, albedo, self.last_background)

    def render(self) -> torch.Tensor:
        self.last_geometry = self.model.geometry()
        self.last_texture = self.texture() if self.textured else None
        self.last_background = backdrop(unit(self.bg), self.image)
        self.last_canvas = self.view(self.cfg["canvas_angle"])
        return self.last_canvas

    def views(self) -> torch.Tensor:
        if not self.textured:
            self.last_views = self.last_canvas.detach()
            return augment(self.last_canvas)
        # Evenly around the surface from a random start: every part of the texture is seen each step.
        n = self.cfg["texture"]["views"]
        start = torch.rand(1).item() * 360
        images = [self.view(start + 360 * i / n) for i in range(n)]
        self.last_views = torch.cat(images).detach()
        # Batch size as in the other fields, however many angles.
        count = math.ceil(RANDOM_TRANSFORM_COUNT / n)
        return torch.cat([augment(image, count) for image in images])

    def all_views(self) -> torch.Tensor:
        """(1, 3, H, W) the last render()'s surface at evenly spaced angles from the canvas angle (the
        texture's views a step, at least 4), tiled on the canvas, the canvas angle outlined."""
        n = max(4, self.cfg["texture"]["views"])
        _, _, (tw, th) = mesh_texture.sheet_layout(n, CANVAS_RESOLUTION)
        with torch.no_grad():
            tiles = torch.cat([self.view(self.cfg["canvas_angle"] + 360 * i / n) for i in range(n)])
            tiles = F.interpolate(tiles, size=(th, tw), mode="area")
            return mesh_texture.tile_sheet(tiles, CANVAS_RESOLUTION, self.last_background, 0)

    def regularized(self) -> torch.Tensor:
        # The texture's own TV, not its projection, which squeezes it towards the silhouette.
        return self.last_texture if self.textured else self.last_canvas

    def before_step(self, optimizer) -> None:
        # A formula can have an undefined gradient where its value is fine (abs at 0, a clamped domain).
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.nan_to_num_(0.0, 0.0, 0.0)

    def previews(self) -> dict[str, torch.Tensor]:
        """(1, 3, H, W) images for the page, by name: the maps, the texture."""
        with torch.no_grad():
            geometry = self.last_geometry if self.last_geometry is not None else self.model.geometry()
            radius = self.profile.radius(self.model.t)[0].max().item()
            maps = surface_of_revolution.geometry_maps(geometry, radius)
            out = {f"rev_{k}": v.expand(-1, 3, -1, -1) for k, v in maps.items()}
            if self.textured:
                out["rev_texture"] = self.texture().expand(-1, 3, -1, -1)
        return out

    def tensors(self) -> dict[str, torch.Tensor]:
        """The canvas-sized maps at angle 0, in [0, 1]. depth: 1 = the largest radius."""
        return {k[4:]: v for k, v in self.previews().items() if k != "rev_texture"}

    def values(self) -> dict:
        """The current parameters and colours, flat, for the page and run.json."""
        with torch.no_grad():
            out = self.profile.values()
            if not self.textured:
                out["color"] = _to_hex(unit(self.color).tolist())
            if self.bg.requires_grad:
                out["background"] = _to_hex(unit(self.bg).tolist())
        return out

    def export(self, folder: Path) -> list[str]:
        """revolution.3dm (surface_of_revolution.write_rhino), texture.png (with a texture),
        revolution.json (the config and the current parameters). -> the file names."""
        folder.mkdir(parents=True, exist_ok=True)
        files, texture = ["revolution.3dm", "revolution.json"], None
        if self.textured:
            texture = folder / "texture.png"
            with torch.no_grad():
                cv2.imwrite(str(texture), to_bgr(self.texture().expand(-1, 3, -1, -1)))
            files.append("texture.png")
        with torch.no_grad():
            rgb = (unit(self.color) * 255).round().int().tolist()
        notes = {"profile": self.cfg["mode"], "height": self.cfg["height"]}
        if self.cfg["mode"] == "parametric":
            notes["formula"] = self.cfg["parametric"]["formula"]
        surface_of_revolution.write_rhino(folder / "revolution.3dm", self.profile, tuple(rgb), texture,
                                          {**notes, **{k: v for k, v in self.values().items() if k != "color"}})
        (folder / "revolution.json").write_text(json.dumps({"config": self.cfg, "values": self.values()}, indent=1))
        return files

    def snapshot(self, folder: Path) -> list[str]:
        return [*self.export(folder), *save_tensors(folder, self.tensors())]


class MeshTextureField:
    """The texture of a Rhino mesh (MESH_TEXTURE, see mesh_texture), seen from fixed cameras. The
    mesh is prerendered once into G-buffers; every step the texture -- the sigmoid of a Laplacian
    pyramid, as PixelField; wrapping around in u for cylindrical / spherical uvs -- is sampled
    through them and lit. Per mode it is the RGB albedo, a gray albedo, or a height map tilting the
    normals over the material colour (optimized with optimize_color). render() is the canvas view
    under the configured light; views() a random views.per_step of the cameras, each under that
    light jittered by shading.jitter, augmented."""

    def __init__(self, device):
        self.cfg = mesh_texture.normalize_config(MESH_TEXTURE)
        if not self.cfg["path"]:
            raise ValueError("MESH_TEXTURE: no mesh file")
        self.mesh = mesh_texture.load_mesh(self.cfg["path"])
        self.gbuffer = mesh_texture.prerender(self.mesh, self.cfg, CANVAS_RESOLUTION, device)
        self.device = device
        texture = self.cfg["texture"]
        w, h = texture["resolution"]
        self.texture_size = (w, h)
        start = texture_start({"resolution": [w, h], "monochrome": mesh_texture.channels(self.cfg) == 1},
                              START_IMAGE, CANVAS_INIT).to(device)
        self.levels = []
        for i in range(texture["levels"]):
            init = (logit(start) if i == 0
                    else torch.zeros(1, start.shape[1], max(1, h >> i), max(4, w >> i), device=device))
            self.levels.append(init.clone().requires_grad_(True))
        self.color = parametric_surface.hex_color(self.cfg["color"], device).clone().requires_grad_(
            self.cfg["optimize_color"] and self.cfg["mode"] == "bump")
        self.bg = solid_color(BACKGROUND_COLOR, False, device)
        self.image = background_image(CANVAS_RESOLUTION, False, device)
        self.canvas_view = torch.tensor([self.cfg["views"]["canvas_view"]], device=device)
        # From the last render(), in its graph:
        self.last_texture: torch.Tensor | None = None
        self.last_background: torch.Tensor | None = None
        self.last_canvas: torch.Tensor | None = None
        self.last_views: torch.Tensor | None = None  # the images of the last views(), detached
        print(f"[mesh texture] {Path(self.cfg['path']).name}: {self.mesh.describe()}; {self.cfg['mode']}, "
              f"uv {self.cfg['uv']}, texture {w}x{h}, {len(self.gbuffer)} views at x{self.cfg['supersample']}")

    def params(self) -> list[dict]:
        last = len(self.levels) - 1
        lr_decay = self.cfg["texture"]["lr_decay"]
        groups = [{"params": [level], "lr": LEARNING_RATE * lr_decay ** (last - i)} for i, level in enumerate(self.levels)]
        colors = [c for c in (self.color, self.bg) if c.requires_grad]
        return groups + ([{"params": colors, "lr": LEARNING_RATE}] if colors else [])

    def blurrable(self) -> list[torch.Tensor]:
        return self.levels[:1]

    def texture(self) -> torch.Tensor:
        """(1, C, H, W) in [0, 1], at the texture's resolution."""
        w, h = self.texture_size
        total = 0
        for level in self.levels:
            if level.shape[-2:] == (h, w):
                total = total + level
            elif self.gbuffer.wrapped:
                # Upsampled with neighbours from around the seam, so that it stays closed.
                margin = round(2 * w / level.shape[-1])
                wrapped = F.pad(level, (2, 2, 0, 0), mode="circular")
                up = F.interpolate(wrapped, size=(h, w + 2 * margin), mode="bicubic", align_corners=False)
                total = total + up[..., margin:margin + w]
            else:
                total = total + F.interpolate(level, size=(h, w), mode="bicubic", align_corners=False)
        return torch.sigmoid(total)

    def _shade(self, views: torch.Tensor, lights: torch.Tensor | None = None) -> torch.Tensor:
        return mesh_texture.shade(self.gbuffer, self.cfg, views, self.last_texture, unit(self.color),
                                  self.last_background, lights)

    def render(self) -> torch.Tensor:
        self.last_texture = self.texture()
        self.last_background = backdrop(unit(self.bg), self.image)
        self.last_canvas = self._shade(self.canvas_view)
        return self.last_canvas

    def views(self) -> torch.Tensor:
        n = self.cfg["views"]["per_step"]
        views = torch.randperm(len(self.gbuffer), device=self.device)[:n]
        lights = mesh_texture.light_directions(self.gbuffer, self.cfg["shading"], views, self.cfg["shading"]["jitter"])
        images = self._shade(views, lights)
        self.last_views = images.detach()
        # Batch size as in the other fields, however many views.
        count = math.ceil(RANDOM_TRANSFORM_COUNT / n)
        return torch.cat([augment(image[None], count) for image in images])

    def all_views(self) -> torch.Tensor:
        """(1, 3, H, W) every camera's view of the last render()'s texture, under the configured light,
        tiled on the canvas, the canvas view outlined."""
        count = len(self.gbuffer)
        _, _, (tw, th) = mesh_texture.sheet_layout(count, CANVAS_RESOLUTION)
        with torch.no_grad():
            tiles = torch.cat([self._shade(views) for views in torch.arange(count, device=self.device).split(4)])
            tiles = F.interpolate(tiles, size=(th, tw), mode="area")
            return mesh_texture.tile_sheet(tiles, CANVAS_RESOLUTION, self.last_background,
                                           self.cfg["views"]["canvas_view"])

    def regularized(self) -> torch.Tensor:
        # The texture's own TV, not its projection.
        return self.last_texture if self.last_texture is not None else self.texture()

    def previews(self) -> dict[str, torch.Tensor]:
        """(1, 3, H, W) images for the page, by name: the texture, the canvas view's maps."""
        with torch.no_grad():
            maps = self.gbuffer.maps(self.cfg["views"]["canvas_view"])
            out = {f"mesh_{k}": v for k, v in maps.items()}
            out["mesh_texture"] = self.texture().expand(-1, 3, -1, -1)
        return out

    def tensors(self) -> dict[str, torch.Tensor]:
        """The canvas view's maps at the canvas size, in [0, 1] (position: within the bounding box)."""
        return {k[5:]: v for k, v in self.previews().items() if k != "mesh_texture"}

    def values(self) -> dict:
        """The optimized colours, for the page and run.json."""
        with torch.no_grad():
            out = {}
            if self.color.requires_grad:
                out["material"] = _to_hex(unit(self.color).tolist())
            if self.bg.requires_grad:
                out["background"] = _to_hex(unit(self.bg).tolist())
        return out

    def export(self, folder: Path) -> list[str]:
        """texture.png, mesh_texture.3dm (mesh_texture.write_rhino: the mesh with the texture as its
        bitmap or bump map), mesh_texture.json (the config and the colours). -> the file names."""
        folder.mkdir(parents=True, exist_ok=True)
        texture = folder / "texture.png"
        with torch.no_grad():
            cv2.imwrite(str(texture), to_bgr(self.texture().expand(-1, 3, -1, -1)))
            rgb = (unit(self.color) * 255).round().int().tolist()
        notes = {"source": self.cfg["path"], "mode": self.cfg["mode"], "uv": self.cfg["uv"], **self.values()}
        mesh_texture.write_rhino(folder / "mesh_texture.3dm", self.mesh, self.cfg, tuple(rgb), texture, notes)
        (folder / "mesh_texture.json").write_text(json.dumps({"config": self.cfg, "values": self.values()}, indent=1))
        return ["texture.png", "mesh_texture.3dm", "mesh_texture.json"]

    def snapshot(self, folder: Path) -> list[str]:
        return [*self.export(folder), *save_tensors(folder, self.tensors())]

class Gaussians3DField:
    """A scene of 3D Gaussian splats (GAUSSIANS_3D, see gaussian_splats_3d), every parameter of every
    Gaussian optimized, seen by orbiting cameras. render() is the canvas camera's view; views() a
    random views.per_step of the cameras, each moved by up to the config's jitter, augmented."""

    def __init__(self, device):
        self.cfg = gaussian_splats_3d.normalize_config(GAUSSIANS_3D)
        self.views_cfg = self.cfg["views"]
        self.device = device
        self.canvas_camera = gaussian_splats_3d.view_camera(self.views_cfg, self.views_cfg["canvas_view"],
                                                            CANVAS_RESOLUTION)
        start = (load_image(START_IMAGE, CANVAS_RESOLUTION, self.cfg["monochrome"])
                 if CANVAS_INIT == "start" and START_IMAGE is not None else None)
        colors = gaussian_splats_3d.start_colors(start, self.canvas_camera, CANVAS_RESOLUTION, CANVAS_INIT,
                                                 CANVAS_NOISE_STD, self.cfg["monochrome"])
        self.splats = gaussian_splats_3d.Splats(self.cfg, device, colors)
        self.bg = solid_color(BACKGROUND_COLOR, False, device)
        self.image = background_image(CANVAS_RESOLUTION, False, device)
        # From the last render(), in its graph:
        self.last_cov: torch.Tensor | None = None
        self.last_background: torch.Tensor | None = None
        self.last_canvas: torch.Tensor | None = None
        self.last_views: torch.Tensor | None = None  # the images of the last views(), detached
        self.stats: dict = {}
        print(f"[gaussians 3d] {self.cfg['count']} splats, {self.views_cfg['count']} cameras, "
              f"{self.views_cfg['per_step']} a step")

    def params(self) -> list[dict]:
        groups = self.splats.param_groups(LEARNING_RATE)
        return groups + ([{"params": [self.bg], "lr": LEARNING_RATE}] if self.bg.requires_grad else [])

    def blurrable(self) -> list[torch.Tensor]:
        return []

    def _render(self, camera) -> torch.Tensor:
        image, self.stats = self.splats.render(camera, CANVAS_RESOLUTION, self.last_background[0], self.last_cov)
        return image[None]

    def render(self) -> torch.Tensor:
        self.last_cov = self.splats.covariances()
        self.last_background = backdrop(unit(self.bg), self.image)
        self.last_canvas = self._render(self.canvas_camera)
        return self.last_canvas

    def views(self) -> torch.Tensor:
        n = self.views_cfg["per_step"]
        picked = torch.randperm(self.views_cfg["count"])[:n].tolist()
        images = torch.cat([self._render(gaussian_splats_3d.view_camera(self.views_cfg, i, CANVAS_RESOLUTION,
                                                                        self.cfg["jitter"]))
                            for i in picked])
        self.last_views = images.detach()
        # Batch size as in the other fields, however many views.
        count = math.ceil(RANDOM_TRANSFORM_COUNT / n)
        return torch.cat([augment(image[None], count) for image in images])

    def regularized(self) -> torch.Tensor:
        return self.last_canvas if self.last_canvas is not None else self.render()

    def all_views(self) -> torch.Tensor:
        """(1, 3, H, W) every camera's unjittered view, tiled on the canvas, the canvas view outlined."""
        background = self.last_background if self.last_background is not None else backdrop(unit(self.bg), self.image)
        return gaussian_splats_3d.all_views(self.splats, CANVAS_RESOLUTION, background.detach())

    def values(self) -> dict:
        """Summary numbers (and the background, when optimized), for the page and run.json."""
        out = self.splats.values()
        if self.bg.requires_grad:
            with torch.no_grad():
                out["background"] = _to_hex(unit(self.bg).tolist())
        return out

    def export(self, folder: Path) -> list[str]:
        """gaussians.ply (the 3D Gaussian splatting layout), gaussians.3dm (the centres as a point
        cloud in their colours), gaussians.json (the config and the summary). -> the file names."""
        folder.mkdir(parents=True, exist_ok=True)
        gaussian_splats_3d.write_ply(folder / "gaussians.ply", self.splats)
        gaussian_splats_3d.write_rhino(folder / "gaussians.3dm", self.splats, {"count": self.cfg["count"], **self.values()})
        (folder / "gaussians.json").write_text(json.dumps({"config": self.cfg, "values": self.values()}, indent=1))
        return ["gaussians.ply", "gaussians.3dm", "gaussians.json"]

    def snapshot(self, folder: Path) -> list[str]:
        return self.export(folder)

class VolumeFieldBase:
    """What every dvolume-rendered field shares (VOXELS, ASSEMBLY, SOLIDS_ON_OFF, EXTRUSIONS): a
    density (and optional colour) volume over the cube from volume(), rendered by a
    voxel_field.VolumeRenderer with voxel_field.normalize_render's settings. render() is the canvas
    camera's view; views() a random views.per_step of the cameras, each moved by up to the config's
    jitter, augmented. The material colour (where there is no colour volume) and the background are
    optimized per their switches. A subclass sets self.renderer and provides volume(),
    model_param_groups(), model_values() and export_model(folder, density, color)."""

    name = "volume"

    def _setup(self, cfg: dict, device, optimize_material: bool):
        self.cfg = cfg
        self.views_cfg = cfg["views"]
        self.device = device
        self.canvas_camera = voxel_field.view_camera(cfg, self.views_cfg["canvas_view"], CANVAS_RESOLUTION)
        self.material = parametric_surface.hex_color(cfg["color"], device).clone().requires_grad_(
            cfg["optimize_color"] and optimize_material)
        self.bg = solid_color(BACKGROUND_COLOR, False, device)
        self.image = background_image(CANVAS_RESOLUTION, False, device)
        # From the last render(), in its graph:
        self.last_density: torch.Tensor | None = None
        self.last_color: torch.Tensor | None = None
        self.last_background: torch.Tensor | None = None
        self.last_canvas: torch.Tensor | None = None
        self.last_views: torch.Tensor | None = None  # the images of the last views(), detached

    def params(self) -> list[dict]:
        groups = self.model_param_groups()
        colors = [c for c in (self.material, self.bg) if c.requires_grad]
        groups = groups + ([{"params": colors, "lr": LEARNING_RATE}] if colors else [])
        if not groups:
            raise ValueError(f"{self.name}: nothing to optimize")
        return groups

    def blurrable(self) -> list[torch.Tensor]:
        return []

    def _render(self, camera) -> torch.Tensor:
        light = voxel_field.light_direction(self.cfg, camera)
        return self.renderer.render(camera, CANVAS_RESOLUTION, self.last_background[0], unit(self.material),
                                    light.to(self.device) if light is not None else None)[None]

    def render(self) -> torch.Tensor:
        self.last_density, self.last_color = self.volume()
        self.renderer.set(self.last_density, self.last_color)
        self.last_background = backdrop(unit(self.bg), self.image)
        self.last_canvas = self._render(self.canvas_camera)
        return self.last_canvas

    def views(self) -> torch.Tensor:
        n = self.views_cfg["per_step"]
        picked = torch.randperm(self.views_cfg["count"])[:n].tolist()
        images = torch.cat([self._render(voxel_field.view_camera(self.cfg, i, CANVAS_RESOLUTION, self.cfg["jitter"]))
                            for i in picked])
        self.last_views = images.detach()
        # Batch size as in the other fields, however many views.
        count = math.ceil(RANDOM_TRANSFORM_COUNT / n)
        return torch.cat([augment(image[None], count) for image in images])

    def regularized(self) -> torch.Tensor:
        return self.last_canvas if self.last_canvas is not None else self.render()

    def all_views(self) -> torch.Tensor:
        """(1, 3, H, W) every camera's unjittered view of the last render()'s volume, tiled on the
        canvas, the canvas view outlined."""
        if self.last_background is None:
            self.render()
        return voxel_field.all_views(self.renderer, CANVAS_RESOLUTION, self.last_background.detach(),
                                     unit(self.material).detach())

    def previews(self) -> dict[str, torch.Tensor]:
        """(1, 3, H, W) for the page: the volume's three central sections (front, right, top)."""
        with torch.no_grad():
            if self.last_density is None:
                self.render()
            return {"vox_slices": voxel_field.slices(self.last_density.detach(), None if self.last_color is None
                                                     else self.last_color.detach())}

    def values(self) -> dict:
        out = self.model_values()
        with torch.no_grad():
            if self.material.requires_grad:
                out["material"] = _to_hex(unit(self.material).tolist())
            if self.bg.requires_grad:
                out["background"] = _to_hex(unit(self.bg).tolist())
        return out

    def export(self, folder: Path) -> list[str]:
        """density.npz (and color.npz with a colour volume: dvolume fields, VolumeField.fromNpzFile reads
        them), the field's own .3dm and .json (export_model: the density's isosurface at iso_level, VTK's
        marching cubes, and whatever meshes the field has), <name>_config.json. -> the file names."""
        folder.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            density, color = self.volume()
            files = ["density.npz"]
            voxel_field.save_npz(folder / "density.npz", density, "density")
            if color is not None:
                voxel_field.save_npz(folder / "color.npz", color, "color")
                files.append("color.npz")
        files += self.export_model(folder, density.detach(), None if color is None else color.detach())
        (folder / f"{self.name}_config.json").write_text(json.dumps({"config": self.cfg, "values": self.values()},
                                                                    indent=1, default=str))
        return [*files, f"{self.name}_config.json"]

    def snapshot(self, folder: Path) -> list[str]:
        return self.export(folder)


class VoxelField(VolumeFieldBase):
    """A voxel volume (VOXELS, see voxel_field): density and, per mode, colour, each the sigmoid of a
    Laplacian pyramid of 3D tensors. penalty() is the density's 3D total variation x smoothness;
    after_step() its blur every blur.every steps."""

    name = "voxels"

    def __init__(self, device):
        cfg = voxel_field.normalize_config(VOXELS)
        self._setup(cfg, device, cfg["mode"] == "density")
        color = None
        if voxel_field.channels(cfg):
            start = (load_image(START_IMAGE, CANVAS_RESOLUTION, cfg["mode"] == "monochrome")
                     if CANVAS_INIT == "start" and START_IMAGE is not None else None)
            color = voxel_field.start_color(cfg, start, self.canvas_camera, CANVAS_RESOLUTION, CANVAS_INIT,
                                            CANVAS_NOISE_STD)
        self.voxels = voxel_field.Voxels(cfg, device, color)
        self.renderer = self.voxels.renderer
        r = cfg["resolution"]
        print(f"[voxels] {r}^3, {cfg['mode']}, {cfg['levels']} levels, init {cfg['init']}; "
              f"{self.views_cfg['count']} cameras, {self.views_cfg['per_step']} a step, {cfg['samples']} samples")

    def volume(self):
        return self.voxels.update()

    def model_param_groups(self) -> list[dict]:
        return self.voxels.param_groups(LEARNING_RATE)

    def model_values(self) -> dict:
        return self.voxels.values()

    def after_step(self, step: int) -> None:
        """Every blur.every steps (counted from 1), dvolume's blur of the finest pyramid levels."""
        every = self.cfg["blur"]["every"]
        if every and (step + 1) % every == 0:
            self.voxels.blur()

    def penalty(self, canvas: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.cfg["smoothness"]:
            return canvas.new_zeros(()), {}
        tv = voxel_field.total_variation_3d(self.last_density)
        return self.cfg["smoothness"] * tv, {"tv3d": tv.item()}

    def export_model(self, folder: Path, density, color) -> list[str]:
        """voxels.3dm: the density's isosurface, coloured from the colour volume (else the material)."""
        with torch.no_grad():
            material = tuple((unit(self.material) * 255).round().int().tolist())
        surface = voxel_field.isosurface(density, color, self.cfg["iso_level"], material)
        voxel_field.write_rhino(folder / "voxels.3dm", surface, "voxels", material,
                                {"level": self.cfg["iso_level"], "resolution": self.cfg["resolution"]})
        return ["voxels.3dm"]


class SolidField(VolumeFieldBase):
    """ASSEMBLY, SOLIDS_ON_OFF or EXTRUSIONS (solid_fields, configs ASSEMBLY / SOLIDS_ON_OFF /
    EXTRUSIONS): the model's volume() rendered as VOXELS'. SOLIDS_ON_OFF's penalty() pushes its
    multipliers on / off (ramped in over the run) and towards its coverage."""

    def __init__(self, kind: str, device):
        settings = {"ASSEMBLY": ASSEMBLY, "SOLIDS_ON_OFF": SOLIDS_ON_OFF, "EXTRUSIONS": EXTRUSIONS}[kind]
        cfg = solid_fields.NORMALIZERS[kind](settings)
        solid_fields.validate(kind, cfg)
        self.kind = kind
        self.name = kind.lower()
        # ASSEMBLY colours its clones by group: the material only tints the others.
        self._setup(cfg, device, kind != "ASSEMBLY")
        self.model = solid_fields.MODELS[kind](cfg, device)
        self.renderer = voxel_field.VolumeRenderer(cfg, device)
        self.total_steps = EPOCHS  # run_pole sets it to the run's
        self.step = 0
        print(f"[{self.name}] {cfg['resolution']}^3; {self.model.values()}")

    def volume(self):
        return self.model.volume()

    def model_param_groups(self) -> list[dict]:
        return self.model.param_groups(LEARNING_RATE)

    def model_values(self) -> dict:
        return self.model.values()

    def penalty(self, canvas: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if not hasattr(self.model, "penalty"):
            return canvas.new_zeros(()), {}
        self.model.progress = min(1.0, self.step / max(1, self.total_steps))
        self.step += 1
        return self.model.penalty()

    def export_model(self, folder: Path, density, color) -> list[str]:
        return self.model.export(folder, density, color)

LEVELSET_LAYER_DEFAULT = {"resolution": [64, 64], "color": "#000000", "alpha": 1.0, "optimize_color": True,
                          "contrast": 10.0, "optimize_contrast": False, "coverage": None, "coverage_weight": 10.0}


def _hex(value, default: str) -> str:
    value = str(value or default)
    ok = len(value) == 7 and value[0] == "#" and all(ch in "0123456789abcdefABCDEF" for ch in value[1:])
    return value.lower() if ok else default


def _to_hex(rgb) -> str:
    return "#" + "".join(f"{round(float(v) * 255):02x}" for v in rgb)


def normalize_levelset(cfg: dict | None) -> dict:
    """A LEVELSET config, defaults filled in and validated: {background (hex), optimize_background,
    layers: [{resolution [w, h], color (hex), alpha 0..1, optimize_color, contrast >= 0,
    optimize_contrast, coverage 0..1 or None, coverage_weight >= 0}]} (at least one layer)."""
    cfg = cfg or {}
    layers = []
    for i, given in enumerate(cfg.get("layers") or [{}]):
        layer = {**LEVELSET_LAYER_DEFAULT, **(given or {})}
        w, h = (int(v) for v in layer["resolution"])
        if not (1 <= w <= 4096 and 1 <= h <= 4096):
            raise ValueError(f"LEVELSET layer {i + 1}: bad resolution {w}x{h}")
        contrast = abs(float(layer["contrast"] or 0))
        if layer["optimize_contrast"] and contrast <= 0:
            raise ValueError(f"LEVELSET layer {i + 1}: an optimized contrast must be > 0 (it is optimized in log space)")
        coverage = layer["coverage"]
        weight = layer["coverage_weight"]
        layers.append({"resolution": [w, h], "color": _hex(layer["color"], "#000000"),
                       "alpha": min(1.0, max(0.0, float(layer["alpha"]))),
                       "optimize_color": bool(layer["optimize_color"]),
                       "contrast": contrast, "optimize_contrast": bool(layer["optimize_contrast"]),
                       "coverage": None if coverage in (None, "") else min(1.0, max(0.0, float(coverage))),
                       "coverage_weight": 10.0 if weight in (None, "") else max(0.0, float(weight))})
    return {"background": _hex(cfg.get("background"), "#000000"),
            "optimize_background": bool(cfg.get("optimize_background", False)), "layers": layers}


class LevelSetField:
    """LEVELSET: monochrome layers, each a (1, 1, h, w) logit tensor at its own resolution. A
    layer's sigmoid is resized to the canvas and put through contrast_curve with the layer's
    contrast: the alpha mask of its RGBA colour, composited over the background in order.
    Optimized colours live in logit space, optimized contrasts in log space; frozen ones are
    constants."""

    def __init__(self, device):
        self.cfg = normalize_levelset(LEVELSET)
        layers = self.cfg["layers"]
        lum = (luminance(load_image(START_IMAGE, CANVAS_RESOLUTION)).to(device)
               if CANVAS_INIT == "start" else None)
        self.levels, self.colors, self.contrasts = [], [], []
        for i, layer in enumerate(layers):
            w, h = layer["resolution"]
            if lum is not None:     # the level set luminance > threshold, at 0.5
                init = (resize(lum, (w, h)) - (i + 1) / (len(layers) + 1) + 0.5).clamp(0, 1)
            elif CANVAS_INIT == "noise":
                init = torch.sigmoid(CANVAS_NOISE_STD * torch.randn(1, 1, h, w)).to(device)
            else:
                init = torch.full((1, 1, h, w), 0.5, device=device)
            self.levels.append(logit(init).requires_grad_(True))
            rgba = torch.tensor([*parametric_surface.hex_color(layer["color"], "cpu").tolist(), layer["alpha"]],
                                device=device)
            self.colors.append(logit(rgba).requires_grad_(True) if layer["optimize_color"] else rgba)
            k = layer["contrast"]
            self.contrasts.append(torch.tensor(math.log(k), device=device, requires_grad=True)
                                  if layer["optimize_contrast"] else k)
        bg = parametric_surface.hex_color(self.cfg["background"], device)
        self.bg = logit(bg).requires_grad_(True) if self.cfg["optimize_background"] else bg
        self.image = background_image(CANVAS_RESOLUTION, False, device)
        print(f"[levelset]{len(layers)} layers: " + ", ".join(f"{w}x{h}" for w, h in (l["resolution"] for l in layers)))

    def params(self) -> list[dict]:
        layers = self.cfg["layers"]
        extra = ([c for c, l in zip(self.colors, layers) if l["optimize_color"]]
                 + [k for k, l in zip(self.contrasts, layers) if l["optimize_contrast"]]
                 + ([self.bg] if self.cfg["optimize_background"] else []))
        return [{"params": self.levels}, *([{"params": extra}] if extra else [])]

    def blurrable(self) -> list[torch.Tensor]:
        return self.levels

    def color(self, i: int) -> torch.Tensor:
        """(4,) RGBA in [0, 1]"""
        return torch.sigmoid(self.colors[i]) if self.cfg["layers"][i]["optimize_color"] else self.colors[i]

    def contrast(self, i: int) -> float | torch.Tensor:
        return self.contrasts[i].exp() if self.cfg["layers"][i]["optimize_contrast"] else self.contrasts[i]

    def background(self) -> torch.Tensor:
        return torch.sigmoid(self.bg) if self.cfg["optimize_background"] else self.bg

    def masks(self) -> list[torch.Tensor]:
        """Per layer, (1, 1, H, W): the canvas-sized layer after its contrast sigmoid (before the
        colour's alpha). Upscaled bilinearly: bicubic overshoots at the layer's edges, which the
        steep contrast curve turns into rings."""
        return [contrast_curve(resize(torch.sigmoid(level), CANVAS_RESOLUTION, "bilinear"), self.contrast(i)).clamp(0, 1)
                for i, level in enumerate(self.levels)]

    def render(self) -> torch.Tensor:
        img = backdrop(self.background(), self.image)
        for i, mask in enumerate(self.masks()):
            rgba = self.color(i)
            alpha = mask * rgba[3]
            img = img * (1 - alpha) + rgba[:3].view(1, 3, 1, 1) * alpha
        return img

    def regularized(self) -> torch.Tensor:
        return self.render()

    def penalty(self, canvas: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Per layer with a coverage budget: coverage_weight x (mean of its mask - coverage)^2.
        Stats "layer <i> coverage": the mask's mean (1-based i)."""
        loss, stats = canvas.new_zeros(()), {}
        budgeted = [(i, l) for i, l in enumerate(self.cfg["layers"]) if l["coverage"] is not None]
        if budgeted:
            masks = self.masks()
            for i, layer in budgeted:
                mean = masks[i].mean()
                loss = loss + layer["coverage_weight"] * (mean - layer["coverage"]) ** 2
                stats[f"layer {i + 1} coverage"] = mean.item()
        return loss, stats

    def values(self) -> dict:
        """The current colours and contrasts, in LEVELSET's format."""
        with torch.no_grad():
            cfg = json.loads(json.dumps(self.cfg))
            cfg["background"] = _to_hex(self.background().clamp(0, 1).tolist())
            for i, layer in enumerate(cfg["layers"]):
                rgba = self.color(i).clamp(0, 1).tolist()
                layer["color"], layer["alpha"] = _to_hex(rgba[:3]), round(rgba[3], 4)
                layer["contrast"] = round(float(self.contrast(i)), 4)
        return cfg

    def tensors(self) -> dict[str, torch.Tensor]:
        """Per layer, detached: layer_<i> (its own resolution, 0..1) and layer_<i>_sigmoid (on
        the canvas, after the contrast sigmoid: the mask its colour's alpha multiplies)."""
        with torch.no_grad():
            out = {}
            for i, (level, mask) in enumerate(zip(self.levels, self.masks())):
                out[f"layer_{i:02d}"] = torch.sigmoid(level)
                out[f"layer_{i:02d}_sigmoid"] = mask
        return out

    def snapshot(self, folder: Path) -> list[str]:
        """levelset.json (the config with the current colours and contrasts) and tensors()
        in levelset/ (save_tensors). -> the file names."""
        (folder / "levelset.json").write_text(json.dumps(self.values(), indent=1))
        files = save_tensors(folder / "levelset", self.tensors())
        return ["levelset.json", f"levelset/ ({len(files)})"]


def flow_to_bgr(offsets: torch.Tensor) -> np.ndarray:
    """(1, 2, H, W) displacement as a BGR image: hue = direction, brightness = amount
    (scaled to the largest displacement, at least 1 px)."""
    d = offsets[0].detach().float()
    mag = d.norm(dim=0)
    hue = (torch.atan2(d[1], d[0]) % (2 * math.pi)) / (2 * math.pi) * 180  # OpenCV hue: 0..180
    val = mag / max(mag.max().item(), 1.0) * 255
    hsv = torch.stack([hue, torch.full_like(hue, 255), val], dim=-1).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def solid_color(rgb: tuple[float, float, float], gray: bool, device) -> torch.Tensor:
    """(C,) an RGB colour (its luminance when gray), read through unit(); a leaf to optimize
    when BACKGROUND_OPTIMIZE. Kept linear, not as a logit: a white or black start sits where
    the sigmoid is flat, and Adam would need ~1000 steps just to move it off the rail."""
    c = torch.tensor(rgb, dtype=torch.float32)
    if gray:
        c = (c * torch.tensor(_LUMA)).sum(0, keepdim=True)
    return c.to(device).requires_grad_(BACKGROUND_OPTIMIZE)


def unit(x: torch.Tensor) -> torch.Tensor:
    """x clamped to [0, 1], with the gradient passed straight through (a value pushed past
    a bound can still come back)."""
    return x + (x.clamp(0, 1) - x).detach()


def composite_over(premult: torch.Tensor, alphas: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    """Layers over a background, later layers on top: premultiplied colours (colour x alpha,
    (N, C, H, W)), alphas (N, 1, H, W), bg (1, C, H, W) -> (1, C, H, W)."""
    keep = (1 - alphas).flip(0).cumprod(0).flip(0)  # keep[i] = prod_{j >= i} (1 - alpha_j)
    keep = torch.cat([keep, torch.ones_like(alphas[:1])])
    return ((premult * keep[1:]).sum(0) + bg[0] * keep[0])[None]


def random_centers(n: int, spread: float = 1.0) -> torch.Tensor:
    """(n, 2) atanh of uniform positions in (-0.9, 0.9) x spread: tanh keeps them on the canvas."""
    return torch.atanh((torch.rand(n, 2) * 1.8 - 0.9) * spread)


class PatchField:
    """PATCH_COUNT m x m images with alpha (sigmoid of logits), each resampled onto the canvas
    through its own affine transform -- centre (tanh, stays on the canvas), side (sigmoid
    into PATCH_SCALE_RANGE) and angle -- and composited over the background in order.
    With PATCH_RANDOMIZE_TRANSFORMS the transforms are redrawn on every render instead
    of optimized. PATCH_SOURCE "images": the patches are PATCH_IMAGES, fixed; only the
    transforms are optimized."""

    def __init__(self, device):
        self.from_images = PATCH_SOURCE == "images"
        if self.from_images:
            entries = [(e["path"], int(e.get("count", 1))) for e in PATCH_IMAGES if int(e.get("count", 1)) > 0]
            if not entries:
                raise ValueError("PIXEL_PATCHES from images: no images (or every count is 0)")
            n, m, self.gray = sum(c for _, c in entries), PATCH_IMAGE_RESOLUTION, False
        else:
            n, m, self.gray = PATCH_COUNT, PATCH_RESOLUTION, PATCH_MONOCHROME
        cw, ch = CANVAS_RESOLUTION
        self.aspect = torch.tensor([cw, ch], dtype=torch.float32, device=device) / min(cw, ch)
        self.n, self.device = n, device
        self.randomized = PATCH_RANDOMIZE_TRANSFORMS and not self.from_images
        self.randomize_transforms()
        if not self.randomized:
            self.pos.requires_grad_(True)
            self.scale.requires_grad_(True)
            self.angle.requires_grad_(PATCH_ROTATION)
        self.opaque = PATCH_OPAQUE
        self.bg = solid_color(BACKGROUND_COLOR, self.gray, device)
        self.image = background_image(CANVAS_RESOLUTION, self.gray, device)
        if self.from_images:
            # (S, 4, m, m) the sources and (S, 1, m, m) each one's rectangle (not its padding)
            loaded = [load_rgba(path, m) for path, _ in entries]
            self.sources = torch.stack([rgba for rgba, _ in loaded]).to(device)
            self.inside = torch.stack([rect for _, rect in loaded]).to(device)
            counts = torch.tensor([c for _, c in entries])
            # Which source each patch shows, in a shuffled stacking order (SEED).
            self.index = torch.repeat_interleave(torch.arange(len(entries)), counts)[torch.randperm(n)].to(device)
            self.pixels = None
            print(f"[patches] {len(entries)} images, {n} patches")
            return

        c, p = 1 if self.gray else 3, 1 if PATCH_IDENTICAL else n
        if CANVAS_INIT == "noise":
            colors = torch.sigmoid(CANVAS_NOISE_STD * torch.randn(p, c, m, m)).to(device)
        elif CANVAS_INIT == "gray":
            colors = torch.full((p, c, m, m), 0.5, device=device)
        else:
            # What START_IMAGE shows under each patch's starting placement (identical: their mean).
            a, b = self.thetas().detach().split([2, 1], dim=-1)
            inv = torch.linalg.inv(a)
            grid = F.affine_grid(torch.cat([inv, -inv @ b], -1), [n, c, m, m], align_corners=False)
            start = load_image(START_IMAGE, CANVAS_RESOLUTION, self.gray).to(device).expand(n, -1, -1, -1)
            colors = F.grid_sample(start, grid, padding_mode="border", align_corners=False)
            if PATCH_IDENTICAL:
                colors = colors.mean(0, keepdim=True)
        alpha = torch.full((p, 1, m, m), 0.8, device=device)  # mostly opaque: the patches read from the start
        self.pixels = logit(torch.cat([colors, alpha], 1)).requires_grad_(True)  # (P, C + 1, m, m)

    def randomize_transforms(self) -> None:
        """Fresh random placements (from torch's CPU generator, so they follow SEED)."""
        n = self.n
        self.pos = random_centers(n).to(self.device)
        self.scale = logit(torch.rand(n) * 0.8 + 0.1).to(self.device)
        self.angle = ((torch.rand(n) * 2 - 1) * math.pi if PATCH_ROTATION else torch.zeros(n)).to(self.device)

    def params(self) -> list[dict]:
        transform = [] if self.randomized else [self.pos, self.scale] + ([self.angle] if PATCH_ROTATION else [])
        return [*([] if self.from_images else [{"params": [self.pixels], "lr": LEARNING_RATE}]),
                *([{"params": transform, "lr": LEARNING_RATE * PATCH_TRANSFORM_LR}] if transform else []),
                *([{"params": [self.bg], "lr": LEARNING_RATE}] if BACKGROUND_OPTIMIZE else [])]

    def blurrable(self) -> list[torch.Tensor]:
        return [] if self.from_images else [self.pixels]

    def thetas(self) -> torch.Tensor:
        """(N, 2, 3) affine_grid matrices: canvas coords (-1..1) -> patch coords (-1..1)."""
        lo, hi = PATCH_SCALE_RANGE
        side = lo + (hi - lo) * torch.sigmoid(self.scale)  # fraction of the short side
        cos, sin = torch.cos(self.angle), torch.sin(self.angle)
        rot = torch.stack([torch.stack([cos, sin], -1), torch.stack([-sin, cos], -1)], -2)  # R(-angle)
        a = rot * (self.aspect / side[:, None])[:, None, :]  # R(-angle) @ diag(aspect / side)
        return torch.cat([a, -(a @ torch.tanh(self.pos)[..., None])], -1)

    def render(self) -> torch.Tensor:
        if self.randomized:
            self.randomize_transforms()
        n = self.n
        w, h = CANVAS_RESOLUTION
        grid = F.affine_grid(self.thetas(), [n, 1, h, w], align_corners=False)
        rgba = self.rgba()
        # Resampled premultiplied: the zero padding outside a patch then only fades its
        # alpha, instead of also blending its colour towards black (a dark rim).
        patches = torch.cat([rgba[:, :-1] * rgba[:, -1:], rgba[:, -1:]], 1).expand(n, -1, -1, -1)
        layers = F.grid_sample(patches, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        img = composite_over(layers[:, :-1], layers[:, -1:], backdrop(unit(self.bg), self.image))
        return img.expand(-1, 3, -1, -1) if self.gray else img

    def rgba(self) -> torch.Tensor:
        """(P, C + 1, m, m) in [0, 1]; PATCH_OPAQUE: alpha 1 (its logits stay unused).
        From images: (N, 4, m, m) each patch's source; opaque: alpha 1 inside the image."""
        if self.from_images:
            rgba = self.sources[self.index]
            return torch.cat([rgba[:, :-1], self.inside[self.index]], 1) if self.opaque else rgba
        rgba = torch.sigmoid(self.pixels)
        return torch.cat([rgba[:, :-1], torch.ones_like(rgba[:, -1:])], 1) if self.opaque else rgba

    def patch_image(self, size: int = 256, cell: int = 16) -> torch.Tensor:
        """(1, 3, size, size) the first patch untransformed, over a grey checkerboard (so its
        alpha shows), nearest-upscaled so its m x m pixels stay crisp."""
        rgba = self.rgba()[:1].detach()
        rgba = F.interpolate(rgba, size=(size, size), mode="nearest")
        ys, xs = torch.meshgrid(torch.arange(size, device=rgba.device), torch.arange(size, device=rgba.device), indexing="ij")
        checker = (0.6 + 0.2 * ((ys // cell + xs // cell) % 2)).float()[None, None]
        img = rgba[:, -1:] * rgba[:, :-1] + (1 - rgba[:, -1:]) * checker
        return img.expand(-1, 3, -1, -1)

    def regularized(self) -> torch.Tensor:
        # Randomized: another render would be another layout, so the TV is of the patches themselves.
        return self.rgba() if self.randomized else self.render()


class GaussianField:
    """GAUSSIAN_COUNT anisotropic Gaussians: centre (tanh, stays on the canvas), two stds
    (sigmoid, up to GAUSSIAN_MAX_SIZE), angle, opacity and colour. Composited over the
    background in order; GAUSSIAN_CONTRAST sharpens each splat's alpha first. With
    GAUSSIAN_MONOCHROME too, their summed density is pushed through a steep sigmoid into
    one fg-over-bg shape mask instead (the splat colours then go unused, and fg and bg
    stay RGB). GAUSSIAN_NEGATIVE_COUNT more splats, in the background colour, go on top of
    everything to cut voids (with one shape: their own mask is subtracted from it). They
    share the geometry tensors: the first n entries are positive, the rest negative.
    GAUSSIAN_RECTILINEAR adds a roundness per splat (sigmoid), GAUSSIAN_AXIS_ALIGNED fixes
    the angles at 0."""

    def __init__(self, device):
        n, self.gray = GAUSSIAN_COUNT, GAUSSIAN_MONOCHROME
        self.n, self.negative = n, GAUSSIAN_NEGATIVE_COUNT
        total = n + self.negative
        self.one_shape = GAUSSIAN_MONOCHROME and GAUSSIAN_CONTRAST > 0
        cw, ch = CANVAS_RESOLUTION
        short = min(cw, ch)
        # Pixel centres in units of the short side, centred on the canvas.
        ys = (torch.arange(ch, device=device) + 0.5 - ch / 2) / short
        xs = (torch.arange(cw, device=device) + 0.5 - cw / 2) / short
        self.gy, self.gx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
        self.half = torch.tensor([cw, ch], dtype=torch.float32, device=device) / (2 * short)
        self.min_size = 0.5 / short  # half a pixel

        self.pos = random_centers(total, GAUSSIAN_SPREAD).to(device).requires_grad_(True)
        self.size = logit(torch.rand(total, 2) * 0.6 + 0.2).to(device).requires_grad_(True)
        self.angle = ((torch.rand(total) * 2 - 1) * math.pi).to(device).requires_grad_(True)
        self.opacity = logit(torch.full((total,), 0.8)).to(device).requires_grad_(True)
        self.roundness = (logit(torch.full((total,), float(GAUSSIAN_ROUNDNESS_INIT))).to(device).requires_grad_(True)
                          if GAUSSIAN_RECTILINEAR else None)

        c = 1 if self.gray else 3
        if CANVAS_INIT == "noise":
            colors = torch.sigmoid(CANVAS_NOISE_STD * torch.randn(n, c)).to(device)
        elif CANVAS_INIT == "gray":
            colors = torch.full((n, c), 0.5, device=device)
        else:
            # START_IMAGE at each centre (tanh(pos) is already in grid_sample's -1..1 coords).
            start = load_image(START_IMAGE, CANVAS_RESOLUTION, self.gray).to(device)
            grid = torch.tanh(self.pos.detach()[:n])[None, :, None]  # (1, N, 1, 2)
            colors = F.grid_sample(start, grid, align_corners=False)[0, :, :, 0].T  # (N, C)
        self.colors = logit(colors).requires_grad_(True)
        self.bg = solid_color(BACKGROUND_COLOR, self.gray and not self.one_shape, device)
        self.image = background_image(CANVAS_RESOLUTION, self.gray and not self.one_shape, device)
        self.fg = solid_color(GAUSSIAN_FG_COLOR, False, device).requires_grad_(GAUSSIAN_FG_OPTIMIZE)

    def params(self) -> list[torch.Tensor]:
        colors = [self.fg, self.bg] if self.one_shape else [self.colors, self.bg]
        shape = ([] if GAUSSIAN_AXIS_ALIGNED else [self.angle]) + ([self.roundness] if GAUSSIAN_RECTILINEAR else [])
        return [self.pos, self.size, self.opacity] + shape + [c for c in colors if c.requires_grad]

    def blurrable(self) -> list[torch.Tensor]:
        return []

    def density(self) -> torch.Tensor:
        """(N, 1, H, W) opacity x Gaussian of every splat."""
        c = torch.tanh(self.pos) * self.half
        dx = self.gx - c[:, 0, None, None]
        dy = self.gy - c[:, 1, None, None]
        size = (GAUSSIAN_MAX_SIZE * torch.sigmoid(self.size)).clamp_min(self.min_size)[..., None, None]
        if GAUSSIAN_AXIS_ALIGNED:
            u, v = dx / size[:, 0], dy / size[:, 1]
        else:
            cos, sin = torch.cos(self.angle)[:, None, None], torch.sin(self.angle)[:, None, None]
            u = (cos * dx + sin * dy) / size[:, 0]
            v = (cos * dy - sin * dx) / size[:, 1]
        if GAUSSIAN_RECTILINEAR:
            # p from 2 (roundness 1) to GAUSSIAN_MAX_POWER (0), log-interpolated. |u| is clamped:
            # below, so d/dp (|u|^p log|u|) is finite at the centre; above, so |u|^p can't overflow
            # (the density is ~0 there anyway).
            p = 2 * (GAUSSIAN_MAX_POWER / 2) ** (1 - torch.sigmoid(self.roundness))[:, None, None]
            r = u.abs().clamp(1e-6, 6) ** p + v.abs().clamp(1e-6, 6) ** p
        else:
            r = u * u + v * v
        return (torch.sigmoid(self.opacity)[:, None, None] * torch.exp(-0.5 * r))[:, None]

    def render(self) -> torch.Tensor:
        alphas = self.density()
        if self.one_shape:
            mask = torch.sigmoid(GAUSSIAN_CONTRAST * (alphas[:self.n].sum(0, keepdim=True) - 0.5))
            if self.negative:
                mask = mask * (1 - torch.sigmoid(GAUSSIAN_CONTRAST * (alphas[self.n:].sum(0, keepdim=True) - 0.5)))
            bg, fg = backdrop(unit(self.bg), self.image), unit(self.fg).view(1, -1, 1, 1)
            return bg + (fg - bg) * mask
        alphas = contrast_curve(alphas, GAUSSIAN_CONTRAST)  # GAUSSIAN_CONTRAST = 0: unchanged
        bg = backdrop(unit(self.bg), self.image)
        premult = torch.cat([torch.sigmoid(self.colors)[..., None, None] * alphas[:self.n],
                             bg * alphas[self.n:]])  # negatives last, on top: the background shows through
        img = composite_over(premult, alphas, bg)
        return img.expand(-1, 3, -1, -1) if self.gray else img

    def regularized(self) -> torch.Tensor:
        return self.render()


FUNCTION_DEFAULT = {"wave": "sine", "offset": 0.0, "optimize_offset": True, "components": 4, "frequency": 2.0,
                    "optimize_frequency": True, "amplitude": 0.05, "color": "#000000", "alpha": 1.0, "optimize_color": True,
                    "filled": True, "thickness": 0.02}


def normalize_function_stack(cfg: dict | None) -> dict:
    """A FUNCTION_STACK config, defaults filled in and validated: {kind, softness > 0, shape_lr >= 0,
    functions: [{wave, offset, optimize_offset, components >= 1, frequency > 0, optimize_frequency, amplitude >= 0,
    color (hex), alpha 0..1, optimize_color, filled, thickness > 0}]} (at least one function)."""
    def filled_in(defaults: dict, given: dict | None) -> dict:
        return {**defaults, **{k: v for k, v in (given or {}).items() if v not in (None, "")}}

    cfg = filled_in({"kind": "line", "softness": 1.0, "shape_lr": 0.5, "functions": [{}]}, cfg)
    if cfg["kind"] not in FUNCTION_KINDS:
        raise ValueError(f"Bad function stack kind: {cfg['kind']!r} (one of {FUNCTION_KINDS})")
    functions = []
    for i, given in enumerate(cfg["functions"] or [{}]):
        fn = filled_in(FUNCTION_DEFAULT, given)
        if fn["wave"] not in FUNCTION_WAVES:
            raise ValueError(f"Function {i + 1}: bad wave {fn['wave']!r} (one of {FUNCTION_WAVES})")
        if float(fn["frequency"]) <= 0:
            raise ValueError(f"Function {i + 1}: the frequency must be > 0")
        functions.append({"wave": fn["wave"], "offset": float(fn["offset"]),
                          "optimize_offset": bool(fn["optimize_offset"]),
                          "components": max(1, int(fn["components"])), "frequency": float(fn["frequency"]),
                          "optimize_frequency": bool(fn["optimize_frequency"]),
                          "amplitude": max(0.0, float(fn["amplitude"])), "color": _hex(fn["color"], "#000000"),
                          "alpha": min(1.0, max(0.0, float(fn["alpha"]))),
                          "optimize_color": bool(fn["optimize_color"]), "filled": bool(fn["filled"]),
                          "thickness": max(1e-4, float(fn["thickness"]))})
    return {"kind": cfg["kind"], "softness": max(0.05, float(cfg["softness"])),
            "shape_lr": max(0.0, float(cfg["shape_lr"])), "functions": functions}


class FunctionStackField:
    """FUNCTION_STACK: each function is a curve, one wave per component on top of a horizontal
    line or a circle, sampled once per canvas column or once along each of a fan of rays. The
    curve becomes an alpha mask through sigmoids of the height / radius difference to it, and the
    masks are composited over the background in order. Colours are linear, read through unit()
    (see solid_color)."""

    def __init__(self, device):
        self.cfg = normalize_function_stack(FUNCTION_STACK)
        self.circle = self.cfg["kind"] == "circle"
        functions = self.cfg["functions"]
        cw, ch = CANVAS_RESOLUTION
        self.short = min(cw, ch)
        # radians per cycle, per unit of self.t
        self.cycle = 1.0 if self.circle else 2 * math.pi * self.short / cw
        self.softness = self.cfg["softness"] / self.short
        ys = (torch.arange(ch, device=device) + 0.5 - ch / 2) / self.short
        xs = (torch.arange(cw, device=device) + 0.5 - cw / 2) / self.short
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
        if self.circle:
            # Enough rays for one per pixel where they leave the canvas
            rays = min(2048, math.ceil(math.pi * math.hypot(cw, ch)))
            self.t = (torch.arange(rays, device=device) + 0.5) / rays * 2 * math.pi - math.pi
            self.position = torch.hypot(gx, gy)  # (H, W), compared with the curve along its ray
            # Each pixel's two nearest rays, and how far it is towards the second
            ray = (torch.atan2(gy, gx) + math.pi) / (2 * math.pi) * rays - 0.5
            self.ray_a = ray.floor().long() % rays
            self.ray_b = (self.ray_a + 1) % rays
            self.ray_mix = ray - ray.floor()
            if not all(fn["filled"] for fn in functions):
                delta = self.t[:, None] - self.t[None]  # (rays, rays): ray - curve sample
                self.delta_cos, self.delta_sin = torch.cos(delta), torch.sin(delta)
        else:
            self.t = xs
            self.position = -gy  # up is positive
        self.offsets, self.amps, self.phases, self.freqs, self.colors = [], [], [], [], []
        for fn in functions:
            k = torch.arange(1, fn["components"] + 1, dtype=torch.float32)
            self.freqs.append((k * fn["frequency"]).to(device).requires_grad_(fn["optimize_frequency"]))
            self.amps.append((fn["amplitude"] / k).to(device).requires_grad_(True))
            self.phases.append((torch.rand(fn["components"]) * 2 * math.pi).to(device).requires_grad_(True))
            self.offsets.append(torch.tensor(fn["offset"], device=device, requires_grad=fn["optimize_offset"]))
            rgba = [*parametric_surface.hex_color(fn["color"], "cpu").tolist(), fn["alpha"]]
            self.colors.append(torch.tensor(rgba, device=device, requires_grad=fn["optimize_color"]))
        self.bg = solid_color(BACKGROUND_COLOR, False, device)
        self.image = background_image(CANVAS_RESOLUTION, False, device)
        print(f"[function stack]{self.cfg['kind']}: {len(functions)} functions")

    def params(self) -> list[dict]:
        shape = self.amps + [o for o in self.offsets if o.requires_grad]
        waves = self.phases + [q for q in self.freqs if q.requires_grad]
        colors = [c for c in [*self.colors, self.bg] if c.requires_grad]
        return [{"params": shape, "lr": LEARNING_RATE * self.cfg["shape_lr"]},
                {"params": waves, "lr": LEARNING_RATE},
                *([{"params": colors, "lr": LEARNING_RATE}] if colors else [])]

    def blurrable(self) -> list[torch.Tensor]:
        return []

    def curve(self, i: int) -> torch.Tensor:
        """(N,) the function's height at each column, or radius along each ray."""
        theta = self.freqs[i][:, None] * self.cycle * self.t + self.phases[i][:, None]
        wave = wave_function(self.cfg["functions"][i]["wave"], theta)
        base = FUNCTION_BASE_RADIUS if self.circle else 0.0
        return base + self.offsets[i] + (self.amps[i][:, None] * wave).sum(0)

    def on_canvas(self, values: torch.Tensor) -> torch.Tensor:
        """(N,) values per column or ray -> (H, W) per pixel."""
        if not self.circle:
            return values[None]
        return torch.lerp(values[self.ray_a], values[self.ray_b], self.ray_mix)

    def stroke_bounds(self, curve: torch.Tensor, half: float) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,) (N,) where each column or ray enters and leaves the curve drawn with a round brush of
        radius `half`, so a stroke keeps its width on the steep parts too. A column's or ray's
        stretch of stroke is taken to be one piece. The brush's own shape is detached: the loss
        moves the curve."""
        if self.circle:
            along = curve * self.delta_cos
            with torch.no_grad():
                inside = half ** 2 - (curve * self.delta_sin) ** 2
                chord = inside.clamp_min(0).sqrt()
                hits = (inside > 0) & (along + chord > 0)
            top = (along + chord).masked_fill(~hits, -1.0).amax(1)
            # a curve behind the centre (a negative radius) may reach no ray: then top < bottom
            return (along - chord).masked_fill(~hits, 3.0).amin(1), top
        reach = max(1, math.ceil(half * self.short))
        steps = torch.arange(-reach, reach + 1, device=curve.device) / self.short
        cap = (half ** 2 - steps ** 2).clamp_min(0).sqrt()
        windows = F.pad(curve[None, None], (reach, reach), mode="replicate")[0, 0].unfold(0, 2 * reach + 1, 1)
        return (windows - cap).amin(1), (windows + cap).amax(1)

    def mask(self, i: int) -> torch.Tensor:
        """(1, 1, H, W) the function's mask, before its colour's alpha."""
        fn, curve = self.cfg["functions"][i], self.curve(i)
        edge = lambda values: torch.sigmoid((self.on_canvas(values) - self.position) / self.softness)
        if fn["filled"]:
            return edge(curve)[None, None]
        bottom, top = self.stroke_bounds(curve, fn["thickness"] / 2)
        return (edge(top) - edge(bottom)).clamp_min(0)[None, None]

    def render(self) -> torch.Tensor:
        img = backdrop(unit(self.bg), self.image)
        for i, color in enumerate(self.colors):
            rgba = unit(color)
            alpha = self.mask(i) * rgba[3]
            img = img * (1 - alpha) + rgba[:3].view(1, 3, 1, 1) * alpha
        return img

    def regularized(self) -> torch.Tensor:
        return self.render()

    def values(self) -> dict:
        """The config with the current offsets and colours, the background, and each function's
        waves: {frequency (cycles across the canvas / around the circle), amplitude, phase}."""
        with torch.no_grad():
            cfg = json.loads(json.dumps(self.cfg))
            cfg["background"] = _to_hex(unit(self.bg).tolist())
            for i, fn in enumerate(cfg["functions"]):
                rgba = unit(self.colors[i]).tolist()
                fn["offset"] = round(self.offsets[i].item(), 5)
                fn["color"], fn["alpha"] = _to_hex(rgba[:3]), round(rgba[3], 4)
                fn["waves"] = [{"frequency": round(q, 5), "amplitude": round(a, 5), "phase": round(p, 5)}
                               for q, a, p in zip(self.freqs[i].tolist(), self.amps[i].tolist(), self.phases[i].tolist())]
        return cfg

    def snapshot(self, folder: Path) -> list[str]:
        """function_stack.json (values()). -> the file names."""
        (folder / "function_stack.json").write_text(json.dumps(self.values(), indent=1))
        return ["function_stack.json"]


class VAE:
    """A frozen diffusers autoencoder (VAE_MODELS[name]) in `precision`, differentiable
    w.r.t. its latent. One is kept loaded across runs (another name or precision replaces
    it) and moved to the device a run asks for."""

    _cache: dict[tuple[str, str], 'VAE'] = {}

    @classmethod
    def get(cls, name: str, device, precision: str = "fp32") -> 'VAE':
        vae = cls._cache.get((name, precision))
        if vae is None:
            cls._cache.clear()
            vae = cls._cache[(name, precision)] = cls(name, device, precision)
        elif vae.device != torch.device(device):
            vae.model.to(device)
            vae.device = torch.device(device)
        return vae

    def __init__(self, name: str, device, precision: str):
        if name not in VAE_MODELS:
            raise ValueError(f"Unknown VAE {name!r} (one of {list(VAE_MODELS)})")
        repo, subfolder, kind = VAE_MODELS[name]
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        try:
            import diffusers
            model = getattr(diffusers, kind).from_pretrained(repo, subfolder=subfolder, torch_dtype=self.dtype)
        except Exception as exc:
            raise RuntimeError(f"VAE {repo!r} could not be loaded ({type(exc).__name__}: {exc}). "
                               "It needs `diffusers` and the model in the Hugging Face cache or online.") from exc
        self.name = name
        self.device = torch.device(device)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.scale = float(model.config.get("scaling_factor") or 1.0)
        self.shift = float(model.config.get("shift_factor") or 0.0)
        # FLUX.2: no scaling factor -- its pipeline normalizes the latent's 2 x 2 patches (as
        # channels) by the VAE's batch-norm statistics. Kept here as (C, 2, 2) per-pixel maps,
        # tiled over the latent, so the field stays (C, h, w).
        bn = getattr(model, "bn", None)
        self.patch = tuple(model.config.get("patch_size") or (1, 1)) if bn is not None else (1, 1)
        if bn is not None:
            c = model.config["latent_channels"]
            self.bn_mean = bn.running_mean.float().view(c, *self.patch)
            self.bn_std = (bn.running_var.float() + model.config["batch_norm_eps"]).sqrt().view(c, *self.patch)
        with torch.no_grad():
            self.factor = 64 // self.encode(torch.full((1, 3, 64, 64), 0.5, device=self.device)).shape[-1]
        self.multiple = self.factor * self.patch[0]  # image sides are a multiple of this
        print(f"[vae] {repo}{'/' + subfolder if subfolder else ''} ({kind}, {precision}): "
              f"{model.config.get('latent_channels')} channels, /{self.factor}")

    def normalization(self, z: torch.Tensor) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        """(shift, scale) for a raw latent like z: unit = (raw - shift) x scale."""
        if self.patch == (1, 1):
            return self.shift, self.scale
        reps = (1, z.shape[-2] // self.patch[0], z.shape[-1] // self.patch[1])
        return self.bn_mean.repeat(*reps)[None], 1 / self.bn_std.repeat(*reps)[None]

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """(1, 3, H, W) in [0, 1] -> (1, C, H / factor, W / factor) unit-variance latent (the
        posterior mean, for a KL autoencoder), float32."""
        out = self.model.encode((img * 2 - 1).to(self.dtype))
        z = (out.latent_dist.mean if hasattr(out, "latent_dist")
             else out.latents if hasattr(out, "latents") else out.latent).float()
        shift, scale = self.normalization(z)
        return (z - shift) * scale

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Unit-variance latent -> (1, 3, h, w) image in [0, 1], float32."""
        shift, scale = self.normalization(z)
        x = self.model.decode((z / scale + shift).to(self.dtype)).sample
        return (x.float() * 0.5 + 0.5).clamp(0, 1)


class VAEField:
    """VAE_MODEL's latent: render() decodes it at the canvas resolution x VAE_DECODE_SCALE
    (rounded to the VAE's factor) and resizes it to the canvas. penalty() keeps it within
    VAE_LATENT_BOUND, where the decoder was trained."""

    def __init__(self, device):
        self.vae = VAE.get(VAE_MODEL, device, VAE_PRECISION)
        m = self.vae.multiple
        self.size = tuple(max(m, round(v * VAE_DECODE_SCALE / m) * m) for v in CANVAS_RESOLUTION)
        start = load_image(START_IMAGE if CANVAS_INIT == "start" else None, self.size).to(device)
        with torch.no_grad():
            z = self.vae.encode(start)
        if CANVAS_INIT == "noise":
            z = z + CANVAS_NOISE_STD * torch.randn(z.shape).to(device)  # CPU generator: follows SEED
        self.latent = z.clone().requires_grad_(True)
        self.last_canvas: torch.Tensor | None = None
        print(f"[vae] latent {tuple(z.shape[1:])}, decoded at {self.size[0]}x{self.size[1]}")

    def params(self) -> list[torch.Tensor]:
        return [self.latent]

    def blurrable(self) -> list[torch.Tensor]:
        return [self.latent]

    def render(self) -> torch.Tensor:
        self.last_canvas = resize(self.vae.decode(self.latent), CANVAS_RESOLUTION)
        return self.last_canvas

    def regularized(self) -> torch.Tensor:
        # The canvas run_pole just decoded (a second decode would double the step's cost).
        return self.last_canvas if self.last_canvas is not None else self.render()

    def penalty(self, canvas: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """VAE_LATENT_WEIGHT x mean(relu(|z| - VAE_LATENT_BOUND)^2). Stats: the latent's std
        and the share of its values past the bound."""
        excess = F.relu(self.latent.abs() - VAE_LATENT_BOUND)
        stats = {"latent std": self.latent.std().item(), "latent out": (excess > 0).float().mean().item()}
        return VAE_LATENT_WEIGHT * excess.pow(2).mean(), stats

    def latent_image(self, cell: int = 4) -> torch.Tensor:
        """(1, 3, H, W) every latent channel as a grey tile (-bound black .. +bound white) in a
        near-square grid, each latent value cell x cell pixels."""
        z = (self.latent.detach().float()[0] / (2 * VAE_LATENT_BOUND) + 0.5).clamp(0, 1)
        c, h, w = z.shape
        cols = math.ceil(math.sqrt(c))
        rows = math.ceil(c / cols)
        grid = z.new_full((rows * (h + 1) - 1, cols * (w + 1) - 1), 0.25)
        for i in range(c):
            r, q = divmod(i, cols)
            grid[r * (h + 1):r * (h + 1) + h, q * (w + 1):q * (w + 1) + w] = z[i]
        grid = F.interpolate(grid[None, None], scale_factor=cell, mode="nearest")
        return grid.expand(-1, 3, -1, -1)

    def snapshot(self, folder: Path) -> list[str]:
        """latent.npy ((C, h, w) float32, unit-variance units, decodes with VAE_MODEL) and
        latent.png (latent_image). -> the file names."""
        np.save(folder / "latent.npy", self.latent.detach().float()[0].cpu().numpy())
        cv2.imwrite(str(folder / "latent.png"), to_bgr(self.latent_image()))
        return ["latent.npy", "latent.png"]


class ConvField:
    """CONVOLUTIONS: an image made by a CNN whose weights are the field (see CONV_INPLACE).
    Generative: params are the input tensor and the weights; inplace: the weights only."""

    def __init__(self, device):
        self.inplace = CONV_INPLACE
        k, c, n = max(1, CONV_KERNEL), max(1, CONV_CHANNELS), max(1, CONV_LAYERS)
        act = {"leaky_relu": lambda: torch.nn.LeakyReLU(0.2), "relu": torch.nn.ReLU, "silu": torch.nn.SiLU}[CONV_ACTIVATION]
        norm = {"none": lambda ch: torch.nn.Identity(),
                "instance": lambda ch: torch.nn.InstanceNorm2d(ch, affine=True),
                "group": lambda ch: torch.nn.GroupNorm(math.gcd(8, ch), ch)}[CONV_NORM]
        layers = []
        if self.inplace:
            # The fixed image as logits; the net's residual is added to them.
            self.base = logit(initial_image(CANVAS_RESOLUTION)).to(device)
            chans = [3] + [c] * (n - 1) + [3]
            for i in range(n):
                layers.append(torch.nn.Conv2d(chans[i], chans[i + 1], k, padding="same", padding_mode="reflect"))
                if i < n - 1:
                    layers += [norm(chans[i + 1]), act()]
        else:
            # Stride-2 transposed convs: out = 2 x in for any kernel >= 2 (output_padding by parity).
            k = max(2, k)
            chans = [max(8, c >> i) for i in range(n)] + [3]
            for i in range(n):
                layers.append(torch.nn.ConvTranspose2d(chans[i], chans[i + 1], k, stride=2, padding=(k - 1) // 2,
                                                       output_padding=k % 2))
                if i < n - 1:
                    layers += [norm(chans[i + 1]), act()]
        gain = ({"a": 0.2, "nonlinearity": "leaky_relu"} if CONV_ACTIVATION == "leaky_relu"
                else {"nonlinearity": "relu"})  # SiLU: close enough to ReLU's gain
        for layer in layers:
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(layer.weight, **gain)
                torch.nn.init.zeros_(layer.bias)
        if self.inplace:
            torch.nn.init.zeros_(layers[-1].weight)  # starts as the identity: canvas = the image
        self.net = torch.nn.Sequential(*layers).to(device)
        w, h = CONV_INPUT_RESOLUTION
        self.input = (None if self.inplace
                      else torch.randn(1, chans[0], max(1, h), max(1, w)).to(device).requires_grad_(True))
        self.chans = chans
        self.last_canvas: torch.Tensor | None = None
        self.last_residual: torch.Tensor | None = None
        weights = sum(t.numel() for t in self.net.parameters())
        if self.inplace:
            print(f"[conv] inplace, {n} layers {chans}, kernel {k}: {weights} weights")
        else:
            print(f"[conv] {w}x{h} input -> {w << n}x{h << n}, {n} layers {chans}, kernel {k}: "
                  f"{weights} weights + {self.input.numel()} input")
            if CANVAS_INIT == "start" and START_IMAGE is not None and CONV_FIT_STEPS > 0:
                self.fit(load_image(START_IMAGE, CANVAS_RESOLUTION).to(device))

    def fit(self, target: torch.Tensor) -> None:
        """Generative: CONV_FIT_STEPS Adam steps of MSE to target, so the run starts from it."""
        optimizer = torch.optim.Adam([*self.net.parameters(), self.input], lr=0.01)
        for _ in range(CONV_FIT_STEPS):
            loss = F.mse_loss(self.render(), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.last_canvas = None
        print(f"[conv] fitted to the start image in {CONV_FIT_STEPS} steps: mse {loss.item():.4f}")

    def params(self) -> list[dict]:
        """The weights at LEARNING_RATE x CONV_WEIGHT_LR, the generative input at LEARNING_RATE."""
        return [{"params": list(self.net.parameters()), "lr": LEARNING_RATE * CONV_WEIGHT_LR},
                *([] if self.inplace else [{"params": [self.input], "lr": LEARNING_RATE}])]

    def blurrable(self) -> list[torch.Tensor]:
        return [] if self.inplace else [self.input]

    def residual(self) -> torch.Tensor:
        """Inplace: (1, 3, H, W) what the net adds to the image's logits, soft-clipped to
        +-CONV_RESIDUAL_MAX (slope 1 near 0) unless that is 0."""
        r = self.net(torch.sigmoid(self.base))
        return CONV_RESIDUAL_MAX * torch.tanh(r / CONV_RESIDUAL_MAX) if CONV_RESIDUAL_MAX > 0 else r

    def render(self) -> torch.Tensor:
        if self.inplace:
            self.last_residual = self.residual()
            self.last_canvas = torch.sigmoid(self.base + self.last_residual)
        else:
            self.last_canvas = resize(torch.sigmoid(self.net(self.input)), CANVAS_RESOLUTION)
        return self.last_canvas

    def penalty(self, canvas: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """No loss; inplace reports the residual's mean |logit change| (stat "residual")."""
        stats = {"residual": self.last_residual.abs().mean().item()} if self.inplace else {}
        return canvas.new_zeros(()), stats

    def regularized(self) -> torch.Tensor:
        return self.last_canvas if self.last_canvas is not None else self.render()

    def snapshot(self, folder: Path) -> list[str]:
        """conv.pt: {"inplace", "kernel", "channels" (per layer, in -> out), "state_dict" (the
        net's weights), "input" (generative)}. -> the file names."""
        torch.save({"inplace": self.inplace, "kernel": CONV_KERNEL, "channels": self.chans,
                    "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                    "input": None if self.inplace else self.input.detach().cpu()}, folder / "conv.pt")
        return ["conv.pt"]


class RearrangeField:
    """REARRANGE: START_IMAGE's tiles, each destination a P-weighted blend of the sources that
    anneals to one tile (see REARRANGE_PERMUTATION). The field is P's logits."""

    def __init__(self, device):
        cols, rows = (max(1, v) for v in REARRANGE_GRID)
        w, h = CANVAS_RESOLUTION
        self.grid, self.tile = (cols, rows), (max(1, w // cols), max(1, h // rows))
        tw, th = self.tile
        image = load_image(START_IMAGE, (cols * tw, rows * th)).to(device)
        # (K, 3, th, tw), row-major
        self.tiles = image.reshape(3, rows, th, cols, tw).permute(1, 3, 0, 2, 4).reshape(-1, 3, th, tw)
        k = self.tiles.shape[0]
        if REARRANGE_INIT == "identity":
            # Its own tile at ~0.95 of the row at the start temperature, the rest shared.
            logits = torch.eye(k) * (math.log(k) + 3) * REARRANGE_TEMP_START
        else:
            logits = REARRANGE_INIT_STD * torch.randn(k, k)  # CPU generator: follows SEED
        self.logits = logits.to(device).requires_grad_(True)
        self.image = image
        self.total_steps, self.step = EPOCHS, 0
        self.last_p: torch.Tensor | None = None
        self.last_canvas: torch.Tensor | None = None
        print(f"[rearrange] {cols}x{rows} tiles of {tw}x{th}, "
              f"{'permutation' if REARRANGE_PERMUTATION else 'free'}, {k * k} logits")

    def params(self) -> list[dict]:
        return [{"params": [self.logits], "lr": LEARNING_RATE * REARRANGE_LR}]

    def reference(self) -> torch.Tensor:
        """The start image as tiled (every tile in its own place), at the canvas size."""
        return resize(self.image, CANVAS_RESOLUTION)

    def blurrable(self) -> list[torch.Tensor]:
        return []

    def progress(self) -> float:
        return min(1.0, self.step / max(1, self.total_steps))

    def temperature(self) -> float:
        t0, t1 = max(1e-4, REARRANGE_TEMP_START), max(1e-4, REARRANGE_TEMP_END)
        return t0 * (t1 / t0) ** self.progress()

    def soft(self) -> torch.Tensor:
        """(K, K) rows = destinations, each summing to 1 (and columns ~1 too with permutations)."""
        log_p = self.logits / self.temperature()
        if REARRANGE_PERMUTATION:
            for _ in range(max(0, REARRANGE_SINKHORN_ITERS)):
                log_p = log_p - log_p.logsumexp(dim=0, keepdim=True)
                log_p = log_p - log_p.logsumexp(dim=1, keepdim=True)
            return log_p.exp()
        return log_p.softmax(dim=1)

    def hard(self, p: torch.Tensor) -> torch.Tensor:
        """(K,) the source of each destination: the most probable permutation (Hungarian) or argmax."""
        if not REARRANGE_PERMUTATION:
            return p.argmax(dim=1)
        from scipy.optimize import linear_sum_assignment
        _, src = linear_sum_assignment(-torch.log(p.detach().float().clamp_min(1e-12)).cpu().numpy())
        return torch.as_tensor(src, device=p.device)

    def render(self) -> torch.Tensor:
        p = self.soft()
        if self.progress() >= REARRANGE_HARD_AFTER:
            p = F.one_hot(self.hard(p), p.shape[0]).to(p) + p - p.detach()  # straight-through
        self.last_p = p
        cols, rows = self.grid
        tw, th = self.tile
        out = torch.einsum("ds,schw->dchw", p, self.tiles)
        image = out.reshape(rows, cols, 3, th, tw).permute(2, 0, 3, 1, 4).reshape(1, 3, rows * th, cols * tw)
        self.last_canvas = resize(image, CANVAS_RESOLUTION)
        return self.last_canvas

    def regularized(self) -> torch.Tensor:
        return self.last_canvas if self.last_canvas is not None else self.render()

    def penalty(self, canvas: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """REARRANGE_ENTROPY_WEIGHT x progress x P's mean row entropy (/ log K). Stats: "temp",
        "entropy", "one-hot" (mean of each row's largest weight) and "sources" (share of the tiles
        the hard assignment uses). Advances the anneal by one step."""
        p = self.soft()
        k = p.shape[0]
        entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=1).mean() / max(math.log(k), 1e-12)
        stats = {"temp": self.temperature(), "entropy": entropy.item(),
                 "one-hot": p.detach().max(dim=1).values.mean().item(),
                 "sources": self.hard(p).unique().numel() / k}
        loss = REARRANGE_ENTROPY_WEIGHT * self.progress() * entropy
        self.step += 1
        return loss, stats

    def assignment_image(self, cell: int = 0) -> torch.Tensor:
        """(1, 3, H, W) P as a K x K grey matrix (rows = destinations, white = 1), cell px per entry."""
        p = (self.last_p if self.last_p is not None else self.soft()).detach().float().clamp(0, 1)
        cell = cell or max(1, 256 // p.shape[0])
        grid = F.interpolate(p[None, None], scale_factor=cell, mode="nearest")
        return grid.expand(-1, 3, -1, -1)

    def snapshot(self, folder: Path) -> list[str]:
        """rearrange.json ({"grid", "tile", "source": the hard source tile of every destination,
        row-major}) and assignment.npy ((K, K) soft P). -> the file names."""
        p = self.soft().detach()
        (folder / "rearrange.json").write_text(json.dumps(
            {"grid": list(self.grid), "tile": list(self.tile), "source": self.hard(p).tolist()}))
        np.save(folder / "assignment.npy", p.float().cpu().numpy())
        return ["rearrange.json", "assignment.npy"]


def grad_to_bgr(grad: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) canvas gradient as a BGR heat map of its magnitude per pixel (inferno,
    black = none), scaled to the 99th percentile so a few hot pixels don't wash it out."""
    mag = grad[0].float().norm(dim=0)
    hi = max(torch.quantile(mag.flatten()[::max(1, mag.numel() // 65536)], 0.99).item(), 1e-12)
    x = (mag / hi).clamp(0, 1).cpu().numpy()
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def make_field(device):
    if FIELD_TYPE == FieldType.PIXELS_COLOR:
        return PixelField(gray=False, device=device)
    if FIELD_TYPE == FieldType.PIXELS_GRAY:
        return PixelField(gray=True, device=device)
    if FIELD_TYPE == FieldType.SHAPES:
        return ShapeField(device)
    if FIELD_TYPE == FieldType.PIXEL_PATCHES:
        return PatchField(device)
    if FIELD_TYPE == FieldType.GAUSSIANS:
        return GaussianField(device)
    if FIELD_TYPE == FieldType.SURFACE:
        return SurfaceField(device)
    if FIELD_TYPE == FieldType.LEVELSET:
        return LevelSetField(device)
    if FIELD_TYPE == FieldType.VAE:
        return VAEField(device)
    if FIELD_TYPE == FieldType.CONVOLUTIONS:
        return ConvField(device)
    if FIELD_TYPE == FieldType.REARRANGE:
        return RearrangeField(device)
    if FIELD_TYPE == FieldType.FUNCTION_STACK:
        return FunctionStackField(device)
    if FIELD_TYPE == FieldType.RELIEF:
        return ReliefField(device)
    if FIELD_TYPE == FieldType.REVOLUTION:
        return RevolutionField(device)
    if FIELD_TYPE == FieldType.MESH_TEXTURE:
        return MeshTextureField(device)
    if FIELD_TYPE == FieldType.GAUSSIANS_3D:
        return Gaussians3DField(device)
    if FIELD_TYPE == FieldType.VOXELS:
        return VoxelField(device)
    if FIELD_TYPE in (FieldType.ASSEMBLY, FieldType.SOLIDS_ON_OFF, FieldType.EXTRUSIONS):
        return SolidField(FIELD_TYPE.name, device)
    return DistortionField(device)


# ── Loss ───────────────────────────────────────────────────────────────────────

_UNIT_CORNERS = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])  # TL, TR, BR, BL


def view_size(width: int, height: int) -> tuple[int, int]:
    """(width, height) of augment's views: the canvas' aspect, short side the models' INPUT_SIZE
    (square canvas: exactly their input, so their own resize is a no-op)."""
    scale = INPUT_SIZE / min(width, height)
    return round(width * scale), round(height * scale)


def view_corners(n: int, width: int, height: int) -> torch.Tensor:
    """(n, 4, 2) where on the canvas (normalized: [-1, 1] spans it) the corners of n random
    views (TL, TR, BR, BL) sample from: a CROP_SCALE / CROP_RATIO window, PERSPECTIVE-pulled
    corners, FLIP, ROTATION, then placed uniformly where it stays within VIEW_MARGIN of the
    canvas (shrunk about its centre first if it cannot)."""
    area = width * height * torch.empty(n).uniform_(*CROP_SCALE)
    ratio = torch.empty(n).uniform_(math.log(CROP_RATIO[0]), math.log(CROP_RATIO[1])).exp()
    half_size = torch.stack([(area * ratio).sqrt().clamp(max=width), (area / ratio).sqrt().clamp(max=height)], -1) / 2
    pts = _UNIT_CORNERS * half_size[:, None]  # (n, 4, 2) canvas pixels about the view's centre
    if PERSPECTIVE > 0:
        warped = (torch.rand(n, 1, 1) < 0.5).float()
        pts = pts * (1 - PERSPECTIVE * warped * torch.rand(n, 4, 2))
    if FLIP:
        pts[..., 0] *= torch.where(torch.rand(n, 1) < 0.5, -1.0, 1.0)
    theta = torch.deg2rad(torch.empty(n).uniform_(-ROTATION, ROTATION))
    cos, sin = theta.cos(), theta.sin()
    rot = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)  # (n, 2, 2)
    pts = pts @ rot.transpose(1, 2)
    bound = torch.tensor([width, height]) * (1 + VIEW_MARGIN) / 2
    lo, hi = pts.amin(1), pts.amax(1)  # (n, 2) bounding box
    fit = (2 * bound / (hi - lo)).amin(-1).clamp(max=1)[:, None]
    pts, lo, hi = pts * fit[:, None], lo * fit, hi * fit
    centre = -bound - lo + (2 * bound - (hi - lo)) * torch.rand(n, 2)
    return (pts + centre[:, None]) * 2 / torch.tensor([width, height])


def corner_homographies(corners: torch.Tensor) -> torch.Tensor:
    """(n, 3, 3) projective maps taking the view's own [-1, 1] square onto corners (n, 4, 2)."""
    n = corners.shape[0]
    x, y = (_UNIT_CORNERS[:, i].expand(n, 4).double() for i in (0, 1))
    u, v = corners[..., 0].double(), corners[..., 1].double()
    zero, one = torch.zeros_like(u), torch.ones_like(u)
    a = torch.cat([torch.stack([x, y, one, zero, zero, zero, -u * x, -u * y], -1),
                   torch.stack([zero, zero, zero, x, y, one, -v * x, -v * y], -1)], 1)  # (n, 8, 8)
    h = torch.linalg.solve(a, torch.cat([u, v], 1))
    return torch.cat([h, torch.ones(n, 1, dtype=h.dtype)], 1).view(n, 3, 3).float()


def warp_views(canvas: torch.Tensor, corners: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """(n, 3, h, w) at size (width, height): the canvas (1, 3, H, W) sampled through each view's
    corners, outside it mid-grey. grid_sample does not filter, so each view samples a copy of the
    canvas first shrunk (antialiased) to about its own scale: no aliasing, and ~1:1 bilinear."""
    (w, h), (H, W) = size, canvas.shape[-2:]
    homs = corner_homographies(corners).to(canvas.device)
    xs = (torch.arange(w, device=canvas.device) * 2 + 1) / w - 1
    ys = (torch.arange(h, device=canvas.device) * 2 + 1) / h - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx, gy, torch.ones_like(gx)], -1).view(1, -1, 3) @ homs.transpose(1, 2)
    grids = (pts[..., :2] / pts[..., 2:]).view(-1, h, w, 2)
    # Shoelace area of each view on the canvas, in canvas pixels, over its own: the squared minification.
    cx, cy = corners[..., 0] * W / 2, corners[..., 1] * H / 2
    area = 0.5 * (cx * cy.roll(-1, 1) - cx.roll(-1, 1) * cy).sum(1).abs()
    views = []
    for grid, minify in zip(grids, (area / (w * h)).sqrt().tolist()):
        src = canvas if minify <= 1 else resize(canvas, (max(1, round(W / minify)), max(1, round(H / minify))))
        views.append(F.grid_sample(src - 0.5, grid[None], mode="bilinear", padding_mode="zeros",
                                   align_corners=False) + 0.5)
    return torch.cat(views)


def augment(canvas: torch.Tensor, count: int | None = None) -> torch.Tensor:
    """(N, 3, h, w) at view_size: `count` (default RANDOM_TRANSFORM_COUNT) random views (+ the bare
    canvas), each with its own colour jitter, then pixel noise. A view's crop, flip, rotation and
    perspective are one homography (view_corners), so the canvas is resampled once -- not
    once per transform and again by the model -- and placing the view after rotating it keeps
    the grey fill to VIEW_MARGIN."""
    height, width = canvas.shape[-2:]
    size = view_size(width, height)
    views = [resize(canvas, size)] if INCLUDE_UNTRANSFORMED else []
    count = RANDOM_TRANSFORM_COUNT if count is None else count
    if count:
        # Bicubic resampling overshoots [0, 1] at hard edges (high-contrast LEVELSET masks), and
        # the hue jitter's HSV conversion turns such values into NaN for saturated colours.
        warped = warp_views(canvas, view_corners(count, width, height), size).clamp(0, 1)
        views += [COLOR_JITTER(view[None]) for view in warped]
    batch = torch.cat(views)
    if TRANSFORM_NOISE > 0:
        batch = batch + TRANSFORM_NOISE * torch.randn_like(batch)
    return batch.clamp(0, 1)


def prompt_label(prompt: Prompt) -> str:
    if isinstance(prompt, ImagePrompt):
        return f"img:{Path(prompt.path).stem}"
    if isinstance(prompt, VLMQuery):
        return f"{prompt.question} {prompt.answer}"
    return prompt if isinstance(prompt, str) else prompt[0] + ("..." if len(prompt) > 1 else "")


def prompt_image(prompt) -> str | None:
    """The image path of an ImagePrompt (for the page's thumbnails), else None."""
    return prompt.path if isinstance(prompt, ImagePrompt) else None


def load_prompt_image(prompt: ImagePrompt, device) -> torch.Tensor:
    """(1, 3, H, W) on the canvas, framed like the start image."""
    return load_image(prompt.path, CANVAS_RESOLUTION).to(device)


def concept_embedding(central: ModelCentral, spec: ModelSpec, prompt: Prompt, device) -> torch.Tensor:
    """(D,) unit vector: mean CLIP text embedding over the spec's templates x paraphrases,
    or the image embedding (CLIP / DINOv2) of an ImagePrompt."""
    if isinstance(prompt, ImagePrompt):
        with torch.no_grad():
            image = load_prompt_image(prompt, device)
            return F.normalize(central.image_embedding(image, spec.central_id(), spec.tokens).float()[0], dim=-1)
    if spec.model_id.is_dino():
        raise ValueError(f"DINOv2 has no text encoder -- targets must be images, got {prompt_label(prompt)!r}")
    phrases = [prompt] if isinstance(prompt, str) else prompt
    texts = [template.format(p) for p in phrases for template in spec.text_templates()]
    return F.normalize(central.clip_text_embeddings(texts, device, spec.central_id()).mean(0), dim=-1)


class ClipObjective:
    word_ends = "text"  # what a target end that isn't an image is (Objective.axes "target")
    """Weighted CLIP similarities to one ModelSpec's targets and directions. `sign` flips
    their weights: the - pole of a BIDIRECTIONAL run. Also serves DINOv2, whose targets
    and directions are all images (ImagePrompt)."""

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        """(N, D) unit per-image vectors the targets are scored on (CLIP or DINOv2)."""
        return self.central.image_embedding(images, self.spec.central_id(), self.spec.tokens).float()

    def _concept(self, prompt: Prompt, device) -> torch.Tensor:
        """(D,) unit vector of a text / image target, see concept_embedding."""
        return concept_embedding(self.central, self.spec, prompt, device)

    def __init__(self, central: ModelCentral, spec: ModelSpec, device):
        self.central = central
        self.spec = spec
        self.sign = 1.0
        self.active: int | None = None  # sequential mode: the one target index scored (see active_weights)
        self.text_emb = None
        if spec.targets:
            self.text_emb = torch.stack([self._concept(t, device) for t, _ in spec.targets])
            self.text_w = torch.tensor([w for _, w in spec.targets], device=device)
        self.directions = None
        if spec.directions:
            self.directions = torch.stack([
                F.normalize(self._concept(t, device) - self._concept(s, device), dim=-1)
                for s, t, _ in spec.directions])  # (K, D)
            self.direction_w = torch.tensor([w for _, _, w in spec.directions], device=device)
        self.start_emb = None
        self.start_clean = None
        self.start_axes = None
        # Per-axis values of the last __call__, relative to the start (see axes()):
        # (K,) for the view mean and (N, K) per augmented view.
        self.last_axes: np.ndarray | None = None
        self.last_view_axes: np.ndarray | None = None

    def target_count(self) -> int:
        """Targets + directions: the index range of `active` (also their axes' order)."""
        return len(self.spec.targets) + len(self.spec.directions)

    def active_weights(self, w: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Weights of targets offset.. with all but the active one zeroed (all kept when
        `active` is None)."""
        if self.active is None:
            return w
        return w * (torch.arange(len(w), device=w.device) + offset == self.active)

    def axes(self) -> list[dict]:
        """One entry per optimization target, in the order of the last_axes columns.
        Text axes are similarity minus the start's; direction axes are the cosine
        between the embedding change and the text direction."""
        axes = [{"kind": "text", "label": prompt_label(t), "image": prompt_image(t), "weight": w}
                for t, w in self.spec.targets]
        axes += [{"kind": "direction", "label": prompt_label(t), "source": prompt_label(s),
                  "image": prompt_image(t), "source_image": prompt_image(s), "weight": w}
                 for s, t, w in self.spec.directions]
        return axes

    def _axis_values(self, emb: torch.Tensor) -> torch.Tensor:
        """(N, K) raw axis values for (N, D) embeddings: text similarities and, per row,
        the direction cosine of its change from the start."""
        cols = []
        if self.text_emb is not None:
            cols.append(emb @ self.text_emb.T)
        if self.directions is not None and self.start_emb is not None:
            cols.append(F.normalize(emb - self.start_emb, dim=-1, eps=1e-6) @ self.directions.T)
        return torch.cat(cols, dim=1) if cols else emb.new_zeros(len(emb), 0)

    def _direction_mask(self) -> torch.Tensor:
        """(K,) bool: which axes are directions (already relative to the start)."""
        n_text = 0 if self.text_emb is None else len(self.text_emb)
        n_dir = 0 if self.directions is None else len(self.directions)
        return torch.tensor([False] * n_text + [True] * n_dir, dtype=torch.bool)

    def set_start(self, canvas: torch.Tensor, sample_views: Callable[[], torch.Tensor] | None = None) -> None:
        """Freeze the reference points: the bare start embedding (edit size, see
        similarity_to_start) and the mean embedding of augmented views of the start (same
        distribution as the loss sees; sample_views draws one batch, default augment(canvas))
        -- the origin for directions and for the axes."""
        sample_views = sample_views or (lambda: augment(canvas.detach()))
        with torch.no_grad():
            self.start_clean = F.normalize(self._embed(canvas.detach()), dim=-1)[0]
            embs, count = [], 0
            while count < START_DIRECTION_VIEWS:
                embs.append(self._embed(sample_views()))
                count += len(embs[-1])
            self.start_emb = torch.cat(embs).mean(0)
            start = self._axis_values(torch.cat(embs)).mean(0)
            self.start_axes = torch.where(self._direction_mask().to(start.device), 0.0, start)

    def similarity_to_start(self, canvas: torch.Tensor) -> float:
        """Cosine between the bare canvas and the bare start embeddings (CLIP; ImageNet:
        logit vectors): the edit-size measure."""
        with torch.no_grad():
            emb = F.normalize(self._embed(canvas.detach()), dim=-1)[0]
        return (emb @ self.start_clean).item()

    def _mean_axis_values(self, emb: torch.Tensor) -> torch.Tensor:
        """(K,) axis values of the view batch: those of the mean embedding, like the loss."""
        return self._axis_values(emb.mean(0, keepdim=True))[0]

    def __call__(self, views: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        emb = self._embed(views)  # (N, D)
        loss, stats = self._loss(emb)
        if self.start_axes is not None:
            with torch.no_grad():
                emb = emb.detach()
                per_view = self._axis_values(emb) - self.start_axes
                mean = self._mean_axis_values(emb) - self.start_axes
            self.last_view_axes = per_view.cpu().numpy()
            self.last_axes = mean.cpu().numpy()
        return loss, stats

    def _loss(self, emb: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        loss = torch.zeros((), device=emb.device)
        stats = {}
        if self.text_emb is not None:
            sims = (emb @ self.text_emb.T).mean(0)  # (T,)
            loss = loss - self.sign * (sims * self.active_weights(self.text_w)).sum()
            stats.update({prompt_label(t): s.item() for (t, _), s in zip(self.spec.targets, sims)})
        if self.directions is not None and self.start_emb is not None:
            # Mean over views first: a per-view delta is dominated by augmentation noise.
            delta = F.normalize(emb.mean(0) - self.start_emb, dim=-1, eps=1e-6)  # (D,)
            cos = self.directions @ delta                                         # (K,)
            n_text = 0 if self.text_emb is None else len(self.text_emb)
            loss = loss - self.sign * (cos * self.active_weights(self.direction_w, n_text)).sum()
            stats.update({f"{prompt_label(s)}->{prompt_label(t)}": c.item()
                          for (s, t, _), c in zip(self.spec.directions, cos)})
        return loss, stats


def imagenet_classes() -> list[str]:
    """The 1000 ImageNet-1k class names, in logit order (the same for every torchvision
    IMAGENET1K weights)."""
    return ModelCentral.imagenet_classes(ModelID.imagenet(ImageNetArch.RESNET50, "IMAGENET1K_V2"))


def class_label(target: int | ImagePrompt, names: list[str]) -> str:
    return prompt_label(target) if isinstance(target, ImagePrompt) else names[target]


class ClassifierObjective(ClipObjective):
    word_ends = "class"
    """ClipObjective's counterpart for a torchvision ImageNet classifier. A class target
    scores the class by spec.score (logits or softmax), a class direction scores
    score[target] - score[source]; both averaged over the views and relative to the start.
    Image targets and the edit-size measure use the cosine between logit
    vectors; an image direction is the cosine between the change of the mean logits from
    the start and L(target) - L(source), like a CLIP direction."""

    def __init__(self, central: ModelCentral, spec: ModelSpec, device):
        self.central = central
        self.spec = spec
        self.sign = 1.0
        self.active: int | None = None
        self.start_emb = None
        self.start_clean = None
        names = self.class_names()
        for c in [c for c, _ in spec.targets] + [e for s, t, _ in spec.directions for e in (s, t)]:
            if isinstance(c, int) and not 0 <= c < len(names):
                raise ValueError(f"class {c}: the model has classes 0 to {len(names) - 1}")
        # Targets: a class index, or the unit logit vector of an image.
        self.targets = [c if isinstance(c, int) else F.normalize(self._image_logits(c, device), dim=-1)
                        for c, _ in spec.targets]
        # Directions: a (source, target) class pair, or a unit logit-space direction between images.
        self.directions = []
        for src, tgt, _ in spec.directions:
            if isinstance(src, ImagePrompt) != isinstance(tgt, ImagePrompt):
                raise ValueError("A class direction joins two classes or two images, not one of each")
            self.directions.append((src, tgt) if isinstance(src, int) else F.normalize(
                self._image_logits(tgt, device) - self._image_logits(src, device), dim=-1))
        self.weights = torch.tensor([w for _, w in spec.targets] + [w for _, _, w in spec.directions],
                                    device=device)
        self.labels = ([class_label(c, names) for c, _ in spec.targets]
                       + [f"{class_label(s, names)}->{class_label(t, names)}" for s, t, _ in spec.directions])
        self._axes = [{"kind": "text", "label": class_label(c, names), "image": prompt_image(c), "weight": w}
                      for c, w in spec.targets]
        self._axes += [{"kind": "direction", "label": class_label(t, names), "source": class_label(s, names),
                        "image": prompt_image(t), "source_image": prompt_image(s), "weight": w}
                       for s, t, w in spec.directions]
        self.start_axes = None
        self.last_axes: np.ndarray | None = None
        self.last_view_axes: np.ndarray | None = None

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.central.imagenet_logits(images, self.spec.model_id).float()  # (N, 1000)

    def class_names(self) -> list[str]:
        return imagenet_classes()

    def target_count(self) -> int:
        return len(self.weights)

    def _image_logits(self, prompt: ImagePrompt, device) -> torch.Tensor:
        with torch.no_grad():
            return self._embed(load_prompt_image(prompt, device))[0]

    def _scores(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.softmax(-1) if self.spec.score == "softmax" else logits

    def axes(self) -> list[dict]:
        return list(self._axes)

    def _axis_values(self, emb: torch.Tensor) -> torch.Tensor:
        """(N, K) per row: class scores / logit cosines to target images, target - source
        class scores / direction cosines of the change from the start."""
        scores = self._scores(emb)
        unit = F.normalize(emb, dim=-1)
        cols = [scores[:, t] if isinstance(t, int) else unit @ t for t in self.targets]
        for d in self.directions:
            if isinstance(d, tuple):
                cols.append(scores[:, d[1]] - scores[:, d[0]])
            elif self.start_emb is None:
                cols.append(emb.new_zeros(len(emb)))
            else:
                cols.append(F.normalize(emb - self.start_emb, dim=-1, eps=1e-6) @ d)
        return torch.stack(cols, dim=1) if cols else emb.new_zeros(len(emb), 0)

    def _direction_mask(self) -> torch.Tensor:
        """(K,) bool: the image directions (already relative to the start)."""
        return torch.tensor([False] * len(self.targets)
                            + [not isinstance(d, tuple) for d in self.directions], dtype=torch.bool)

    def _mean_axis_values(self, emb: torch.Tensor) -> torch.Tensor:
        # Mean of the per-view scores, like the loss (softmax of the mean logits would differ);
        # image directions use the mean logits' change, as CLIP directions do.
        per_view = self._axis_values(emb).mean(0)
        of_mean = self._axis_values(emb.mean(0, keepdim=True))[0]
        return torch.where(self._direction_mask().to(emb.device), of_mean, per_view)

    def _loss(self, emb: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        values = self._mean_axis_values(emb)
        n = len(self.weights)
        loss = -self.sign * (values[:n] * self.active_weights(self.weights)).sum()
        stats = {label: v.item() for label, v in zip(self.labels, values[:n])}
        return loss, stats


def custom_model(spec: ModelSpec, device):
    """The differentiable custom model of a spec (loaded once)."""
    from shared.custom_models import differentiable
    return differentiable.load(spec.model_id.arch, device, precision=spec.precision)


class CustomClassifierObjective(ClassifierObjective):
    """ClassifierObjective for a classifier trained in custom_models: its own classes."""

    def __init__(self, central: ModelCentral, spec: ModelSpec, device):
        self.model = custom_model(spec, device)
        super().__init__(central, spec, device)

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)  # (N, classes) logits

    def class_names(self) -> list[str]:
        return self.model.class_names


class CustomEmbeddingObjective(ClipObjective):
    """ClipObjective in a custom autoencoder's latent space (as unit vectors). No text
    encoder: like DINOv2, the targets and directions are images."""

    def __init__(self, central: ModelCentral, spec: ModelSpec, device):
        self.model = custom_model(spec, device)
        super().__init__(central, spec, device)

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model(images), dim=-1)

    def _concept(self, prompt: Prompt, device) -> torch.Tensor:
        if not isinstance(prompt, ImagePrompt):
            raise ValueError(f"An autoencoder has no text encoder -- targets must be images, got {prompt_label(prompt)!r}")
        with torch.no_grad():
            return self._embed(load_prompt_image(prompt, device))[0]


class ScoreObjective(ClipObjective):
    word_ends = "score"
    """A RankNet or an aesthetic score: one number per image, mean over the views, to
    maximize, minimize or bring to spec.value (spec.goal), weighted spec.weight. `sign`
    (the - pole) swaps maximize and minimize. The edit size (similarity_to_start) is the
    cosine between the features the score is computed from (the RankNet's backbone
    embeddings, the scorer's CLIP embedding)."""

    def __init__(self, central: ModelCentral, spec: ModelSpec, device):
        if spec.goal not in ("maximize", "minimize", "value"):
            raise ValueError(f"Bad score goal: {spec.goal!r}")
        self.central = central
        self.spec = spec
        self.model = custom_model(spec, device)
        # PickScore / HPSv2: this spec's own prompt (another spec may score against another one)
        self.text = self.model.prompt_embedding(spec.prompt) if self.model.uses_prompt else None
        self.sign = 1.0
        self.active: int | None = None
        self.text_emb = self.directions = None
        self.start_emb = self.start_clean = None
        self.start_axes = None
        self.last_axes: np.ndarray | None = None
        self.last_view_axes: np.ndarray | None = None

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.features(images)  # (N, D), not unit: the score head takes them as they are

    def _goal_label(self) -> str:
        return f"to {self.spec.value:g}" if self.spec.goal == "value" else self.spec.goal

    def axes(self) -> list[dict]:
        spec = self.spec
        weight = -spec.weight if spec.goal == "minimize" else spec.weight
        prompt = f' "{spec.prompt}"' if self.model.uses_prompt else ""
        return [{"kind": "text", "label": f"{self.model.entry['label']}{prompt} score ({self._goal_label()})",
                 "weight": weight}]

    def _axis_values(self, emb: torch.Tensor) -> torch.Tensor:
        """(N, 1): the score of every view."""
        return self.model.score_of(emb, self.text).unsqueeze(1)

    def _direction_mask(self) -> torch.Tensor:
        return torch.tensor([False])

    def _mean_axis_values(self, emb: torch.Tensor) -> torch.Tensor:
        return self._axis_values(emb).mean(0)  # mean of the per-view values, like the loss

    def target_count(self) -> int:
        return 1

    def _loss(self, emb: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        spec = self.spec
        score = self._mean_axis_values(emb)[0]
        weight = spec.weight if self.active in (None, 0) else 0.0  # sequential mode: only on its turn
        if spec.goal == "value":
            loss = weight * (score - spec.value) ** 2
        else:
            loss = (-1.0 if spec.goal == "maximize" else 1.0) * self.sign * weight * score
        return loss, {"score": score.item()}


class VLMObjective(ClassifierObjective):
    word_ends = "answer"
    """A vision-language model asked about the image (shared.models.vlm): a target
    VLMQuery(question, answer) scores log p(answer | image, question) (spec.score: "logprob"
    the answer's, "mean" per token), a direction (source, target) scores logp(target) -
    logp(source) -- No -> Yes: the log-odds of a yes / no question. Like a classifier's
    classes, each distinct query is a column of the (N, Q) scores of the views; means over
    the views, relative to the start. The edit size is the cosine between the model's image
    features (its mean projected image token)."""

    def __init__(self, central: ModelCentral, spec: ModelSpec, device):
        from shared.models import vlm
        if spec.score not in vlm.SCORES:
            raise ValueError(f"Bad VLM score {spec.score!r}: one of {vlm.SCORES}")
        ends = [t for t, _ in spec.targets] + [e for s, t, _ in spec.directions for e in (s, t)]
        if not ends:
            raise ValueError(f"{spec.short_name()}: a VLM needs a target or a direction (question, answer)")
        bad = [e for e in ends if not isinstance(e, VLMQuery)]
        if bad:
            raise ValueError(f"{spec.short_name()}: VLM targets are (question, answer) queries, "
                             f"not {prompt_label(bad[0])!r}")
        self.central = central
        self.spec = spec
        self.sign = 1.0
        self.active: int | None = None
        self.start_emb = self.start_clean = None
        self.model = vlm.load(spec.model_id.arch, device, spec.precision)
        pairs = list(dict.fromkeys((e.question, e.answer) for e in ends))  # distinct, in order
        self.queries = self.model.queries(pairs)
        column = lambda e: pairs.index((e.question, e.answer))
        self.targets = [column(t) for t, _ in spec.targets]
        self.directions = [(column(s), column(t)) for s, t, _ in spec.directions]
        self.weights = torch.tensor([w for _, w in spec.targets] + [w for _, _, w in spec.directions],
                                    device=device)
        self.labels = ([prompt_label(t) for t, _ in spec.targets]
                       + [vlm_direction_label(s, t) for s, t, _ in spec.directions])
        self._axes = [{"kind": "text", "label": prompt_label(t), "weight": w} for t, w in spec.targets]
        self._axes += [{"kind": "direction", "label": prompt_label(t), "source": prompt_label(s), "weight": w}
                       for s, t, w in spec.directions]
        self.start_features = None
        self.start_axes = None
        self.last_axes: np.ndarray | None = None
        self.last_view_axes: np.ndarray | None = None

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images, self.queries, self.spec.score)  # (N, Q) answer log-probabilities

    def _scores(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def _features(self, canvas: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.normalize(self.model.features(canvas.detach(), self.queries), dim=-1)[0]

    def set_start(self, canvas: torch.Tensor, sample_views: Callable[[], torch.Tensor] | None = None) -> None:
        super().set_start(canvas, sample_views)
        self.start_features = self._features(canvas)

    def similarity_to_start(self, canvas: torch.Tensor) -> float:
        return (self._features(canvas) @ self.start_features).item()


def vlm_direction_label(source: VLMQuery, target: VLMQuery) -> str:
    """'question a->b' for two answers to one question, else both queries"""
    if source.question == target.question:
        return f"{source.question} {source.answer}->{target.answer}"
    return f"{prompt_label(source)}->{prompt_label(target)}"


def model_objective(central: ModelCentral, spec: ModelSpec, device) -> ClipObjective:
    """The objective of one spec: ClassifierObjective for an ImageNet model, a custom
    objective for a custom model (by its kind), VLMObjective for a VLM, else ClipObjective
    (CLIP or DINOv2)."""
    if spec.model_id.is_custom():
        kind = custom_model(spec, device).kind
        cls = {"classifier": CustomClassifierObjective, "autoencoder": CustomEmbeddingObjective}.get(kind, ScoreObjective)
    elif spec.model_id.is_vlm():
        cls = VLMObjective
    else:
        cls = ClassifierObjective if spec.model_id.is_imagenet() else ClipObjective
    return cls(central, spec, device)


# ModelSpec fields saved with an experiment (besides the model)
SPEC_KEYS = ("targets", "directions", "score", "tokens", "prompt", "goal", "value", "weight", "precision")


class Objective:
    """Every model of MODELS at once: the sum of their losses (`sign` flips each one's
    targets, see BIDIRECTIONAL), their axes side by side, and the image matching terms
    (`match`, run_pole adds them). Stats and axes are named by model when there are
    several. The edit size (similarity_to_start) is the first model's; without models,
    1 - the RMS pixel difference to the start."""

    def __init__(self, central: ModelCentral, device):
        self.parts = [model_objective(central, spec, device) for spec in MODELS]
        names = [spec.short_name() for spec in MODELS]
        # the same model twice (other targets): "name", "name #2"
        self.names = [n if names[:i].count(n) == 0 else f"{n} #{names[:i].count(n) + 1}" for i, n in enumerate(names)]
        self.match = ImageMatch.create(device)  # rgb / gray / depth / edges vs the target, on the bare canvas
        self.field = None
        self.start: torch.Tensor | None = None
        self._sign = 1.0
        self._active: int | None = None
        self.last_axes: np.ndarray | None = None
        self.last_view_axes: np.ndarray | None = None
        self.last_canvas_grad: torch.Tensor | None = None  # CANVAS_GRAD: (1, C, H, W) of the last step

    @property
    def sign(self) -> float:
        return self._sign

    @sign.setter
    def sign(self, value: float) -> None:
        self._sign = value
        for part in self.parts:
            part.sign = value

    def target_count(self) -> int:
        """Every part's targets + directions, in axes order: the index range of `active`."""
        return sum(part.target_count() for part in self.parts)

    @property
    def active(self) -> int | None:
        return self._active

    @active.setter
    def active(self, value: int | None) -> None:
        """Sequential mode: the one target scored, an index into axes(). Each part gets it
        relative to its own first axis (out of its range: none of its targets scored)."""
        self._active = value
        offset = 0
        for part in self.parts:
            part.active = None if value is None else value - offset
            offset += part.target_count()

    def axes(self) -> list[dict]:
        """Every part's axes with its model name and "target": what the axis is about --
        "text" (CLIP), "class" (classifiers), "image" (an image target, any model), "score"."""
        return [{**axis, "model": name, "target": "image" if axis.get("image") else part.word_ends}
                for name, part in zip(self.names, self.parts) for axis in part.axes()]

    def set_start(self, canvas: torch.Tensor, sample_views: Callable[[], torch.Tensor] | None = None) -> None:
        self.start = canvas.detach()
        for part in self.parts:
            part.set_start(canvas, sample_views)

    def similarity_to_start(self, canvas: torch.Tensor) -> float:
        if self.parts:
            return self.parts[0].similarity_to_start(canvas)
        with torch.no_grad():
            return 1.0 - (canvas.detach() - self.start).pow(2).mean().sqrt().item()

    def __call__(self, views: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        loss, stats = views.new_zeros(()), {}
        for name, part in zip(self.names, self.parts):
            part_loss, part_stats = part(views)
            loss = loss + part_loss
            stats.update({(f"{name}: {k}" if len(self.parts) > 1 else k): v for k, v in part_stats.items()})
        if self.start is not None and all(part.last_axes is not None for part in self.parts):
            self.last_axes = np.concatenate([p.last_axes for p in self.parts]) if self.parts else np.zeros(0)
            self.last_view_axes = (np.concatenate([p.last_view_axes for p in self.parts], axis=1) if self.parts
                                   else np.zeros((len(views), 0)))
        return loss, stats


def make_objective(central: ModelCentral, device, log: RunLog | None = None) -> Objective:
    """The Objective of MODELS. Only their models stay loaded: the gpu memory of the
    others is freed before anything new loads (`log` notes the memory after each)."""
    from shared.custom_models import differentiable
    from shared.models import vlm
    differentiable.release(keep={differentiable.cache_key(s.model_id.arch, device, s.precision)
                                 for s in MODELS if s.model_id.is_custom()})
    vlm.release(keep={vlm.cache_key(s.model_id.arch, device, s.precision) for s in MODELS if s.model_id.is_vlm()})
    central.release(keep={s.central_id() for s in MODELS if not (s.model_id.is_custom() or s.model_id.is_vlm())})
    if log:
        log.event("released the models this run doesn't use")
    objective = Objective(central, device)
    if log:
        log.event("models loaded: " + (", ".join(spec.short_name() for spec in MODELS) or "none"))
    return objective


# ── Preview / output ───────────────────────────────────────────────────────────

WINDOW = "CLIP optimization"


def to_bgr(img: torch.Tensor) -> np.ndarray:
    arr = (img[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def save_tensors(folder: Path, tensors: dict[str, torch.Tensor]) -> list[str]:
    """Each (1, C, H, W) tensor in [0, 1] as <name>.npy ((C, H, W) float32) and <name>.png
    into folder. -> the file names."""
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for name, t in tensors.items():
        t = t.detach().float()
        np.save(folder / f"{name}.npy", t[0].cpu().numpy())
        cv2.imwrite(str(folder / f"{name}.png"), to_bgr(t.expand(-1, 3, -1, -1) if t.shape[1] == 1 else t))
        files += [f"{name}.npy", f"{name}.png"]
    return files


def put_lines(img: np.ndarray, lines: list[str], x: int = 8, y: int = 20) -> None:
    for i, line in enumerate(lines):
        yy = y + 18 * i
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def show_preview(canvas: torch.Tensor, views: torch.Tensor, title: str, stats: dict[str, float],
                 original: np.ndarray | None = None) -> int:
    """Canvas at PREVIEW_SIZE (START_IMAGE beside it, if any), a strip of augmented views
    below, stats overlaid. Returns key."""
    size = (PREVIEW_SIZE, PREVIEW_SIZE * canvas.shape[-2] // canvas.shape[-1])
    main = cv2.resize(to_bgr(canvas), size, interpolation=cv2.INTER_NEAREST)
    if original is not None:
        main = np.hstack([cv2.resize(original, size, interpolation=cv2.INTER_AREA), main])
    width = main.shape[1]
    n = min(len(views), 6 * width // PREVIEW_SIZE)
    tile = width // n
    strip = np.hstack([cv2.resize(to_bgr(views[i:i + 1]), (tile, tile)) for i in range(n)])
    strip = cv2.copyMakeBorder(strip, 0, 0, 0, width - strip.shape[1], cv2.BORDER_CONSTANT)
    frame = np.vstack([main, strip])
    # stats go over the live canvas
    put_lines(frame, [title] + [f"{k}: {v:.3f}" for k, v in stats.items()], x=width - size[0] + 8)
    cv2.imshow(WINDOW, frame)
    return cv2.waitKey(1) & 0xFF


def run_slug() -> str:
    """The models' positive targets (a score: its model and goal), for file names."""
    def label(spec, target):
        if isinstance(target, int):
            return imagenet_classes()[target] if spec.model_id.is_imagenet() else str(target)
        return prompt_label(target)

    names = []
    for spec in MODELS:
        if spec.model_id.is_custom() and not spec.targets and not spec.directions:
            names += [spec.short_name(), spec.goal]
        names += [label(spec, t) for t, w in spec.targets if w > 0]
        names += [f"to-{label(spec, t)}" for _, t, w in spec.directions if w > 0]
    return re.sub(r"[^\w\-]+", "-", "_".join(names))[:60].strip("-") or "image"


def save(canvas: torch.Tensor, step: int) -> Path:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_FOLDER / f"{time.strftime('%Y%m%d_%H%M%S')}_{FIELD_TYPE.name.lower()}_{run_slug()}_{step}.png"
    cv2.imwrite(str(path), to_bgr(canvas))
    print(f"Saved -> {path}")
    return path


def pole_name(sign: int) -> str:
    return "plus" if sign > 0 else "minus"


def compose(start: torch.Tensor, poles: dict[int, dict]) -> np.ndarray:
    """Top: [- pole | start | + pole]. Bottom: difference maps (mid-grey = no change,
    amplified by DIFF_GAIN): [- vs start | + vs - | + vs start]."""
    def diff(a, b, label):
        tile = to_bgr((0.5 + DIFF_GAIN * (a - b)).clamp(0, 1))
        put_lines(tile, [f"{label}  x{DIFF_GAIN:g}"])
        return tile

    def pole_tile(sign):
        r = poles[sign]
        tile = to_bgr(r["canvas"])
        lines = [f"{'+' if sign > 0 else '-'} pole  {r['steps']} steps"
                 + ("" if r["reached_budget"] else "  (budget not reached)"),
                 f"sim->start: {r['similarity_to_start']:.3f}"]
        put_lines(tile, lines + [f"{k}: {v:.3f}" for k, v in r["stats"].items()
                                 if k not in ("loss", "tv", "sim->start")])
        return tile

    start_tile = to_bgr(start)
    put_lines(start_tile, ["start"])
    top, bottom = [start_tile], [np.full_like(start_tile, 128)]
    if -1 in poles:
        top.insert(0, pole_tile(-1))
        bottom.insert(0, diff(poles[-1]["canvas"], start, "(-) - start"))
        if 1 in poles:
            bottom[1] = diff(poles[1]["canvas"], poles[-1]["canvas"], "(+) - (-)")
    if 1 in poles:
        top.append(pole_tile(1))
        bottom.append(diff(poles[1]["canvas"], start, "(+) - start"))
    return np.vstack([np.hstack(top), np.hstack(bottom)])


def run_record() -> dict:
    """The settings of a run, for run.json"""
    return {
        "models": [{"model": str(spec.model_id), **{key: getattr(spec, key) for key in SPEC_KEYS},
                    **({"text_templates": spec.text_templates()} if spec.model_id.is_clip() else {})}
                   for spec in MODELS],
        "field_type": FIELD_TYPE.name, "start_image": START_IMAGE,
        "canvas_init": CANVAS_INIT if FIELD_TYPE not in (FieldType.DISTORTION, FieldType.REARRANGE) else "start",
        **({"distortion": {"resolution": DISTORTION_FIELD_RESOLUTION, "levels": DISTORTION_PYRAMID_LEVELS,
                           "lr_decay": DISTORTION_PYRAMID_LR_DECAY, "offset_mode": DISTORTION_OFFSET_MODE,
                           "max": DISTORTION_MAX, "edge_mode": DISTORTION_EDGE_MODE,
                           "fold_weight": DISTORTION_FOLD_WEIGHT if DISTORTION_PREVENT_FOLDS else None}}
           if FIELD_TYPE == FieldType.DISTORTION else {}),
        **({"patches": {"source": PATCH_SOURCE, "rotation": PATCH_ROTATION, "scale_range": PATCH_SCALE_RANGE,
                        "opaque": PATCH_OPAQUE, "transform_lr": PATCH_TRANSFORM_LR,
                        "images": PATCH_IMAGES, "image_resolution": PATCH_IMAGE_RESOLUTION}}
           if FIELD_TYPE == FieldType.PIXEL_PATCHES and PATCH_SOURCE == "images" else
           {"patches": {"source": PATCH_SOURCE, "count": PATCH_COUNT, "resolution": PATCH_RESOLUTION, "identical": PATCH_IDENTICAL,
                        "rotation": PATCH_ROTATION, "scale_range": PATCH_SCALE_RANGE,
                        "monochrome": PATCH_MONOCHROME, "opaque": PATCH_OPAQUE, "transform_lr": PATCH_TRANSFORM_LR,
                        "randomize_transforms": PATCH_RANDOMIZE_TRANSFORMS}}
           if FIELD_TYPE == FieldType.PIXEL_PATCHES else {}),
        **({"gaussians": {"count": GAUSSIAN_COUNT, "negative_count": GAUSSIAN_NEGATIVE_COUNT,
                          "max_size": GAUSSIAN_MAX_SIZE, "spread": GAUSSIAN_SPREAD,
                          "monochrome": GAUSSIAN_MONOCHROME, "contrast": GAUSSIAN_CONTRAST,
                          "fg_color": GAUSSIAN_FG_COLOR if GAUSSIAN_MONOCHROME and GAUSSIAN_CONTRAST > 0 else None,
                          "fg_optimize": GAUSSIAN_FG_OPTIMIZE if GAUSSIAN_MONOCHROME and GAUSSIAN_CONTRAST > 0 else None,
                          "rectilinear": GAUSSIAN_RECTILINEAR, "axis_aligned": GAUSSIAN_AXIS_ALIGNED,
                          **({"max_power": GAUSSIAN_MAX_POWER, "roundness_init": GAUSSIAN_ROUNDNESS_INIT}
                             if GAUSSIAN_RECTILINEAR else {})}}
           if FIELD_TYPE == FieldType.GAUSSIANS else {}),
        **({"shapes": {"mode": SHAPE_MODE,
                       **({"color_1": SHAPE_COLOR_1, "color_1_optimize": SHAPE_COLOR_1_OPTIMIZE}
                          if SHAPE_MODE != "color" else {}),
                       **({"color_2": SHAPE_COLOR_2, "color_2_optimize": SHAPE_COLOR_2_OPTIMIZE}
                          if SHAPE_MODE == "monochrome" else {}),
                       **({"opacity_budget": SHAPE_OPACITY_BUDGET, "opacity_weight": SHAPE_OPACITY_WEIGHT,
                           "harden_steps": SHAPE_HARDEN_STEPS, "harden_weight": SHAPE_HARDEN_WEIGHT}
                          if SHAPE_MODE == "masking" else {})}}
           if FIELD_TYPE == FieldType.SHAPES else {}),
        **({"relief": {"shapes": SHAPE_SRC, "color": RELIEF_COLOR, "color_optimize": RELIEF_COLOR_OPTIMIZE,
                       "depth": RELIEF_DEPTH, "smooth": RELIEF_SMOOTH, "ambient": LIGHT_AMBIENT,
                       "lights": {"count": LIGHT_COUNT, "azimuth": LIGHT_AZIMUTH, "elevation": LIGHT_ELEVATION,
                                  "random_count": LIGHT_RANDOM_COUNT, "random_elevation": LIGHT_RANDOM_ELEVATION}}}
           if FIELD_TYPE == FieldType.RELIEF else {}),
        **({"background": {"color": BACKGROUND_COLOR, "optimize": BACKGROUND_OPTIMIZE}}
           if FIELD_TYPE in (FieldType.PIXEL_PATCHES, FieldType.GAUSSIANS, FieldType.FUNCTION_STACK, FieldType.RELIEF,
                             FieldType.SHAPES, FieldType.REVOLUTION, FieldType.MESH_TEXTURE,
                             FieldType.GAUSSIANS_3D, *VOLUME_FIELDS) else {}),
        **({"background_image": {"path": BACKGROUND_IMAGE, "fit": BACKGROUND_IMAGE_FIT,
                                 "multiply": BACKGROUND_IMAGE_MULTIPLY}}
           if BACKGROUND_IMAGE and FIELD_TYPE in BACKGROUND_IMAGE_FIELDS else {}),
        **({"surface": SURFACE} if FIELD_TYPE == FieldType.SURFACE else {}),
        **({"vae": {"model": VAE_MODEL, "repo": VAE_MODELS[VAE_MODEL], "precision": VAE_PRECISION,
                    "decode_scale": VAE_DECODE_SCALE, "latent_bound": VAE_LATENT_BOUND,
                    "latent_weight": VAE_LATENT_WEIGHT}}
           if FIELD_TYPE == FieldType.VAE else {}),
        **({"convolutions": {"inplace": CONV_INPLACE, "layers": CONV_LAYERS, "channels": CONV_CHANNELS,
                             "kernel": CONV_KERNEL, "norm": CONV_NORM, "activation": CONV_ACTIVATION,
                             "weight_lr": CONV_WEIGHT_LR,
                             **({"residual_max": CONV_RESIDUAL_MAX} if CONV_INPLACE else
                                {}),
                             **({} if CONV_INPLACE else {"input_resolution": CONV_INPUT_RESOLUTION,
                                                         "fit_steps": CONV_FIT_STEPS})}}
           if FIELD_TYPE == FieldType.CONVOLUTIONS else {}),
        **({"rearrange": {"grid": REARRANGE_GRID, "permutation": REARRANGE_PERMUTATION, "init": REARRANGE_INIT,
                          "init_std": REARRANGE_INIT_STD, "temp_start": REARRANGE_TEMP_START,
                          "temp_end": REARRANGE_TEMP_END, "entropy_weight": REARRANGE_ENTROPY_WEIGHT,
                          "hard_after": REARRANGE_HARD_AFTER, "sinkhorn_iters": REARRANGE_SINKHORN_ITERS,
                          "lr": REARRANGE_LR}}
           if FIELD_TYPE == FieldType.REARRANGE else {}),
        **({"levelset": normalize_levelset(LEVELSET)} if FIELD_TYPE == FieldType.LEVELSET else {}),
        **({"function_stack": normalize_function_stack(FUNCTION_STACK)} if FIELD_TYPE == FieldType.FUNCTION_STACK else {}),
        **({"revolution": REVOLUTION} if FIELD_TYPE == FieldType.REVOLUTION else {}),
        **({"mesh_texture": MESH_TEXTURE} if FIELD_TYPE == FieldType.MESH_TEXTURE else {}),
        **({"gaussians_3d": GAUSSIANS_3D} if FIELD_TYPE == FieldType.GAUSSIANS_3D else {}),
        **({"voxels": VOXELS} if FIELD_TYPE == FieldType.VOXELS else {}),
        **({"assembly": ASSEMBLY} if FIELD_TYPE == FieldType.ASSEMBLY else {}),
        **({"solids_on_off": SOLIDS_ON_OFF} if FIELD_TYPE == FieldType.SOLIDS_ON_OFF else {}),
        **({"extrusions": EXTRUSIONS} if FIELD_TYPE == FieldType.EXTRUSIONS else {}),
        **({"pixel_contrast": PIXEL_CONTRAST} if FIELD_TYPE in (FieldType.PIXELS_COLOR, FieldType.PIXELS_GRAY) else {}),
        **({"pixel_coverage": PIXEL_COVERAGE, "pixel_coverage_weight": PIXEL_COVERAGE_WEIGHT}
           if FIELD_TYPE == FieldType.PIXELS_GRAY and PIXEL_COVERAGE is not None else {}),
        "canvas_resolution": CANVAS_RESOLUTION, "seed": SEED, "learning_rate": LEARNING_RATE,
        "image_sim_target": IMAGE_SIM_TARGET, "match_targets": MATCH_TARGETS, "image_sim_weight": IMAGE_SIM_WEIGHT,
        "image_gray_weight": IMAGE_GRAY_WEIGHT, "image_depth_weight": IMAGE_DEPTH_WEIGHT,
        "image_edge_weight": IMAGE_EDGE_WEIGHT, "edge_operator": EDGE_OPERATOR,
        "image_color_stats_weight": IMAGE_COLOR_STATS_WEIGHT, "image_edge_stats_weight": IMAGE_EDGE_STATS_WEIGHT,
        "edge_stats_scales": EDGE_STATS_SCALES if IMAGE_EDGE_STATS_WEIGHT else None,
        "image_lines_weight": IMAGE_LINES_WEIGHT,
        "lines_blur": LINES_BLUR if IMAGE_LINES_WEIGHT else None,
        "depth_model": DEPTH_MODEL if IMAGE_DEPTH_WEIGHT else None,
        "depth_augment_views": DEPTH_AUGMENT_VIEWS if IMAGE_DEPTH_WEIGHT else None,
        "depth_normalization": DEPTH_NORMALIZATION if IMAGE_DEPTH_WEIGHT else None,
        "stop_at_similarity": STOP_AT_SIMILARITY, "max_steps": EPOCHS,
        "sequential_targets": ({"steps_per_target": SEQUENTIAL_STEPS, "cycles": SEQUENTIAL_CYCLES}
                               if SEQUENTIAL_TARGETS else None),
        "tv_weight": TV_WEIGHT, "blur_strength": BLUR_STRENGTH,
        "median_weight": MEDIAN_WEIGHT, "median_kernel": MEDIAN_KERNEL if MEDIAN_WEIGHT else None,
        "pyramid_levels": PIXEL_PYRAMID_LEVELS, "pyramid_lr_decay": PIXEL_PYRAMID_LR_DECAY,
        "canvas_noise_std": CANVAS_NOISE_STD if CANVAS_INIT == "noise" else None,
        "symmetry_weight": SYMMETRY_WEIGHT, "symmetry_axis": SYMMETRY_AXIS,
    }


def run_json(record: dict) -> str:
    def encode(o):
        if isinstance(o, ImagePrompt):
            return {"image": o.path}
        if isinstance(o, VLMQuery):
            return {"question": o.question, "answer": o.answer}
        return str(o)
    return json.dumps(record, indent=1, default=encode)


def save_experiment(start: torch.Tensor, poles: dict[int, dict]) -> Path:
    """Folder with start / plus / minus PNGs, composite.png, run.json (config + scores, and the
    field's final parameters for SURFACE / LEVELSET / RELIEF) and, for fields with tensors (PIXELS_*:
    before / after the contrast sigmoid, LEVELSET: the layers, RELIEF: height and normals,
    REVOLUTION, MESH_TEXTURE: their maps), <pole>_field/ (save_tensors), there too what a field
    exports for good (REVOLUTION, MESH_TEXTURE: the Rhino file and the texture; GAUSSIANS_3D: the
    splats as a .ply and a .3dm point cloud with their isosurface; VOXELS: the volumes as dvolume
    .npz fields and the isosurface as a .3dm mesh)."""
    folder = OUTPUT_FOLDER / f"{time.strftime('%Y%m%d_%H%M%S')}_{FIELD_TYPE.name.lower()}_{run_slug()}"
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "start.png"), to_bgr(start))
    for sign, r in poles.items():
        cv2.imwrite(str(folder / f"{pole_name(sign)}.png"), to_bgr(r["canvas"]))
        if r.get("tensors"):
            save_tensors(folder / f"{pole_name(sign)}_field", r["tensors"])
        if r.get("export"):
            r["export"](folder / f"{pole_name(sign)}_field")
    cv2.imwrite(str(folder / "composite.png"), compose(start, poles))
    record = {**run_record(),
              "poles": {pole_name(sign): {k: v for k, v in r.items() if k not in ("canvas", "tensors", "export")}
                        for sign, r in poles.items()}}
    (folder / "run.json").write_text(run_json(record))
    print(f"Saved -> {folder}")
    return folder


def start_log(device, files: dict[str, str] | None = None) -> RunLog:
    """A new run's log folder (run_log.py) in OUTPUT_FOLDER/logs: run.json, `files`
    ({name: text}), then the gpu's state while the run goes on."""
    folder = OUTPUT_FOLDER / "logs" / f"{time.strftime('%Y%m%d_%H%M%S')}_{FIELD_TYPE.name.lower()}_{run_slug()}"
    return RunLog(folder, {"run.json": run_json(run_record()), **(files or {})}, device)


# ── Main ───────────────────────────────────────────────────────────────────────

StepCallback = Callable[[int, int, torch.Tensor, torch.Tensor, dict[str, float]], bool]


def cv2_preview(original: np.ndarray | None, objective: 'Objective') -> StepCallback:
    """on_step for run_pole: the OpenCV preview window. Esc stops, S saves."""
    def on_step(sign, step, canvas, views, stats):
        if step % PREVIEW_EVERY:
            return True
        pole = "+" if sign > 0 else "-"
        key = show_preview(canvas, views, f"{FIELD_TYPE.name} [{pole}]  step {step}/{pole_steps(objective)}", stats, original)
        if key == 27:
            print("Stopped.")
            return False
        if key in (ord("s"), ord("S")):
            save(canvas, step)
        return True
    return on_step


def sequential(objective: 'Objective') -> bool:
    return SEQUENTIAL_TARGETS and objective.target_count() > 0


def pole_steps(objective: 'Objective') -> int:
    """Steps per pole: EPOCHS, or every target's turn in sequential mode."""
    if sequential(objective):
        return objective.target_count() * max(1, SEQUENTIAL_CYCLES) * max(1, SEQUENTIAL_STEPS)
    return EPOCHS


def active_target(objective: 'Objective', step: int) -> int | None:
    """The target scored at `step` in sequential mode (in target / axes order), else None."""
    if not sequential(objective):
        return None
    n = objective.target_count()
    return min(step // max(1, SEQUENTIAL_STEPS), n * max(1, SEQUENTIAL_CYCLES) - 1) % n


def match_renders(match: ImageMatch, renders: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """ImageMatch averaged over (B, 3, H, W) images (RELIEF: one per fixed light). Run
    last-to-first, so the preview maps it keeps are those of the displayed image 0."""
    loss, sums = 0.0, {}
    for i in reversed(range(len(renders))):
        image_loss, stats = match(renders[i:i + 1])
        loss = loss + image_loss
        for key, value in stats.items():
            sums[key] = sums.get(key, 0.0) + value
    return loss / len(renders), {key: value / len(renders) for key, value in sums.items()}


def run_pole(sign: int, objective: 'Objective', device,
             on_step: StepCallback) -> tuple[torch.Tensor, dict] | None:
    """One optimization from a fresh field, same seed for every pole. Returns the start
    canvas and the result at the last scored step, or None when on_step(sign, step,
    canvas, views, stats) returns False. on_step sees every scored canvas, the final
    one included; objective.last_axes / last_view_axes hold its axis values."""
    torch.manual_seed(SEED)
    field = make_field(device)
    # REARRANGE: the start image, in its own arrangement, whatever the blend starts as.
    start = (field.reference() if hasattr(field, "reference") else field.render()).detach()
    # A field with its own views (SURFACE: its cameras, RELIEF: its lights) replaces the plain augmentations.
    sample_views = field.views if hasattr(field, "views") else None
    objective.set_start(start, sample_views)
    objective.sign = float(sign)
    objective.field = field  # the field being optimized, for previews
    optimizer = torch.optim.Adam(field.params(), lr=LEARNING_RATE)
    pole = "+" if sign > 0 else "-"
    epochs = pole_steps(objective)
    if hasattr(field, "total_steps"):
        field.total_steps = epochs  # REARRANGE: its anneal spans the run
    # REARRANGE, hardening SHAPES: no early stop -- the result would be a soft blend, not yet a
    # rearrangement or hard shapes.
    budget = (None if sequential(objective) or FIELD_TYPE == FieldType.REARRANGE or getattr(field, "harden_steps", 0)
              else STOP_AT_SIMILARITY)

    result = {}
    for step in range(epochs + 1):
        objective.active = active_target(objective, step)
        canvas = field.render()
        views = sample_views() if sample_views else augment(canvas)
        clip_loss, stats = objective(views)
        tv = total_variation(field.regularized())
        loss = clip_loss + TV_WEIGHT * tv
        if hasattr(field, "penalty"):
            field_loss, field_stats = field.penalty(canvas)
            loss = loss + field_loss
            stats = {**field_stats, **stats}
        if objective.match is not None:
            renders = field.fixed_renders() if hasattr(field, "fixed_renders") else canvas
            match_loss, match_stats = match_renders(objective.match, renders)
            loss = loss + match_loss
            stats = {**match_stats, **stats}
        if SYMMETRY_WEIGHT:
            sym = symmetry_error(field.regularized() if FIELD_TYPE == FieldType.RELIEF else canvas)
            loss = loss + SYMMETRY_WEIGHT * sym
            stats = {"sym": sym.item(), **stats}
        sim = objective.similarity_to_start(canvas)
        stats = {"loss": loss.item(), "tv": tv.item(), "sim->start": sim, **stats}
        # `step` updates have been applied to this canvas.
        result = {"canvas": canvas.detach(), "steps": step, "similarity_to_start": sim,
                  "reached_budget": budget is not None and sim <= budget,
                  "stats": stats}
        if not on_step(sign, step, canvas, views, stats):
            return None
        if result["reached_budget"]:
            print(f"[{pole}] edit budget reached after {step} steps: sim->start {sim:.3f}")
            break
        if step == epochs:
            if budget is not None:
                print(f"[{pole}] budget not reached in {epochs} steps: sim->start {sim:.3f}")
            break

        optimizer.zero_grad(set_to_none=True)
        keep_grad = CANVAS_GRAD and canvas.requires_grad
        if keep_grad:
            canvas.retain_grad()
        loss.backward()
        # None when the loss doesn't reach the canvas (SURFACE: its views are rendered apart)
        objective.last_canvas_grad = canvas.grad.detach() if keep_grad and canvas.grad is not None else None
        if hasattr(field, "before_step"):
            field.before_step(optimizer)
        optimizer.step()
        if BLUR_STRENGTH > 0:
            for p in field.blurrable():
                gaussian_blur_(p, BLUR_SIGMA, BLUR_STRENGTH)
        if MEDIAN_WEIGHT > 0:
            for p in field.blurrable():
                median_blur_(p, MEDIAN_KERNEL, MEDIAN_WEIGHT)
        if hasattr(field, "after_step"):
            field.after_step(step)  # VOXELS: its own blur, between steps

        if step % 50 == 0:
            print(f"[{pole}] step {step:5d}  " + "  ".join(f"{k}={v:.3f}" for k, v in stats.items()), flush=True)
    # The field is still at the result's step: its tensors / parameters, for save_experiment.
    if hasattr(field, "tensors"):
        result["tensors"] = field.tensors()
    if hasattr(field, "values"):
        result["field_values"] = field.values()
    if hasattr(field, "export"):
        result["export"] = field.export  # save_experiment's: nothing changes this pole's field any more
    return start, result


def main():
    cap_gpu_memory()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    central = ModelCentral(device, text_on_cpu=True)
    log = start_log(device)
    try:
        objective = make_objective(central, device, log)

        gray = FIELD_TYPE == FieldType.PIXELS_GRAY
        original = (to_bgr(load_image(START_IMAGE, CANVAS_RESOLUTION, gray).expand(-1, 3, -1, -1))
                    if START_IMAGE is not None else None)

        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        start, poles, status = None, {}, "done"
        for sign in ((1, -1) if BIDIRECTIONAL else (1,)):
            out = run_pole(sign, objective, device, log.wrap(cv2_preview(original, objective)))
            if out is None:
                status = "stopped"
                break
            start, poles[sign] = out
        folder = save_experiment(start, poles) if poles else None
        if folder:
            log.event(f"saved -> {folder}")
    except BaseException:
        log.close("error:\n" + traceback.format_exc())
        raise
    log.close(status)

    if folder:
        composite = cv2.imread(str(folder / "composite.png"))
        scale = min(1.0, 1800 / composite.shape[1])
        cv2.imshow(WINDOW, cv2.resize(composite, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    print("Done -- any key in the preview window closes it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
