"""Model central: the web app of custom_models (web/app.html), three tabs:

  Embedding Map  (web/embedding_map.js, most of this file)
  Training       (web/training.js, training_engine.py, near the end of this file)
  Dataset        (web/dataset.js, dataset_tools.py, audio_tools.py, at the end):
                 Converter: make a new dataset from one, every image converted
                 (grayscale, Canny edges, Depth Anything depth, depth edges)
                 Audio: cut a sound file into fragments, one spectrogram image
                 + the fragment's sound each. On the map, an image with a sound
                 file of the same name next to it plays it when clicked.

Embedding map: a web page that places the images of a dataset on a 2D map
according to the vectors a model gives them, and groups them into clusters.

    images  ->  model  ->  one vector per image  ->  projection  ->  (x, y)
                                                 ->  clustering  ->  group

Models (picked in the page) are every folder under ROOT_MODEL_FOLDER whose
config.json is an autoencoder or a classifier, plus pretrained DINOv2 and CLIP
image encoders. See embedders.py.

Datasets are every folder under ROOT_DATASETS_FOLDER that directly contains
at least one image, and the same under each of ADDITIONAL_DATASET_FOLDERS
(only read: new datasets are always written to ROOT_DATASETS_FOLDER). Those
are listed as "@<folder name>/...".

Projections squeeze the vectors (128, 768, ... numbers) down to 2:

  PCA    the 2 directions along which the vectors spread the most. Linear, fast,
         keeps large distances, but crowds most images into the middle.
  t-SNE  keeps neighbours together: images close in the vector space end up
         close on the map. Distances between far-apart groups mean little.
         perplexity ~ how many neighbours each image cares about.
  UMAP   like t-SNE, faster, keeps somewhat more of the global layout.
         neighbours = how many neighbours each image cares about.

  axes   two chosen components of the vectors as x and y, as they are (after
         the "vectors" preparation). For a classifier, e.g. the scores of two
         classes: images of the first class right, of the second up.

  score (one component)  one number per image: a chosen component of the
         vectors (or a similarity, below), laid out as
         histogram  along x, the images stacked in columns (bins), lowest
             at the bottom.
         central  the highest (or lowest) scoring image in the middle, the
             others around it from the most to the least: a quasi-square
             grid (rectangular) or concentric rings (radial), whose radii
             can follow how far each ring's scores are from the middle one's.
         The histogram is the view of a RankNet, whose vectors are one
         number, the score: left = worst, right = best. Vectors of one number
         are shown like this when another projection is chosen (t-SNE, UMAP,
         PCA have nothing to work with in 1D). Pretrained encoders can instead
         order the images by their cosine similarity to a prompt (CLIP) or
         to an image uploaded from the page (CLIP, DINOv2): most similar right.

  clusters  not a projection of the vectors: a grid, int(sqrt(N)) columns,
         one section of rows per cluster (computed on the vectors), each
         cluster's images from left to right from the most representative
         (closest to the cluster's center) to the least.

  spring forces  computed live in the page (web/spring.js): springs between
         images, their rest length from how similar the images are. The
         server only sends the vectors (at most SPRING_DIMENSIONS numbers
         each, PCA when longer) and a PCA map to start from.

Clusterings group the images, on the vectors or on the 2D map:

  k-means   k groups, each image goes to the nearest of k centers
  spectral  k groups that are connected chains of neighbours, can find
            groups of any shape
  DBSCAN    groups are regions dense with images, the number of groups is
            found by itself. eps = how close counts as neighbour, images in no
            dense region are noise (-1, gray).

The browser only gets coordinates and cluster numbers; the images glide to
their new place whenever something changes.

The page can take a selected image off the map, and add images from its own
computer to it (/api/set/...). Both last until the server stops and leave the
dataset's folder alone: added images are copies in ADDED_FOLDER.

Autoencoders can also interpolate: pick two images A and B on the map, the
codes on the straight line between their codes are decoded into images,
encoded again and placed on the map among their nearest images, joined by a
line from A to B (/api/interpolate).

Export PDF saves the map as it is shown (zoom, pan, highlighted cluster,
interpolation...) as a PDF, with the legend (map_pdf.py, /api/export/pdf).

The Training tab of the page trains a new autoencoder or classifier (or keeps
training a saved one) into ROOT_MODEL_FOLDER, in a background thread, and
shows its progress, loss curves and a preview after every epoch; it can be
stopped at any moment, with or without saving (training_engine.py,
/api/train/...). A classifier trains on a folder with one subfolder of
images per class.

The Dataset tab also collects rankings: two images of a dataset at a time,
which one is better (for a question, e.g. "which one would you hang on your
wall?"). They are saved as ranking_<name>.json in the dataset's folder
(ranknet/rankings.py), as many as wanted (people, questions), and a RankNet
trains on one of them in the Training tab: one score per image. On the map a
RankNet's vectors are that one number, shown as a histogram, best on the right.

The model list also has ready-made aesthetic scores (LAION aesthetic, PickScore,
HPSv2: aesthetic/aesthetic_scorers.py), one number per image like a RankNet;
PickScore and HPSv2 score against a prompt typed in the page.

The Dataset tab's Sampler makes a small, well spread dataset out of a large
one (dataset_sampler.py): cluster by a model's vectors, take a few images of
every cluster. The Ranking page can then ask pairs within a cluster.

Run it and open http://127.0.0.1:5002
"""

import base64
import io
import itertools
import json
import math
import os
import random
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import torch
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageOps

import env_config
import dataset_sampler
import map_pdf
from autoencoder.autoencoder_model import AUTOENCODERS, AutoencoderType, folder_model_type, read_config
from classifier.classifier_model import ARCHS, CLIPArch, CLIPPretrained, ClassifierBackbone
from embedders import find_models, load_embedder, query_kinds
from ranknet import rankings
from model_types import DINOV2Arch, ModelID
from training_engine import TrainingRun
import audio_tools
from audio_tools import AUDIO_EXTENSIONS, AudioClip, AudioRun
from dataset_tools import (CONVERSIONS, DEPTH_MODELS, FORMATS, ConversionRun,
                           convert as convert_images, data_url as dataset_data_url,
                           load as dataset_load)

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_MODEL_FOLDER = str(env_config.ROOT_MODEL_FOLDER)        # set in env_config.json (see env_config.py)
ROOT_DATASETS_FOLDER = str(env_config.ROOT_DATASETS_FOLDER)
# "@<folder name>" -> an additional (read-only) datasets folder
EXTRA_DATASET_ROOTS = {}
for _root in env_config.ADDITIONAL_DATASET_FOLDERS:
    _prefix, _n = f"@{_root.name}", 2
    while _prefix in EXTRA_DATASET_ROOTS:
        _prefix, _n = f"@{_root.name}{_n}", _n + 1
    EXTRA_DATASET_ROOTS[_prefix] = _root
DEFAULT_MODEL = "pretrained/dinov2_vits14"

MAX_IMAGES = 1000           # default of the page's "max images": bigger datasets use a random sample
LOAD_SIZE = 256             # images are decoded at about this size (enough for every model)
THUMBNAIL_SIZE = 96         # images on the map
PREVIEW_SIZE = 320          # image in the side panel
RANKING_IMAGE_SIZE = 1024   # largest image the ranking page asks for
BATCH_SIZE = 32
PORT = 5002

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROJECTIONS = {"tsne": "t-SNE", "umap": "UMAP", "pca": "PCA", "axes": "selected axes",
               "histogram": "score (one component)", "clusters": "clusters (grid)",
               "spring": "spring forces (live)"}
GRID_LAYOUTS = ("clusters", "histogram")     # images in touching cells (the result has "cell")
# spring forces: the page compares every pair it looks at, so the vectors it
# gets are kept short (PCA keeps most of their spread)
SPRING_DIMENSIONS = 64
PREPS = {"l2": "L2-normalized / cosine", "standardize": "standardized", "none": "as they are"}
CLUSTER_GAP = 0.4           # clusters grid: empty space between clusters, in rows

# Classifiers: every class also gets a dummy "image" on the map, its anchor:
# the logits of an image the classifier is completely sure of, ANCHOR_LOGIT for
# that class and 0 for the others (softmax: practically 1 and 0).
ANCHOR_LOGIT = 20.0

# Autoencoders: an interpolated image is placed at the weighted average position
# of this many of its nearest dataset images
INTERPOLATION_NEIGHBORS = 5
CLUSTERINGS = {"kmeans": "k-means", "spectral": "spectral", "dbscan": "DBSCAN", "none": "none"}

app = Flask(__name__)
app.json.sort_keys = False      # keep the order of dictionaries (e.g. a conversion's parameters)

# One model in memory at a time. model_lock guards loading and using it, job_lock
# makes the map computations run one after another (flask answers requests in
# parallel threads).
model_lock = threading.Lock()
job_lock = threading.Lock()
loaded = {"id": None, "embedder": None, "base": None}   # base: the model a prompt variant is of

models = {}                 # id -> {"id", "label", "kind", "path"}
datasets = {}               # id -> list of all its image paths
dataset_info = {}           # id -> {"group", "label", "folder", "recursive"}, see scan()
sound_files = {}            # image path without extension -> its sound file (same name, next to it)
class_datasets = {}         # id -> {"classes", "count"}: folders with one subfolder of images per class
sample_orders = {}          # id -> the image indices in a fixed random order
embeddings = {}             # (model id, dataset id) -> {image index: vector}
projections = {}            # (model id, dataset id, image indices, prep, method, parameter) -> array [N, 2]
                            # parameter: perplexity, neighbours, (x, y) components, None
thumbnails = {}             # (dataset id, image index) -> JPEG bytes
jobs = {}                   # job id -> progress and result
next_job = itertools.count(1)

# What the page changed in a dataset's map, until the server stops. The folders
# aren't touched: an image taken off the map stays in its folder, an image
# added from the page's computer is a copy in ADDED_FOLDER, only on the map.
# Added images have ids from ADDED_BASE on (dataset images: 0..N-1, class
# anchors < 0), the same in every dataset.
ADDED_BASE = 1_000_000_000
ADDED_FOLDER = Path(tempfile.gettempdir()) / "model_central_added"
added_files = []            # id - ADDED_BASE -> (copy in ADDED_FOLDER, original file name)
added = {}                  # dataset id -> ids of the images added to its map
removed = {}                # dataset id -> ids taken off its map (its own images or added ones)

# The histogram can order the images by their similarity to a prompt or to an
# image from the page's computer (kept here, not as a file), until the server stops
QUERY_KINDS = ("component", "prompt", "image")
query_images = {}           # id -> PIL image
query_vectors = {}          # (model id, "prompt" or "image", text or image id) -> vector [D]


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

def scan_models() -> None:
    found = {m["id"]: m for m in find_models(ROOT_MODEL_FOLDER)}
    models.clear()
    models.update(found)


def scan() -> None:
    """Datasets are

      "a/b"     the images directly in the folder a/b
      "a/**"    every image anywhere under a, for every folder with 2 or more
                image folders under it (a parent folder picked as a whole)

    The page groups them by folder: "all of a/" and the image folders inside a.
    Folders under an additional datasets folder are "@<its name>/a/b"; they are
    not part of "everything" (the main datasets folder)."""
    scan_models()

    datasets.clear()
    class_datasets.clear()
    dataset_info.clear()
    sound_files.clear()
    direct = {}
    for prefix, root in [("", Path(ROOT_DATASETS_FOLDER)), *EXTRA_DATASET_ROOTS.items()]:
        if not root.is_dir():
            continue
        for folder, subfolders, files in os.walk(root):
            subfolders[:] = sorted(s for s in subfolders if not s.startswith((".", "__")))
            paths = sorted(Path(folder) / f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS)
            if paths:
                rel = Path(folder).relative_to(root).as_posix()
                if prefix:
                    rel = prefix if rel == "." else f"{prefix}/{rel}"
                direct[rel] = paths
            # a sound file next to an image, with the same name, is that image's sound
            for f in files:
                if Path(f).suffix.lower() in AUDIO_EXTENSIONS:
                    sound_files[str(Path(folder) / Path(f).stem)] = Path(folder) / f

    # every folder -> the image folders under it (itself included)
    below = {}
    for dataset_id in direct:
        p = Path(dataset_id)
        for folder in [p, *p.parents]:
            if folder.as_posix() == "." and is_extra(dataset_id):
                break
            below.setdefault(folder.as_posix(), []).append(dataset_id)
    groups = {folder: ids for folder, ids in below.items() if len(ids) >= 2}

    for folder, ids in groups.items():
        dataset_id = "**" if folder == "." else f"{folder}/**"
        datasets[dataset_id] = sorted(p for d in ids for p in direct[d])
        dataset_info[dataset_id] = {"group": folder, "folder": folder, "recursive": True,
                                    "label": "everything" if folder == "." else f"all of {folder}/"}
    for dataset_id, paths in direct.items():
        datasets[dataset_id] = paths
        if dataset_id in groups:        # it has image folders inside: its own images are one more option
            info = {"group": dataset_id, "label": f"only the images directly in {Path(dataset_id).name}"}
        else:
            info = {"group": Path(dataset_id).parent.as_posix(), "label": Path(dataset_id).name}
        if dataset_id == ".":
            info["label"] = "only the images directly in the datasets folder"
        dataset_info[dataset_id] = {**info, "folder": dataset_id, "recursive": False}
    for dataset_id, paths in datasets.items():
        sample_orders[dataset_id] = random.Random(0).sample(range(len(paths)), len(paths))

    # A folder with 2 or more subfolders of images can train a classifier: one
    # class per subfolder (as torchvision's ImageFolder reads it)
    children = {}
    for dataset_id, paths in direct.items():
        if dataset_id != "." and dataset_id not in EXTRA_DATASET_ROOTS:
            parent = Path(dataset_id).parent.as_posix()
            children.setdefault(parent, []).append((Path(dataset_id).name, len(paths)))
    for parent, classes in sorted(children.items()):
        if len(classes) >= 2:
            class_datasets[parent] = {"classes": [name for name, _ in classes],
                                      "count": sum(n for _, n in classes)}
    print(f"{len(models)} models, {len(datasets)} datasets, {len(class_datasets)} with classes")


def dataset_list() -> list[dict]:
    """For the page's dataset dropdowns: grouped by folder, in each group "all
    of ..." first, then the image folders by name."""
    entries = [{"id": d, "count": len(datasets[d]), **dataset_info[d]} for d in datasets]
    return sorted(entries, key=lambda e: (e["group"] != ".", e["group"].lower(), not e["recursive"],
                                          e["folder"] != e["group"], e["label"].lower()))


def sound_of(path: Path) -> Path | None:
    return sound_files.get(str(path.with_suffix("")))


def is_extra(folder: str) -> bool:
    """Is this folder ("a/b", "@name/a/b") under an additional datasets folder?"""
    return folder.split("/")[0] in EXTRA_DATASET_ROOTS


def folder_path(folder: str) -> Path:
    """"a/b" -> ROOT_DATASETS_FOLDER/a/b, "@name/a/b" -> that additional folder/a/b"""
    if is_extra(folder):
        prefix, _, rest = folder.partition("/")
        return EXTRA_DATASET_ROOTS[prefix] / rest
    return Path(ROOT_DATASETS_FOLDER) / folder


def dataset_folder(dataset_id: str) -> Path:
    return folder_path(dataset_info[dataset_id]["folder"])


def image_name(dataset_id: str, path: Path) -> str:
    """The image's path inside its dataset's folder (subfolder/name.jpg for "all of ...")"""
    return path.relative_to(dataset_folder(dataset_id)).as_posix()


# Grayscale: the page asks for a dataset as "<id>~gray". Its images are
# converted to gray as they are opened (for the model and the thumbnails), and
# everything computed from them is kept apart from the color ones (the caches
# are keyed by this id). The images themselves, their sample and the images
# taken off or added to the map are the dataset's (base_dataset).
GRAY_SUFFIX = "~gray"


def base_dataset(dataset_id: str) -> str:
    return dataset_id.removesuffix(GRAY_SUFFIX)


def is_gray(dataset_id: str) -> bool:
    return dataset_id.endswith(GRAY_SUFFIX)


def sample(dataset_id: str, max_images: int) -> list[int]:
    """Indices of the images used: all of them, or max_images picked at random.
    Always the same random order, so a bigger sample contains the smaller one
    (and their embeddings are reused). Then the ones taken off the map go,
    the ones added to it come after."""
    dataset_id = base_dataset(dataset_id)
    gone = removed.get(dataset_id, set())
    picked = sorted(i for i in sample_orders[dataset_id][:max(1, max_images)] if i not in gone)
    return picked + [i for i in added.get(dataset_id, []) if i not in gone]


def map_name(dataset_id: str, i: int) -> str:
    if i >= ADDED_BASE:
        return f"added: {added_files[i - ADDED_BASE][1]}"
    dataset_id = base_dataset(dataset_id)
    return image_name(dataset_id, datasets[dataset_id][i])


def open_image(path: Path, gray: bool = False) -> Image.Image:
    try:
        image = Image.open(path)
        image.draft("RGB", (LOAD_SIZE, LOAD_SIZE))     # JPEGs: decode at reduced size, much faster
        image = image.convert("L").convert("RGB") if gray else image.convert("RGB")
        image.thumbnail((LOAD_SIZE * 2, LOAD_SIZE * 2))
        return image
    except Exception as e:
        print(f"can't read {path}: {e}")
        return Image.new("RGB", (LOAD_SIZE, LOAD_SIZE), "gray")


def open_dataset_image(dataset_id: str, i: int) -> Image.Image:
    """Image i of a dataset's map, in gray for a "~gray" dataset id"""
    return open_image(dataset_image(dataset_id, i), is_gray(dataset_id))


def jpeg(image: Image.Image, size: int) -> bytes:
    image = image.copy()
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Embedding, projection, clustering
# ---------------------------------------------------------------------------

class Cancelled(Exception):
    pass


def load_model(model_id: str) -> None:
    """Called with model_lock held."""
    if loaded["id"] == model_id:
        return
    entry = models[model_id]
    base = entry.get("variant_of", model_id)
    if loaded["base"] == base and loaded["embedder"] is not None and hasattr(loaded["embedder"], "set_prompt"):
        loaded["embedder"].set_prompt(entry.get("prompt_text"))
        loaded["id"] = model_id
        return
    loaded["id"], loaded["embedder"], loaded["base"] = None, None, None   # free the previous one first
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    print(f"loading {model_id}")
    loaded["embedder"] = load_embedder(entry, DEVICE)
    loaded["id"], loaded["base"] = model_id, base
    print(loaded["embedder"].description)


def get_embeddings(model_id: str, dataset_id: str, indices: list[int], job: dict) -> np.ndarray:
    """Vectors [len(indices), D] of these images, computing only the ones not
    done before."""
    done = embeddings.setdefault((model_id, dataset_id), {})
    missing = [i for i in indices if i not in done]
    if missing:
        with model_lock:
            job["stage"] = f"loading {models[model_id]['label']}"
            load_model(model_id)
            for start in range(0, len(missing), BATCH_SIZE):
                if job["cancelled"]:
                    raise Cancelled()
                job["stage"] = f"embedding {start}/{len(missing)}"
                job["progress"] = start / len(missing)
                batch = missing[start:start + BATCH_SIZE]
                images = []
                for i in batch:
                    image = open_dataset_image(dataset_id, i)
                    # the images are open anyway: make the map thumbnails now
                    thumbnails.setdefault((dataset_id, i), jpeg(image, THUMBNAIL_SIZE))
                    images.append(image)
                vectors = loaded["embedder"].embed(images).flatten(1).numpy().astype(np.float32)
                done.update(zip(batch, vectors))
    return np.stack([done[i] for i in indices])


def query_vector(model_id: str, kind: str, query: str, job: dict, gray: bool = False) -> np.ndarray:
    """The vector of a prompt or of an uploaded image (gray: in gray, like the
    images it is compared with), as the model embeds it"""
    key = (model_id, kind, query, gray and kind == "image")
    if key not in query_vectors:
        with model_lock:
            job["stage"] = f"loading {models[model_id]['label']}"
            load_model(model_id)
            job["stage"] = f"embedding the {kind}"
            embedder = loaded["embedder"]
            if kind == "prompt":
                vector = embedder.embed_text([query])
            else:
                image = query_images[query]
                vector = embedder.embed([image.convert("L").convert("RGB") if gray else image])
            query_vectors[key] = vector[0].numpy().astype(np.float32)
    return query_vectors[key]


def cosine(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """cosine similarity of every row of x with v"""
    return (x @ v) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(v), 1e-8)


def class_names_of(model_id: str) -> list[str] | None:
    """A classifier's class names (from its config.json), None for other models"""
    model = models[model_id]
    if model["kind"] != "classifier":
        return None
    return read_config(model["path"])["class_names"]


def model_with_prompt(model_id: str, prompt: str) -> str:
    """A scorer that compares images with a prompt: the id of its variant for
    this prompt (a model of its own: its own embeddings, kept for each prompt)"""
    model = models[model_id]
    if not model.get("prompt"):
        return model_id
    text = (prompt or "").strip() or "an image"
    variant = f"{model_id}~{text}"
    models.setdefault(variant, {**model, "id": variant, "label": f'{model["label"]}: "{text}"',
                                "variant_of": model_id, "prompt_text": text})
    return variant


def components_of(model_id: str) -> list[str] | None:
    """Names of the components of a model's vectors, for picking the selected
    axes: a classifier's classes, an autoencoder's latent dimensions. None for
    the pretrained ones (known after their first map, from its "dimensions")."""
    model = models[model_id]
    if model["kind"] == "pretrained":
        return None
    if model["kind"] in ("ranknet", "aesthetic"):
        return ["score"]
    try:
        config = read_config(model["path"])
    except (OSError, ValueError):
        return None
    if model["kind"] == "classifier":
        return config["class_names"]
    return [str(d) for d in range(config["latent_dim"])]


def default_prep(model_id: str) -> str:
    """How a model's vectors are best compared, used when the page asks for "default".

      l2           CLIP, DINOv2: they are trained by comparing directions
                   (cosine similarity), the length means little
      none         classifiers: the logits as they are, how big they are is how
                   sure the classifier is. RankNets, aesthetic scores: the
                   score (l2 of one number would leave only its sign). VAE, VAE-GAN: the KL loss already
                   keeps every dimension near mean 0, spread 1. Sparse: 0 means
                   "unit off", keep it so.
      standardize  the other autoencoders: the decoder reads the raw vector, so
                   its length matters (no l2), but a few dimensions can have a
                   much bigger range than the rest and would dominate distances
    """
    model = models[model_id]
    if model["kind"] == "pretrained":
        return "l2"
    if model["kind"] in ("classifier", "ranknet", "aesthetic"):
        return "none"
    try:
        autoencoder_type = read_config(model["path"])["autoencoder_type"]
    except (OSError, ValueError, KeyError):
        return "standardize"
    return "none" if autoencoder_type in ("vae", "vae_gan", "sparse") else "standardize"


def prepare(x: np.ndarray, prep: str) -> np.ndarray:
    """l2: every vector scaled to length 1, so only its direction counts
           (distances become cosine distances, the usual for CLIP / DINOv2)
       standardize: every dimension scaled to mean 0, spread 1 (z-score over
           the images in the map), so all dimensions count equally
       none: as the model gives them"""
    if prep == "l2":
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    if prep == "standardize":
        return (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-8)
    return x


def to_unit_square(xy: np.ndarray, fit: np.ndarray | None = None) -> np.ndarray:
    """Center in 0..1 x 0..1, keeping the proportions. fit: which points the
    frame is made for (default all), the others are clipped to its edges."""
    frame = xy if fit is None or not fit.any() else xy[fit]
    center = (frame.min(axis=0) + frame.max(axis=0)) / 2
    extent = np.abs(frame - center).max() or 1.0
    return np.clip((xy - center) / (2 * extent) + 0.5, 0, 1)


def project(x: np.ndarray, method: str, parameter, anchor: np.ndarray | None = None) -> np.ndarray:
    """parameter: t-SNE perplexity, UMAP neighbours, (x component, y component)
    for the selected axes. anchor: which rows are classifier class anchors"""
    n = len(x)
    if method == "axes":
        axis_x, axis_y = parameter
        if not (0 <= axis_x < x.shape[1] and 0 <= axis_y < x.shape[1]):
            raise ValueError(f"axes {axis_x}, {axis_y}: the vectors have components 0 to {x.shape[1] - 1}")
        xy = x[:, [axis_x, axis_y]].astype(float)
        xy[:, 1] *= -1          # the map's y goes down, bigger values go up
        # the class anchors (ANCHOR_LOGIT, far beyond the images' scores) would
        # squeeze the images into a corner: the map is framed on the images,
        # the anchors stay on its edges
        return to_unit_square(xy, None if anchor is None else ~anchor)
    if n < 4:       # nothing to learn from: spread them on a line
        return to_unit_square(np.stack([np.arange(n), np.zeros(n)], axis=1).astype(float))
    from sklearn.decomposition import PCA
    if method == "pca":
        xy = PCA(2).fit_transform(x)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        if x.shape[1] > 50:     # common practice: PCA to 50 first, faster and less noisy
            x = PCA(min(50, n)).fit_transform(x)
        xy = TSNE(2, perplexity=min(parameter, (n - 1) / 3), init="pca", learning_rate="auto",
                  random_state=0).fit_transform(x)
    elif method == "umap":
        import umap         # slow to import (compiles with numba), only when needed
        # a fixed random_state gives the same map every time, but runs on one
        # core (n_jobs=1 says so, instead of a warning)
        xy = umap.UMAP(n_neighbors=int(min(parameter, n - 1)), min_dist=0.1,
                       random_state=0, n_jobs=1).fit_transform(x)
    else:
        raise ValueError(f"unknown projection {method}")
    return to_unit_square(xy)


def spring_vectors(x: np.ndarray) -> dict:
    """The vectors the page's spring forces compare: at most SPRING_DIMENSIONS
    numbers each, as little-endian float32 in base64 (a fraction of the JSON)"""
    if x.shape[1] > SPRING_DIMENSIONS and len(x) > SPRING_DIMENSIONS:
        from sklearn.decomposition import PCA
        x = PCA(SPRING_DIMENSIONS).fit_transform(x)
    data = np.ascontiguousarray(x, dtype="<f4").tobytes()
    return {"shape": list(x.shape), "data": base64.b64encode(data).decode("ascii")}


def cluster(x: np.ndarray, method: str, k: int, eps: float, min_samples: int) -> np.ndarray:
    """-> cluster number of every point, -1 = noise. Clusters are numbered by
    size, 0 = the biggest."""
    n = len(x)
    if method == "none" or n < 2:
        return np.zeros(n, dtype=int)
    k = max(1, min(k, n))
    if method == "kmeans":
        from sklearn.cluster import KMeans
        labels = KMeans(k, n_init="auto", random_state=0).fit_predict(x)
    elif method == "spectral":
        from sklearn.cluster import SpectralClustering
        labels = SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=min(10, n - 1),
                                    assign_labels="cluster_qr", random_state=0).fit_predict(x)
    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
    else:
        raise ValueError(f"unknown clustering {method}")

    found = [c for c in np.unique(labels) if c >= 0]
    by_size = sorted(found, key=lambda c: -(labels == c).sum())
    renumber = {old: new for new, old in enumerate(by_size)}
    return np.array([renumber.get(c, -1) for c in labels])


def cluster_with_anchors(x: np.ndarray, anchor: np.ndarray, p: dict) -> np.ndarray:
    """Cluster the images only, then give each class anchor the cluster whose
    center is closest. The anchors are far out (extreme logits), clustered with
    the images they would each take a cluster of their own."""
    labels = np.full(len(x), -1)
    labels[~anchor] = cluster(x[~anchor], p["clustering"], p["k"], p["eps"], p["min_samples"])
    found = sorted(set(labels[~anchor].tolist()) - {-1})
    if anchor.any() and found:
        centers = np.stack([x[~anchor & (labels == c)].mean(axis=0) for c in found])
        distance = np.linalg.norm(x[anchor][:, None] - centers[None], axis=2)
        labels[anchor] = np.array(found)[distance.argmin(axis=1)]
    return labels


def histogram_grid(values: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, float]:
    """The "histogram" layout: about sqrt(2.5 N) columns (bins) along the range
    of values, each bin's images stacked from the bottom up, lowest value first.
    The range is the images' (class anchors are put in the first or last bin).
    -> (positions in 0..1 x 0..1, size of a grid cell in the same units)"""
    n = len(values)
    real = values[~anchor] if (~anchor).any() else values
    lo, hi = float(real.min()), float(real.max())
    bins = max(1, min(n, round(math.sqrt(2.5 * n))))
    column = np.clip(((values - lo) / ((hi - lo) or 1.0) * bins).astype(int), 0, bins - 1)
    cells = np.zeros((n, 2))
    height = 1
    for b in range(bins):
        members = np.flatnonzero(column == b)
        members = members[np.argsort(values[members], kind="stable")]
        cells[members, 0] = b
        cells[members, 1] = np.arange(len(members))
        height = max(height, len(members))
    cells[:, 1] = height - 1 - cells[:, 1]      # the map's y goes down: stack from the bottom
    side = max(bins, height)
    xy = (cells + 0.5 + [(side - bins) / 2, (side - height) / 2]) / side
    return xy, 1 / side


def central_layout(values: np.ndarray, anchor: np.ndarray, top: str, shape: str,
                   by_distance: bool) -> tuple[np.ndarray, float]:
    """The "central" score layout: the highest (top "highest") or lowest
    scoring image in the middle, the others around it from the most to the
    least, class anchors last.

      rectangular  the cells of a square grid by square ring (Chebyshev
                   distance from the middle), then by distance, then clockwise
                   from the top: a dense, quasi-square block
      radial       concentric rings one image apart, each holding as many
                   images as fit around it (2 pi r). by_distance: a ring's
                   radius follows its images' average score distance from the
                   middle image, never closer than one image to the ring inside

    -> (positions in 0..1 x 0..1, size of an image in the same units)"""
    n = len(values)
    sign = -1 if top == "highest" else 1
    order = np.lexsort((sign * values, anchor))         # real images by score, then the anchors
    pos = np.zeros((n, 2))                              # in images, the middle at 0, 0; y down
    if shape == "rectangular":
        half = math.ceil(math.sqrt(n)) // 2 + 1
        g = np.arange(-half, half + 1)
        cx, cy = [a.ravel() for a in np.meshgrid(g, g)]
        clockwise = np.arctan2(cx, -cy) % (2 * np.pi)   # 0 at the top
        cells = np.lexsort((clockwise, np.hypot(cx, cy), np.maximum(abs(cx), abs(cy))))[:n]
        pos[order] = np.stack([cx[cells], cy[cells]], axis=1)
    else:
        real = order[~anchor[order]]
        distance = np.abs(values[order] - (values[real[0]] if len(real) else 0))
        distance[anchor[order]] = distance.max() if n else 0
        # by_distance: score units -> images, so that the outermost image is
        # about where it would be with the disc packed evenly
        scale = math.sqrt(n / math.pi) / (distance.max() or 1.0)
        k, radius = 1, 0.0
        while k < n:
            least = radius + 1
            r = least
            if by_distance:
                for _ in range(2):      # the ring's images depend on its radius and back
                    r = max(least, scale * distance[k:k + int(2 * math.pi * r)].mean())
            members = order[k:k + int(2 * math.pi * r)]
            angle = np.arange(len(members)) / len(members) * 2 * np.pi
            pos[members] = r * np.stack([np.sin(angle), -np.cos(angle)], axis=1)
            k, radius = k + len(members), r
    extent = np.abs(pos).max() + 0.5
    return (pos + extent) / (2 * extent), 1 / (2 * extent)


def score_grid(values: np.ndarray, anchor: np.ndarray, p: dict) -> tuple[np.ndarray, float, list | None]:
    """-> (positions, cell size, central only: how far each image's score is
    from the middle image's, 0..1 (1: the farthest), for fading them out)"""
    if p["score_layout"] != "central":
        return (*histogram_grid(values, anchor), None)
    xy, cell = central_layout(values, anchor, p["central_top"], p["central_shape"], p["radii_by_distance"])
    real = values[~anchor] if (~anchor).any() else values
    middle = real.max() if p["central_top"] == "highest" else real.min()
    distance = np.abs(values - middle) / ((real.max() - real.min()) or 1.0)
    distance[anchor] = 0                    # class anchors stay readable
    return xy, cell, np.round(distance.astype(np.float64), 4).tolist()


def cluster_grid(x: np.ndarray, labels: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, float]:
    """The "clusters" layout: a grid of int(sqrt(N)) columns. Each cluster
    starts on a new row (noise last) and fills as many rows as it needs, its
    images sorted from the most representative to the least: by distance to
    the cluster's center (the mean of its vectors), closest first. Class
    anchors (anchor = True) come first in their cluster, like a heading, and
    don't count for the center.
    -> (positions in 0..1 x 0..1, size of a grid cell in the same units)"""
    columns = max(1, int(math.sqrt(len(x))))
    cells = np.zeros((len(x), 2))
    row = 0.0
    for c in sorted(set(labels.tolist()), key=lambda c: (c < 0, c)):
        members = np.flatnonzero(labels == c)
        images = members[~anchor[members]]
        center = x[images if len(images) else members].mean(axis=0)
        distance = np.linalg.norm(x[members] - center, axis=1)
        members = members[np.lexsort((distance, ~anchor[members]))]    # anchors, then by distance
        k = np.arange(len(members))
        cells[members, 0] = k % columns
        cells[members, 1] = row + k // columns
        row += math.ceil(len(members) / columns) + CLUSTER_GAP
    width, height = columns, row - CLUSTER_GAP
    side = max(width, height)
    # cell centers, the whole grid centered in a side x side square
    xy = (cells + 0.5 + [(side - width) / 2, (side - height) / 2]) / side
    return xy, 1 / side


def run_job(job: dict, p: dict) -> None:
    try:
        with job_lock:
            if job["cancelled"]:
                raise Cancelled()
            timings = {}
            start = time.perf_counter()
            indices = sample(p["dataset"], p["max_images"])
            if not indices:
                raise ValueError("no images left on the map: restore the removed ones (Data tab)")
            raw = get_embeddings(p["model"], p["dataset"], indices, job)
            timings["embedding"] = time.perf_counter() - start

            # Classifiers: one dummy "image" per class, appended after the real
            # ones, with ids -1, -2, ... (real images have their index, >= 0)
            paths = datasets[base_dataset(p["dataset"])]
            names = [map_name(p["dataset"], i) for i in indices]
            class_names = class_names_of(p["model"])
            ids = list(indices)
            vectors = raw
            if class_names:
                vectors = np.concatenate([raw, ANCHOR_LOGIT * np.eye(len(class_names), dtype=np.float32)])
                ids += [-(c + 1) for c in range(len(class_names))]
                names += class_names
            anchor = np.array([i < 0 for i in ids])

            x = prepare(vectors, p["prep"])
            projection, note = p["projection"], None
            if x.shape[1] == 1 and projection not in GRID_LAYOUTS:
                projection, note = "histogram", "one number per image: shown as a histogram"
                p = {**p, "score_layout": "histogram"}
            grid = projection == "clusters"
            cell = None
            fade = None
            springs = None

            if projection == "spring":
                key = (p["model"], p["dataset"], tuple(indices), p["prep"], projection)
                if key not in projections:
                    job["stage"] = "preparing the vectors for the springs"
                    job["progress"] = 1.0
                    start = time.perf_counter()
                    projections[key] = (project(x, "pca", None, anchor), spring_vectors(x))
                    timings["projection"] = time.perf_counter() - start
                xy, springs = projections[key]
            elif projection == "histogram" and p["axis_mode"] != "component":
                # the model's own vectors: cosine doesn't depend on their length, standardizing would change it
                query = p["axis_prompt"] if p["axis_mode"] == "prompt" else p["axis_image"]
                xy, cell, fade = score_grid(cosine(vectors, query_vector(p["model"], p["axis_mode"], query, job, is_gray(p["dataset"]))), anchor, p)
            elif projection == "histogram":
                component = p["axis_x"] if x.shape[1] > 1 else 0
                if not 0 <= component < x.shape[1]:
                    raise ValueError(f"component {component}: the vectors have components 0 to {x.shape[1] - 1}")
                xy, cell, fade = score_grid(x[:, component], anchor, p)
            elif not grid:
                parameter = {"tsne": p["perplexity"], "umap": p["neighbors"],
                             "axes": (p["axis_x"], p["axis_y"])}.get(projection)
                key = (p["model"], p["dataset"], tuple(indices), p["prep"], projection, parameter)
                if key not in projections:
                    job["stage"] = f"projecting ({PROJECTIONS[projection]})"
                    job["progress"] = 1.0
                    start = time.perf_counter()
                    projections[key] = project(x, projection, parameter, anchor)
                    timings["projection"] = time.perf_counter() - start
                xy = projections[key]

            if job["cancelled"]:
                raise Cancelled()
            job["stage"] = f"clustering ({CLUSTERINGS[p['clustering']]})"
            start = time.perf_counter()
            # the grid has no map yet, the springs' map is made in the page: their
            # clusters are always computed on the vectors
            on_vectors = grid or projection == "spring" or p["cluster_on"] == "embedding"
            labels = cluster_with_anchors(x if on_vectors else xy, anchor, p)
            timings["clustering"] = time.perf_counter() - start
            if grid:
                xy, cell = cluster_grid(x, labels, anchor)

            job["result"] = {
                "model": p["model"],
                "dataset": p["dataset"],
                "projection": projection,           # what was used (1 number per image: histogram)
                "note": note,
                "prep": p["prep"],                  # what was used ("default" resolved)
                "prep_is_default": p["prep_is_default"],
                "kind": models[p["model"]]["kind"],
                "reconstructs": models[p["model"]]["kind"] == "autoencoder",
                # which images of the dataset (thumbnails etc.), then the class anchors (< 0)
                "indices": ids,
                "total": len(paths),                # images in the dataset
                "names": names,
                # which images have a sound (played when clicked on the map)
                "sounds": [i >= 0 and sound_of(dataset_image(p["dataset"], i)) is not None for i in ids],
                # what the page changed: images taken off the map (they can be restored)
                "removed": len(removed.get(base_dataset(p["dataset"]), ())),
                "class_names": class_names,         # classifiers only, else None
                "anchor_logit": ANCHOR_LOGIT,
                "xy": np.round(xy, 5).tolist(),
                "cell": cell,                       # grid layouts only: size of a cell, as xy
                "score_distance": fade,             # central score layout only: 0 (the middle) .. 1, per image
                "spring_vectors": springs,          # spring forces only: {shape, data (base64 float32)}
                "labels": labels.tolist(),
                "dimensions": raw.shape[1],
                "values": np.round(raw[:, 0], 5).tolist() if raw.shape[1] == 1 else None,   # scores
                # per dimension, for drawing an image's vector relative to the dataset
                "dim_mean": np.round(raw.mean(axis=0), 5).tolist(),
                "dim_std": np.round(raw.std(axis=0), 5).tolist(),
                "seconds": {k: round(v, 2) for k, v in timings.items()},
            }
            job["stage"] = "done"
    except Cancelled:
        job["stage"] = "cancelled"
    except Exception as e:
        traceback.print_exc()
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        job["finished"] = True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(WEB, "app.html")


@app.route("/<path:name>")
def static_file(name):
    return send_from_directory(WEB, name)


@app.get("/api/options")
def options():
    model_list = [{**{k: m[k] for k in ("id", "label", "kind")}, "default_prep": default_prep(m["id"]),
                   "components": components_of(m["id"]), "prompt": bool(m.get("prompt")),
                   "queries": query_kinds(m)}
                  for m in models.values() if "variant_of" not in m]
    return jsonify({
        "models": model_list,
        "datasets": dataset_list(),
        "projections": PROJECTIONS,
        "clusterings": CLUSTERINGS,
        "preps": PREPS,
        "default_model": DEFAULT_MODEL if DEFAULT_MODEL in models else (model_list[0]["id"] if model_list else None),
        "model_root": ROOT_MODEL_FOLDER,
        "dataset_root": ROOT_DATASETS_FOLDER,
        "max_images": MAX_IMAGES,
    })


@app.post("/api/rescan")
def rescan():
    scan()
    return options()


@app.post("/api/run")
def run():
    """Start computing a map. Returns a job id to poll with /api/job/<id>.
    Any earlier job still running is cancelled."""
    body = request.get_json()
    p = {
        "model": body["model"],
        "dataset": body["dataset"],
        "prep": body.get("prep", "default"),
        "projection": body.get("projection", "tsne"),
        "perplexity": float(body.get("perplexity", 30)),
        "neighbors": float(body.get("neighbors", 15)),
        "axis_x": int(body.get("axis_x", 0)),
        "axis_y": int(body.get("axis_y", 1)),
        "axis_mode": body.get("axis_mode", "component"),
        "axis_prompt": (body.get("axis_prompt") or "").strip(),
        "axis_image": body.get("axis_image"),
        "score_layout": body.get("score_layout", "histogram"),
        "central_shape": body.get("central_shape", "rectangular"),
        "central_top": body.get("central_top", "highest"),
        "radii_by_distance": bool(body.get("radii_by_distance", False)),
        "clustering": body.get("clustering", "kmeans"),
        "cluster_on": body.get("cluster_on", "embedding"),
        "k": int(body.get("k", 8)),
        "eps": float(body.get("eps", 0.5)),
        "min_samples": int(body.get("min_samples", 5)),
        "max_images": int(body.get("max_images", MAX_IMAGES)),
    }
    if p["model"] not in models:
        abort(404, f"no model {p['model']}")
    if (p["score_layout"] not in ("histogram", "central") or p["central_shape"] not in ("rectangular", "radial")
            or p["central_top"] not in ("highest", "lowest")):
        abort(400, "unknown score layout")
    if p["axis_mode"] not in QUERY_KINDS:
        abort(400, f"unknown histogram order {p['axis_mode']}")
    if p["projection"] != "histogram":
        p["axis_mode"] = "component"
    if p["axis_mode"] != "component" and p["axis_mode"] not in query_kinds(models[p["model"]]):
        abort(400, f"{models[p['model']]['label']} can't compare images with a {p['axis_mode']}")
    if p["axis_mode"] == "prompt" and not p["axis_prompt"]:
        abort(400, "type a prompt to order the images by")
    if p["axis_mode"] == "image" and p["axis_image"] not in query_images:
        abort(400, "choose the image to order by again (the server has restarted)")
    p["model"] = model_with_prompt(p["model"], body.get("prompt"))
    if base_dataset(p["dataset"]) not in datasets:
        abort(404, f"no dataset {p['dataset']}")
    if p["projection"] not in PROJECTIONS or p["clustering"] not in CLUSTERINGS:
        abort(400, "unknown projection or clustering")
    if p["prep"] != "default" and p["prep"] not in PREPS:
        abort(400, f"unknown vector preparation {p['prep']}")
    # anything other than "default" overrides the model's default
    p["prep_is_default"] = p["prep"] == "default"
    if p["prep_is_default"]:
        p["prep"] = default_prep(p["model"])

    for old in jobs.values():
        old["cancelled"] = True
    job_id = next(next_job)
    jobs[job_id] = {"id": job_id, "stage": "waiting", "progress": 0.0, "cancelled": False,
                    "finished": False, "result": None, "error": None}
    threading.Thread(target=run_job, args=(jobs[job_id], p), daemon=True).start()
    return jsonify({"job": job_id})


@app.get("/api/job/<int:job_id>")
def job_status(job_id: int):
    if job_id not in jobs:
        abort(404)
    job = jobs[job_id]
    if job["finished"]:
        jobs.pop(job_id)    # the result is collected once
    return jsonify({k: job[k] for k in ("id", "stage", "progress", "finished", "result", "error")})


def dataset_image(dataset_id: str, i: int) -> Path:
    """The file of image i on a dataset's map: one of the dataset, or one added from the page"""
    dataset_id = base_dataset(dataset_id)
    if dataset_id in datasets and ADDED_BASE <= i < ADDED_BASE + len(added_files):
        return added_files[i - ADDED_BASE][0]
    if dataset_id not in datasets or not 0 <= i < len(datasets[dataset_id]):
        abort(404)
    return datasets[dataset_id][i]


@app.post("/api/set/remove")
def set_remove():
    """Take an image off a dataset's map (it stays in its folder)"""
    body = request.get_json()
    dataset_id, i = base_dataset(body["dataset"]), int(body["i"])
    dataset_image(dataset_id, i)        # 404 if there is no such image
    removed.setdefault(dataset_id, set()).add(i)
    return jsonify({"removed": len(removed[dataset_id])})


@app.post("/api/set/restore")
def set_restore():
    """Every image taken off a dataset's map comes back"""
    removed.pop(base_dataset(request.get_json()["dataset"]), None)
    return jsonify({"removed": 0})


@app.post("/api/set/add")
def set_add():
    """Images from the page's computer onto a dataset's map (form: dataset,
    images). They are copied into ADDED_FOLDER, not into the dataset's folder."""
    dataset_id = base_dataset(request.form["dataset"])
    if dataset_id not in datasets:
        abort(404, f"no dataset {dataset_id}")
    ADDED_FOLDER.mkdir(parents=True, exist_ok=True)
    ids = []
    for file in request.files.getlist("images"):
        name = Path(file.filename or "image").name
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            abort(400, f"{name}: not one of {', '.join(sorted(IMAGE_EXTENSIONS))}")
        path = ADDED_FOLDER / f"{uuid.uuid4().hex}{suffix}"
        file.save(path)
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            path.unlink(missing_ok=True)
            abort(400, f"{name}: can't read it as an image")
        added_files.append((path, name))
        ids.append(ADDED_BASE + len(added_files) - 1)
    added.setdefault(dataset_id, []).extend(ids)
    return jsonify({"added": ids})


@app.post("/api/query-image")
def query_image():
    """An image from the page's computer to order the histogram by (form:
    image). Kept in memory, not saved. -> {"id"}"""
    file = request.files.get("image")
    if file is None:
        abort(400, "no image")
    try:
        image = Image.open(io.BytesIO(file.read()))
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        abort(400, f"{file.filename}: can't read it as an image")
    image.thumbnail((LOAD_SIZE * 2, LOAD_SIZE * 2))
    query_id = uuid.uuid4().hex
    query_images[query_id] = image
    return jsonify({"id": query_id})


def thumbnail_jpeg(dataset_id: str, i: int) -> bytes:
    key = (dataset_id, i)
    if key not in thumbnails:
        thumbnails[key] = jpeg(open_dataset_image(dataset_id, i), THUMBNAIL_SIZE)
    return thumbnails[key]


@app.get("/api/thumbnail")
def thumbnail():
    response = send_file(io.BytesIO(thumbnail_jpeg(request.args["dataset"], int(request.args["i"]))),
                         mimetype="image/jpeg")
    response.headers["Cache-Control"] = "max-age=3600"
    return response


@app.get("/api/sound")
def sound():
    """The sound file next to an image (same name), if there is one"""
    path = sound_of(dataset_image(request.args["dataset"], int(request.args["i"])))
    if path is None:
        abort(404, "this image has no sound")
    mimetype = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".flac": "audio/flac"}
    return send_file(path, mimetype=mimetype.get(path.suffix.lower(), "application/octet-stream"))


@app.get("/api/preview")
def preview():
    """An image of a dataset, at most size px (default PREVIEW_SIZE)"""
    dataset_id, i = request.args["dataset"], int(request.args["i"])
    size = min(int(request.args.get("size", PREVIEW_SIZE)), RANKING_IMAGE_SIZE)
    if size <= LOAD_SIZE * 2:
        image = open_dataset_image(dataset_id, i)
    else:
        image = Image.open(dataset_image(dataset_id, i)).convert("L" if is_gray(dataset_id) else "RGB").convert("RGB")
    return send_file(io.BytesIO(jpeg(image, size)), mimetype="image/jpeg")


@app.get("/api/embedding")
def embedding():
    """The vector of one image (as the model gives it)"""
    vector = embeddings.get((request.args["model"], request.args["dataset"]), {}).get(int(request.args["i"]))
    if vector is None:
        abort(404, "not computed yet")
    return jsonify({"values": np.round(vector, 5).tolist()})


@app.get("/api/reconstruction")
def reconstruction():
    """What an autoencoder makes of one image (PNG, at the model's size)"""
    model_id = request.args["model"]
    source = open_dataset_image(request.args["dataset"], int(request.args["i"]))
    if models.get(model_id, {}).get("kind") != "autoencoder":
        abort(400, "only autoencoders reconstruct")
    with model_lock:
        load_model(model_id)
        image = loaded["embedder"].reconstruct([source])[0]
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return send_file(io.BytesIO(buffer.getvalue()), mimetype="image/png")


def prepare_like(reference: np.ndarray, vectors: np.ndarray, prep: str) -> tuple[np.ndarray, np.ndarray]:
    """prepare() both, with the reference's statistics for "standardize", so
    new vectors are measured the same way as the dataset's."""
    if prep == "standardize":
        mean, std = reference.mean(axis=0), np.maximum(reference.std(axis=0), 1e-8)
        return (reference - mean) / std, (vectors - mean) / std
    return prepare(reference, prep), prepare(vectors, prep)


def data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@app.post("/api/interpolate")
def interpolate():
    """Walk in a straight line from the code of image a to the code of image b.

    The `steps` points in between are decoded into images, and those images
    are encoded again (the decoder's image doesn't give back exactly the code
    it was made from). Each one is placed among its INTERPOLATION_NEIGHBORS
    nearest dataset images: the browser puts it at their weighted average
    position on its map, which works for every projection (t-SNE has no way to
    add new points to a map it made) and for the grid."""
    body = request.get_json()
    model_id, dataset_id = body["model"], body["dataset"]
    a, b = int(body["a"]), int(body["b"])
    steps = max(1, min(int(body.get("steps", 8)), 100))
    if models.get(model_id, {}).get("kind") != "autoencoder":
        abort(400, "only autoencoders interpolate")
    prep = body.get("prep", "default")
    if prep == "default":
        prep = default_prep(model_id)

    # the images on the map (computed by /api/run already, so normally cached)
    indices = sample(dataset_id, int(body.get("max_images", MAX_IMAGES)))
    reference = get_embeddings(model_id, dataset_id, indices, {"cancelled": False})

    with model_lock:
        load_model(model_id)
        autoencoder = loaded["embedder"].model
        ends = autoencoder.embed([open_dataset_image(dataset_id, i) for i in (a, b)])
        t_all = torch.linspace(0, 1, steps + 2)[:, None]       # 0 = a ... 1 = b
        codes = (1 - t_all) * ends[0] + t_all * ends[1]
        with torch.inference_mode():
            device = next(autoencoder.parameters()).device
            decoded_all = [autoencoder.to_image(x) for x in autoencoder.decode(codes.to(device))]
        # a and b themselves are already on the map: only the ones in between are
        # placed; the decoded a and b are the first and last frame of the animation
        t, decoded = t_all[1:-1], decoded_all[1:-1]
        vectors = autoencoder.embed(decoded).numpy()

    ref, new = prepare_like(reference, vectors, prep)
    distance = np.linalg.norm(new[:, None] - ref[None], axis=2)        # [steps, N]
    k = min(INTERPOLATION_NEIGHBORS, len(indices))
    nearest = np.argsort(distance, axis=1)[:, :k]
    weights = 1 / (np.take_along_axis(distance, nearest, axis=1) + 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)

    return jsonify({
        "model": model_id, "dataset": dataset_id, "a": a, "b": b,
        "a_image": data_url(decoded_all[0]), "b_image": data_url(decoded_all[-1]),
        "steps": [{
            "t": round(float(t[s]), 4),
            "image": data_url(decoded[s]),
            "vector": np.round(vectors[s], 5).tolist(),
            "neighbors": [indices[j] for j in nearest[s]],
            "weights": np.round(weights[s], 5).tolist(),
        } for s in range(steps)],
    })


def cover_jpeg(image: Image.Image, w: float, h: float, side: int) -> bytes:
    """The image cropped to the proportions of a w x h box and scaled to `side`
    pixels along its longer side, as JPEG (CSS background-size: cover)"""
    scale = side / max(w, h, 1e-6)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    buffer = io.BytesIO()
    ImageOps.fit(image.convert("RGB"), size, Image.LANCZOS).save(buffer, "JPEG", quality=88)
    return buffer.getvalue()


@app.post("/api/export/pdf")
def export_pdf():
    """The map as the page shows it, as a PDF. The page sends every mark it
    draws (see map_pdf.py); here the images get their pictures: about 2 image
    pixels per pixel on the screen, so they stay sharp when printed."""
    spec = request.get_json()
    for mark in spec["marks"]:
        if mark["kind"] != "image":
            continue
        side = min(PREVIEW_SIZE, max(THUMBNAIL_SIZE, round(2 * max(mark["w"], mark["h"]))))
        if mark.get("src"):            # an interpolated image, a data URL
            image = Image.open(io.BytesIO(base64.b64decode(mark.pop("src").split(",", 1)[1])))
        elif side <= THUMBNAIL_SIZE:
            image = Image.open(io.BytesIO(thumbnail_jpeg(spec["dataset"], int(mark["i"]))))
        else:
            image = open_dataset_image(spec["dataset"], int(mark["i"]))
        mark["jpeg"] = cover_jpeg(image, mark["w"], mark["h"], side)
    return send_file(io.BytesIO(map_pdf.make_pdf(spec)), mimetype="application/pdf")


# ---------------------------------------------------------------------------
# Training (the Training tab of the page, see training_engine.py)
# ---------------------------------------------------------------------------

training = {"run": None}        # the current (or last) TrainingRun

# learning rate the page suggests for each autoencoder kind
AUTOENCODER_LEARNING_RATES = {AutoencoderType.ATTENTION: 1e-4, AutoencoderType.VAE_GAN: 2e-4}

# CLIP weights that exist for each architecture (open_clip.list_pretrained())
CLIP_WEIGHTS = {
    CLIPArch.VITB32: [CLIPPretrained.OPENAI, CLIPPretrained.LAION2B_S34B_B79K, CLIPPretrained.DATACOMP_XL_S13B_B90K],
    CLIPArch.VITL14: [CLIPPretrained.OPENAI, CLIPPretrained.DATACOMP_XL_S13B_B90K],
}

NAME_CHARACTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")


def backbone_choices() -> list[dict]:
    """Every backbone a classifier can be trained on, for the page's dropdown"""
    choices = []
    for family in (ClassifierBackbone.DINOV2, ClassifierBackbone.CLIP, ClassifierBackbone.IMAGE_NET):
        for arch in ARCHS[family]:
            weights = CLIP_WEIGHTS.get(arch, [None]) if family == ClassifierBackbone.CLIP else [None]
            for pretrained in weights:
                value = {"backbone": family.value, "arch": arch.value,
                         "pretrained": pretrained.value if pretrained else None}
                label = f"{family.value}  {arch.value}" + (f"  {pretrained.value}" if pretrained else "")
                choices.append({"id": "|".join(str(v) for v in value.values()), "label": label, "value": value})
    return choices


def ranknet_backbones() -> list[dict]:
    """Every pretrained encoder a RankNet can use (one or several, concatenated)"""
    choices = [{"id": str(ModelID.dino(arch)), "label": f"DINOv2 {arch.value}"} for arch in DINOV2Arch]
    for arch, weights in CLIP_WEIGHTS.items():
        choices += [{"id": str(ModelID.clip(arch, w)), "label": f"CLIP {arch.value} {w.value}"} for w in weights]
    return choices


def dataset_names(dataset_id: str) -> list[str]:
    """The names of a dataset's images as ranking files store them: their path
    inside the dataset's folder"""
    return [image_name(dataset_id, p) for p in datasets[dataset_id]]


def ranked_datasets() -> list[dict]:
    """The datasets with ranking files in their folder, with what is in them"""
    found = []
    for d in dataset_list():
        files = rankings.ranking_files(dataset_folder(d["id"]))
        if files:
            names = dataset_names(d["id"])
            found.append({**d, "rankings": [rankings.summary(f, names) for f in files]})
    return found


def model_saved(folder: Path) -> None:
    """A training run saved a model: show it in the model list, and forget
    whatever was computed with an earlier version of it."""
    model_id = f"folder/{folder.relative_to(ROOT_MODEL_FOLDER).as_posix()}"
    for key in [k for k in embeddings if k[0] == model_id]:
        embeddings.pop(key)
    for key in [k for k in projections if k[0] == model_id]:
        projections.pop(key)
    with model_lock:
        if loaded["id"] == model_id:
            loaded["id"], loaded["embedder"] = None, None
    scan_models()


def model_folder(name: str) -> Path:
    if not name or not set(name) <= NAME_CHARACTERS or name.startswith(".") or ".." in name:
        abort(400, "a model name can only have letters, digits, _ - and . (not at the start)")
    return Path(ROOT_MODEL_FOLDER) / name


@app.get("/api/train/options")
def train_options():
    autoencoder_types = {t.value: {"defaults": AUTOENCODERS[t].DEFAULTS,
                                   "learning_rate": AUTOENCODER_LEARNING_RATES.get(t, 1e-3)}
                         for t in AutoencoderType}
    return jsonify({
        "models_folder": ROOT_MODEL_FOLDER,
        "autoencoder_types": autoencoder_types,
        "backbones": backbone_choices(),
        # an autoencoder trains on the images of any dataset of the map
        "autoencoder_datasets": dataset_list(),
        "classifier_datasets": [{"id": d, "label": "(datasets folder)" if d == "." else d, **info}
                                for d, info in class_datasets.items()],
        # a RankNet trains on a ranking file of a dataset
        "ranknet_datasets": ranked_datasets(),
        "ranknet_backbones": ranknet_backbones(),
    })


@app.get("/api/train/model")
def train_model_info():
    """Is there a model with this name already? What is it?"""
    folder = model_folder(request.args.get("name", ""))
    if not (folder / "config.json").exists():
        return jsonify({"exists": False, "folder": str(folder)})
    config = read_config(folder)
    history = folder / "history.json"
    epochs = len(json.loads(history.read_text())) if history.exists() else None
    kind = folder_model_type(folder)
    if kind == "autoencoder":
        summary = f"autoencoder {config['autoencoder_type']}, {config['image_size']}px, {config['latent_dim']} latent"
    elif kind == "classifier":
        summary = (f"classifier {config['backbone']} {config['arch']}, {config['head']} head, "
                   f"{len(config['class_names'])} classes")
    elif kind == "ranknet":
        trained = config.get("trained_on") or {}
        summary = (f"RankNet {' + '.join(config['model_ids'])}, head {config['hidden_dim']}"
                   + (f", trained on {trained['dataset']} / {trained['ranking']}" if trained else ""))
    else:
        summary = "not a model this page can train"
    return jsonify({"exists": True, "folder": str(folder), "kind": kind, "summary": summary,
                    "epochs": epochs, "config": config})


@app.post("/api/train/start")
def train_start():
    run = training["run"]
    if run and run.active:
        abort(409, f"{run.settings['name']} is still training, stop it first")
    s = request.get_json()
    folder = model_folder(s.get("name", ""))
    exists = (folder / "config.json").exists()
    if exists and not s.get("continue") and not s.get("overwrite"):
        return jsonify({"exists": True, "error": f"{folder.name} already exists: tick 'continue' to keep "
                                                 f"training it, or confirm to overwrite it"}), 409
    if exists and s.get("continue") and folder_model_type(folder) != s["kind"]:
        abort(400, f"{folder.name} is a {folder_model_type(folder)}, not a {s['kind']}")
    available = class_datasets if s["kind"] == "classifier" else datasets
    if s.get("dataset") not in available:
        abort(400, f"pick a dataset ({'one with class subfolders' if s['kind'] == 'classifier' else 'any'})")
    if s["kind"] == "ranknet":
        ranking = dataset_folder(s["dataset"]) / str(s.get("ranking", ""))
        if ranking.parent != dataset_folder(s["dataset"]) or ranking not in rankings.ranking_files(ranking.parent):
            abort(400, "pick a ranking file of the dataset")
        s["ranking_path"] = str(ranking)
        s["image_paths"] = [str(p) for p in datasets[s["dataset"]]]
        s["image_names"] = dataset_names(s["dataset"])
    elif s["kind"] == "classifier":
        s["dataset_path"] = str(folder_path(s["dataset"]))
    else:
        # exactly the images of that dataset ("all of a/" or the images directly in a folder)
        s["dataset_path"] = str(dataset_folder(s["dataset"]))
        s["image_paths"] = [str(p) for p in datasets[s["dataset"]]]

    run = TrainingRun(s, folder, DEVICE, on_saved=model_saved)
    training["run"] = run
    run.start()
    return jsonify({"started": True})


@app.get("/api/train/status")
def train_status():
    run = training["run"]
    if run is None:
        return jsonify({"state": "idle"})
    return jsonify(run.status(int(request.args.get("preview", -1))))


@app.post("/api/train/stop")
def train_stop():
    run = training["run"]
    if run is None or not run.active:
        abort(409, "nothing is training")
    run.request_stop(save=bool(request.get_json().get("save")))
    return jsonify({"stopping": True})


# ---------------------------------------------------------------------------
# Ranking (the Dataset tab's Ranking: pairs of images, which one is better)
# ---------------------------------------------------------------------------
#
# The page draws the pairs and keeps the answers; the server only reads and
# writes the ranking files (ranknet/rankings.py). Indices are those of the
# dataset's images (the same as everywhere else on the page).

def ranking_file(dataset_id: str, file: str) -> Path:
    if dataset_id not in datasets:
        abort(404, f"no dataset {dataset_id}")
    path = dataset_folder(dataset_id) / file
    if path not in rankings.ranking_files(path.parent):
        abort(404, f"no ranking {file} in {dataset_id}")
    return path


@app.get("/api/ranking/list")
def ranking_list():
    dataset_id = request.args["dataset"]
    if dataset_id not in datasets:
        abort(404, f"no dataset {dataset_id}")
    folder = dataset_folder(dataset_id)
    names = dataset_names(dataset_id)
    return jsonify({"folder": str(folder), "count": len(names),
                    "rankings": [rankings.summary(f, names) for f in rankings.ranking_files(folder)]})


@app.post("/api/ranking/new")
def ranking_new():
    body = request.get_json()
    dataset_id = body["dataset"]
    if dataset_id not in datasets:
        abort(404, f"no dataset {dataset_id}")
    try:
        path = rankings.ranking_path(dataset_folder(dataset_id), body.get("name", "").strip())
    except ValueError as e:
        abort(400, str(e))
    if path.exists():
        abort(409, f"{path.name} exists already")
    rankings.save_ranking(path, dataset_names(dataset_id), [], [], body.get("question", ""))
    return jsonify({"file": path.name})


@app.get("/api/ranking/open")
def ranking_open():
    dataset_id = request.args["dataset"]
    path = ranking_file(dataset_id, request.args["file"])
    ranking = rankings.read_ranking(path)
    pairs, repeats, dropped = rankings.remap(ranking, dataset_names(dataset_id))
    return jsonify({"file": path.name, "label": rankings.label_of(path), "legacy": rankings.is_legacy(path),
                    "question": ranking["question"], "created": ranking["created"],
                    "count": len(datasets[dataset_id]), "pairs": pairs, "repeats": repeats, "dropped": dropped,
                    # a dataset made by the Sampler: the cluster of every image
                    "clusters": dataset_sampler.clusters_of(dataset_folder(dataset_id), dataset_names(dataset_id))})


def ranking_helper_job(job: dict, p: dict) -> None:
    """For the Ranking page's choice of pairs, for every image of a dataset:
    "regions"   a cluster (spherical k-means on a model's vectors): pairs of
                alike images
    "scores"    a model's score (a RankNet, an aesthetic score): pairs close
                in its ranking, the ones it is least sure about"""
    try:
        indices = list(range(len(datasets[p["dataset"]])))
        vectors = get_embeddings(p["model"], p["dataset"], indices, job)
        if p["kind"] == "scores":
            if vectors.shape[1] != 1:
                raise ValueError(f"{models[p['model']]['label']} gives {vectors.shape[1]} numbers per image, not a score")
            job["result"] = {"scores": np.round(vectors[:, 0], 5).tolist()}
        else:
            k = max(2, min(50, len(indices) // 6))
            features = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            labels, _ = dataset_sampler.spherical_kmeans(features, k)
            job["result"] = {"clusters": labels.tolist(), "k": k}
        job["stage"] = "done"
    except Cancelled:
        job["stage"] = "cancelled"
    except Exception as e:
        traceback.print_exc()
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        job["finished"] = True


@app.post("/api/ranking/helper")
def ranking_helper():
    """Start a ranking_helper_job, poll it with /api/job/<id>"""
    body = request.get_json()
    if body.get("dataset") not in datasets or body.get("model") not in models:
        abort(404, "unknown dataset or model")
    if body.get("kind") not in ("regions", "scores"):
        abort(400, "kind is regions or scores")
    p = {"dataset": body["dataset"], "kind": body["kind"], "model": model_with_prompt(body["model"], body.get("prompt"))}
    job_id = next(next_job)
    jobs[job_id] = {"id": job_id, "stage": "waiting", "progress": 0.0, "cancelled": False,
                    "finished": False, "result": None, "error": None}
    threading.Thread(target=ranking_helper_job, args=(jobs[job_id], p), daemon=True).start()
    return jsonify({"job": job_id})


@app.post("/api/ranking/save")
def ranking_save():
    """Write the page's answers. A file in DiffDrawing's older format is not
    overwritten: the answers go to ranking_<its name>.json"""
    body = request.get_json()
    dataset_id = body["dataset"]
    path = ranking_file(dataset_id, body["file"])
    if rankings.is_legacy(path):
        name = path.stem[len(rankings.LEGACY_PREFIX):].lstrip("_") or "ranked_pairs"
        path = rankings.ranking_path(path.parent, name)
    n = len(datasets[dataset_id])
    valid = lambda e: len(e) >= 3 and all(isinstance(v, int) for v in e) and 0 <= e[0] < n and 0 <= e[1] < n
    pairs = [e[:3] for e in body.get("pairs", []) if valid(e)]
    repeats = [e[:4] for e in body.get("repeats", []) if valid(e) and len(e) >= 4]
    rankings.save_ranking(path, dataset_names(dataset_id), pairs, repeats, body.get("question", ""),
                          body.get("created"))
    return jsonify({"file": path.name, "pairs": len(pairs)})


# ---------------------------------------------------------------------------
# Dataset (the Dataset tab of the page, see dataset_tools.py)
# ---------------------------------------------------------------------------

conversion = {"run": None}      # the current (or last) ConversionRun
PREVIEW_IMAGES = 4
PREVIEW_MAX_SIDE = 768          # previews are converted at most this big (fast)

DATASET_NAME_CHARACTERS = NAME_CHARACTERS | {"/"}


def new_dataset_folder(name: str) -> Path:
    name = name.strip().strip("/")
    parts = name.split("/")
    if not name or not set(name) <= DATASET_NAME_CHARACTERS or any(not p or p.startswith(".") for p in parts):
        abort(400, "a dataset name can only have letters, digits, _ - . and / between folders")
    return Path(ROOT_DATASETS_FOLDER) / name


def conversion_settings(body: dict) -> tuple[str, dict]:
    kind = body.get("conversion")
    if kind not in CONVERSIONS:
        abort(400, f"unknown conversion {kind}")
    defaults = CONVERSIONS[kind]["params"]
    params = {k: type(v)(body.get("params", {}).get(k, v)) for k, v in defaults.items()}
    if "depth_model" in params and params["depth_model"] not in DEPTH_MODELS:
        abort(400, f"unknown depth model {params['depth_model']}")
    return kind, params


@app.get("/api/convert/options")
def convert_options():
    return jsonify({"conversions": CONVERSIONS, "depth_models": list(DEPTH_MODELS), "formats": list(FORMATS),
                    "datasets": dataset_list(), "datasets_folder": ROOT_DATASETS_FOLDER})


@app.get("/api/convert/check")
def convert_check():
    """Is there a folder with this name already?"""
    folder = new_dataset_folder(request.args.get("name", ""))
    files = sum(1 for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else 0
    return jsonify({"folder": str(folder), "exists": folder.exists(), "files": files})


@app.post("/api/convert/preview")
def convert_preview():
    """A few images of the dataset, before and after (not saved)"""
    body = request.get_json()
    dataset_id = body.get("dataset")
    if dataset_id not in datasets:
        abort(404, f"no dataset {dataset_id}")
    kind, params = conversion_settings(body)
    order = sample_orders[dataset_id]
    offset = int(body.get("offset", 0)) * PREVIEW_IMAGES     # "other images" button
    picks = [order[(offset + k) % len(order)] for k in range(min(PREVIEW_IMAGES, len(order)))]
    paths = [datasets[dataset_id][i] for i in picks]
    max_side = int(body.get("max_side") or 0)
    images = [dataset_load(p, min(max_side, PREVIEW_MAX_SIDE) if max_side else PREVIEW_MAX_SIDE) for p in paths]
    start = time.perf_counter()
    try:
        converted = convert_images(images, kind, params, DEVICE)
    except RuntimeError as e:
        abort(500, str(e))
    return jsonify({"seconds": round(time.perf_counter() - start, 2), "images": [
        {"name": image_name(dataset_id, p), "before": dataset_data_url(a, 320), "after": dataset_data_url(b, 320)}
        for p, a, b in zip(paths, images, converted)]})


@app.post("/api/convert/start")
def convert_start():
    run = conversion["run"]
    if run and run.active:
        abort(409, "a conversion is still running, stop it first")
    body = request.get_json()
    dataset_id = body.get("dataset")
    if dataset_id not in datasets:
        abort(400, "pick a source dataset")
    kind, params = conversion_settings(body)
    target = new_dataset_folder(body.get("name", ""))
    source = dataset_folder(dataset_id)
    if target.resolve() == source.resolve() or source.resolve() in target.resolve().parents \
            or target.resolve() in source.resolve().parents:
        abort(400, "the new dataset can't be inside the source folder (or contain it)")
    if target.exists() and any(target.iterdir()) and not body.get("overwrite"):
        return jsonify({"exists": True, "error": f"{target} exists and is not empty"}), 409
    fmt = body.get("format", "png")
    if fmt not in FORMATS:
        abort(400, f"unknown format {fmt}")

    run = ConversionRun(datasets[dataset_id], source, target, kind, params, fmt, int(body.get("max_side") or 0),
                        DEVICE, on_done=scan)
    conversion["run"] = run
    run.start()
    return jsonify({"started": True, "target": str(target)})


# --- audio -> spectrogram dataset (the Audio side tab, audio_tools.py) --------

audio_clips = {}                # token -> AudioClip (the last few uploaded)
AUDIO_PREVIEWS = 4
UPLOADS = Path(tempfile.gettempdir()) / "qa2026_audio_uploads"


def audio_clip(body: dict) -> AudioClip:
    clip = audio_clips.get(body.get("token"))
    if clip is None:
        abort(404, "upload the sound file again (the server was restarted or it is too old)")
    return clip


def audio_settings(body: dict) -> dict:
    try:
        return audio_tools.settings(body.get("settings", {}))
    except (ValueError, TypeError) as e:
        abort(400, str(e))


@app.post("/api/audio/upload")
def audio_upload():
    """A wav / mp3 from the page: saved in a temporary folder, decoded once"""
    file = request.files.get("file")
    if file is None or not file.filename:
        abort(400, "no file")
    token = uuid.uuid4().hex
    folder = UPLOADS / token
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(file.filename).name
    file.save(path)
    clip = AudioClip(path, Path(file.filename).name)
    try:
        samples = clip.samples(audio_tools.DEFAULTS["sample_rate"])
    except Exception as e:
        abort(400, f"can't read {file.filename} as sound: {e}")
    for old in list(audio_clips)[:-2]:          # keep the last 3
        audio_clips.pop(old)
    audio_clips[token] = clip
    return jsonify({"token": token, "name": clip.name,
                    "duration": round(len(samples) / audio_tools.DEFAULTS["sample_rate"], 2)})


@app.post("/api/audio/preview")
def audio_preview():
    """A few fragments with their spectrogram and sound (not saved)"""
    body = request.get_json()
    clip, p = audio_clip(body), audio_settings(body)
    sr = int(p["sample_rate"])
    y = clip.samples(sr)
    starts = audio_tools.fragment_starts(len(y), sr, p["duration"], p["overlap"])
    if not starts:
        abort(400, f"the sound is shorter than one fragment ({p['duration']} s)")
    # evenly spread over the sound; "other fragments" shifts them
    offset = int(body.get("offset", 0))
    count = min(AUDIO_PREVIEWS, len(starts))
    picks = sorted({(k * len(starts) // count + offset) % len(starts) for k in range(count)})
    top_db = audio_tools.clip_top_db(y, sr, p) if p["normalization"] == "clip" else None
    length = round(p["duration"] * sr)
    out = []
    for k in picks:
        fragment = y[starts[k]:starts[k] + length]
        out.append({"index": k, "start": round(starts[k] / sr, 2), "end": round((starts[k] + length) / sr, 2),
                    "image": audio_tools.image_data_url(audio_tools.spectrogram(fragment, sr, p, top_db)),
                    "sound": audio_tools.sound_data_url(fragment, sr)})
    return jsonify({"fragments": len(starts), "duration": round(len(y) / sr, 2), "previews": out})


@app.post("/api/audio/start")
def audio_start():
    run = conversion["run"]
    if run and run.active:
        abort(409, "a conversion is still running, stop it first")
    body = request.get_json()
    clip, p = audio_clip(body), audio_settings(body)
    target = new_dataset_folder(body.get("name", ""))
    if target.exists() and any(target.iterdir()) and not body.get("overwrite"):
        return jsonify({"exists": True, "error": f"{target} exists and is not empty"}), 409
    run = AudioRun(clip, target, p, on_done=scan)
    conversion["run"] = run
    run.start()
    return jsonify({"started": True, "target": str(target)})


# ---------------------------------------------------------------------------
# Sampler (the Dataset tab's Sampler, see dataset_sampler.py), in the same
# slot and monitor as the conversions
# ---------------------------------------------------------------------------

@app.get("/api/sample/options")
def sample_options():
    # models whose vectors can be clustered: not the ones that give one number
    usable = [{"id": m["id"], "label": m["label"], "kind": m["kind"]} for m in models.values()
              if m["kind"] not in ("ranknet", "aesthetic") and "variant_of" not in m]
    return jsonify({"models": usable, "defaults": dataset_sampler.DEFAULTS, "groupings": dataset_sampler.GROUPINGS,
                    "default_model": "pretrained/clip_b32_datacomp", "datasets": dataset_list(),
                    "datasets_folder": ROOT_DATASETS_FOLDER})


@app.post("/api/sample/start")
def sample_start():
    run = conversion["run"]
    if run and run.active:
        abort(409, "a conversion is still running, stop it first")
    body = request.get_json()
    dataset_id, model_id = body.get("dataset"), body.get("model")
    if dataset_id not in datasets or model_id not in models:
        abort(404, "unknown dataset or model")
    defaults = dataset_sampler.DEFAULTS
    p = {k: type(v)(body.get("params", {}).get(k, v)) for k, v in defaults.items()}
    if p["group_by"] not in dataset_sampler.GROUPINGS:
        abort(400, f"unknown grouping {p['group_by']}")
    if p["count"] < 1 or p["clusters"] < 1:
        abort(400, "count and clusters must be at least 1")
    target = new_dataset_folder(body.get("name", ""))
    if target.exists() and any(target.iterdir()) and not body.get("overwrite"):
        return jsonify({"exists": True, "error": f"{target} exists and is not empty"}), 409
    indices = list(range(len(datasets[dataset_id])))
    run = dataset_sampler.SamplerRun(
        datasets[dataset_id], dataset_names(dataset_id), dataset_id, target, p, models[model_id]["label"],
        embed=lambda job: get_embeddings(model_id, dataset_id, indices, job), on_done=scan)
    conversion["run"] = run
    run.start()
    return jsonify({"started": True, "target": str(target)})


@app.get("/api/convert/status")
def convert_status():
    run = conversion["run"]
    return jsonify(run.status() if run else {"state": "idle"})


@app.post("/api/convert/stop")
def convert_stop():
    run = conversion["run"]
    if run is None or not run.active:
        abort(409, "nothing is converting")
    run.stop()
    return jsonify({"stopping": True})


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    scan()
    # no debug reloader: it would start the app (and load the model) twice
    app.run(port=PORT, threaded=True)
