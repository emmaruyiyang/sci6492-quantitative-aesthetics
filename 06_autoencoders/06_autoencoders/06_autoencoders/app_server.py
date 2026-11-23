"""The web app of this folder (web/app.html), three tabs:

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
at least one image.

Projections squeeze the vectors (128, 768, ... numbers) down to 2:

  PCA    the 2 directions along which the vectors spread the most. Linear, fast,
         keeps large distances, but crowds most images into the middle.
  t-SNE  keeps neighbours together: images close in the vector space end up
         close on the map. Distances between far-apart groups mean little.
         perplexity ~ how many neighbours each image cares about.
  UMAP   like t-SNE, faster, keeps somewhat more of the global layout.
         neighbours = how many neighbours each image cares about.

  clusters  not a projection of the vectors: a grid, int(sqrt(N)) columns,
         one section of rows per cluster (computed on the vectors), each
         cluster's images from left to right from the most representative
         (closest to the cluster's center) to the least.

Clusterings group the images, on the vectors or on the 2D map:

  k-means   k groups, each image goes to the nearest of k centers
  spectral  k groups that are connected chains of neighbours, can find
            groups of any shape
  DBSCAN    groups are regions dense with images, the number of groups is
            found by itself. eps = how close counts as neighbour, images in no
            dense region are noise (-1, gray).

The browser only gets coordinates and cluster numbers; the images glide to
their new place whenever something changes.

Autoencoders can also interpolate: pick two images A and B on the map, the
codes on the straight line between their codes are decoded into images,
encoded again and placed on the map among their nearest images, joined by a
line from A to B (/api/interpolate).

The Training tab of the page trains a new autoencoder or classifier (or keeps
training a saved one) into ROOT_MODEL_FOLDER, in a background thread, and
shows its progress, loss curves and a preview after every epoch; it can be
stopped at any moment, with or without saving (training_engine.py,
/api/train/...). A classifier trains on a folder with one subfolder of
images per class.

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
import uuid
from pathlib import Path

import numpy as np
import torch
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image

from autoencoder_model import AUTOENCODERS, AutoencoderType, folder_model_type, read_config
from classifier_model import ARCHS, CLIPArch, CLIPPretrained, ClassifierBackbone
from embedders import find_models, load_embedder
from training_engine import TrainingRun
import audio_tools
from audio_tools import AUDIO_EXTENSIONS, AudioClip, AudioRun
from dataset_tools import (CONVERSIONS, DEPTH_MODELS, FORMATS, ConversionRun, convert as convert_images,
                           data_url as dataset_data_url, load as dataset_load)

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_MODEL_FOLDER = r"D:\ML\QA2026\models"
ROOT_DATASETS_FOLDER = r"D:\ML\QA2026\datasets"
DEFAULT_MODEL = "pretrained/dinov2_vits14"

MAX_IMAGES = 1000           # default of the page's "max images": bigger datasets use a random sample
LOAD_SIZE = 256             # images are decoded at about this size (enough for every model)
THUMBNAIL_SIZE = 96         # images on the map
PREVIEW_SIZE = 320          # image in the side panel
BATCH_SIZE = 32
PORT = 5002

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROJECTIONS = {"tsne": "t-SNE", "umap": "UMAP", "pca": "PCA", "clusters": "clusters (grid)"}
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
loaded = {"id": None, "embedder": None}

models = {}                 # id -> {"id", "label", "kind", "path"}
datasets = {}               # id -> list of all its image paths
dataset_info = {}           # id -> {"group", "label", "folder", "recursive"}, see scan()
sound_files = {}            # image path without extension -> its sound file (same name, next to it)
class_datasets = {}         # id -> {"classes", "count"}: folders with one subfolder of images per class
sample_orders = {}          # id -> the image indices in a fixed random order
embeddings = {}             # (model id, dataset id) -> {image index: vector}
projections = {}            # (model id, dataset id, number of images, prep, method, parameter) -> array [N, 2]
thumbnails = {}             # (dataset id, image index) -> JPEG bytes
jobs = {}                   # job id -> progress and result
next_job = itertools.count(1)


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

    The page groups them by folder: "all of a/" and the image folders inside a."""
    scan_models()

    datasets.clear()
    class_datasets.clear()
    dataset_info.clear()
    sound_files.clear()
    direct = {}
    root = Path(ROOT_DATASETS_FOLDER)
    if root.is_dir():
        for folder, subfolders, files in os.walk(root):
            subfolders[:] = sorted(s for s in subfolders if not s.startswith((".", "__")))
            paths = sorted(Path(folder) / f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS)
            if paths:
                direct[Path(folder).relative_to(root).as_posix()] = paths
            # a sound file next to an image, with the same name, is that image's sound
            for f in files:
                if Path(f).suffix.lower() in AUDIO_EXTENSIONS:
                    sound_files[str(Path(folder) / Path(f).stem)] = Path(folder) / f

    # every folder -> the image folders under it (itself included)
    below = {}
    for dataset_id in direct:
        p = Path(dataset_id)
        for folder in [p, *p.parents]:
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
        if dataset_id != ".":
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


def dataset_folder(dataset_id: str) -> Path:
    return Path(ROOT_DATASETS_FOLDER) / dataset_info[dataset_id]["folder"]


def image_name(dataset_id: str, path: Path) -> str:
    """The image's path inside its dataset's folder (subfolder/name.jpg for "all of ...")"""
    return path.relative_to(dataset_folder(dataset_id)).as_posix()


def sample(dataset_id: str, max_images: int) -> list[int]:
    """Indices of the images used: all of them, or max_images picked at random.
    Always the same random order, so a bigger sample contains the smaller one
    (and their embeddings are reused)."""
    return sorted(sample_orders[dataset_id][:max(1, max_images)])


def open_image(path: Path) -> Image.Image:
    try:
        image = Image.open(path)
        image.draft("RGB", (LOAD_SIZE, LOAD_SIZE))     # JPEGs: decode at reduced size, much faster
        image = image.convert("RGB")
        image.thumbnail((LOAD_SIZE * 2, LOAD_SIZE * 2))
        return image
    except Exception as e:
        print(f"can't read {path}: {e}")
        return Image.new("RGB", (LOAD_SIZE, LOAD_SIZE), "gray")


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
    loaded["id"], loaded["embedder"] = None, None   # free the previous one first
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    print(f"loading {model_id}")
    loaded["embedder"] = load_embedder(models[model_id], DEVICE)
    loaded["id"] = model_id
    print(loaded["embedder"].description)


def get_embeddings(model_id: str, dataset_id: str, indices: list[int], job: dict) -> np.ndarray:
    """Vectors [len(indices), D] of these images, computing only the ones not
    done before."""
    done = embeddings.setdefault((model_id, dataset_id), {})
    missing = [i for i in indices if i not in done]
    if missing:
        paths = datasets[dataset_id]
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
                    image = open_image(paths[i])
                    # the images are open anyway: make the map thumbnails now
                    thumbnails.setdefault((dataset_id, i), jpeg(image, THUMBNAIL_SIZE))
                    images.append(image)
                vectors = loaded["embedder"].embed(images).flatten(1).numpy().astype(np.float32)
                done.update(zip(batch, vectors))
    return np.stack([done[i] for i in indices])


def class_names_of(model_id: str) -> list[str] | None:
    """A classifier's class names (from its config.json), None for other models"""
    model = models[model_id]
    if model["kind"] != "classifier":
        return None
    return read_config(model["path"])["class_names"]


def default_prep(model_id: str) -> str:
    """How a model's vectors are best compared, used when the page asks for "default".

      l2           CLIP, DINOv2: they are trained by comparing directions
                   (cosine similarity), the length means little
      none         classifiers: the logits as they are, how big they are is how
                   sure the classifier is. VAE, VAE-GAN: the KL loss already
                   keeps every dimension near mean 0, spread 1. Sparse: 0 means
                   "unit off", keep it so.
      standardize  the other autoencoders: the decoder reads the raw vector, so
                   its length matters (no l2), but a few dimensions can have a
                   much bigger range than the rest and would dominate distances
    """
    model = models[model_id]
    if model["kind"] == "pretrained":
        return "l2"
    if model["kind"] == "classifier":
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


def to_unit_square(xy: np.ndarray) -> np.ndarray:
    """Center in 0..1 x 0..1, keeping the proportions."""
    xy = xy - (xy.min(axis=0) + xy.max(axis=0)) / 2
    extent = np.abs(xy).max() or 1.0
    return xy / (2 * extent) + 0.5


def project(x: np.ndarray, method: str, parameter: float) -> np.ndarray:
    n = len(x)
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
            raw = get_embeddings(p["model"], p["dataset"], indices, job)
            timings["embedding"] = time.perf_counter() - start

            # Classifiers: one dummy "image" per class, appended after the real
            # ones, with ids -1, -2, ... (real images have their index, >= 0)
            paths = datasets[p["dataset"]]
            names = [image_name(p["dataset"], paths[i]) for i in indices]
            class_names = class_names_of(p["model"])
            ids = list(indices)
            vectors = raw
            if class_names:
                vectors = np.concatenate([raw, ANCHOR_LOGIT * np.eye(len(class_names), dtype=np.float32)])
                ids += [-(c + 1) for c in range(len(class_names))]
                names += class_names
            anchor = np.array([i < 0 for i in ids])

            x = prepare(vectors, p["prep"])
            grid = p["projection"] == "clusters"
            cell = None

            if not grid:
                parameter = p["perplexity"] if p["projection"] == "tsne" else p["neighbors"]
                key = (p["model"], p["dataset"], len(indices), p["prep"], p["projection"],
                       parameter if p["projection"] != "pca" else None)
                if key not in projections:
                    job["stage"] = f"projecting ({PROJECTIONS[p['projection']]})"
                    job["progress"] = 1.0
                    start = time.perf_counter()
                    projections[key] = project(x, p["projection"], parameter)
                    timings["projection"] = time.perf_counter() - start
                xy = projections[key]

            if job["cancelled"]:
                raise Cancelled()
            job["stage"] = f"clustering ({CLUSTERINGS[p['clustering']]})"
            start = time.perf_counter()
            # the grid has no map yet: its clusters are always computed on the vectors
            on_vectors = grid or p["cluster_on"] == "embedding"
            labels = cluster_with_anchors(x if on_vectors else xy, anchor, p)
            timings["clustering"] = time.perf_counter() - start
            if grid:
                xy, cell = cluster_grid(x, labels, anchor)

            job["result"] = {
                "model": p["model"],
                "dataset": p["dataset"],
                "projection": p["projection"],
                "prep": p["prep"],                  # what was used ("default" resolved)
                "prep_is_default": p["prep_is_default"],
                "kind": models[p["model"]]["kind"],
                "reconstructs": models[p["model"]]["kind"] == "autoencoder",
                # which images of the dataset (thumbnails etc.), then the class anchors (< 0)
                "indices": ids,
                "total": len(paths),                # images in the dataset
                "names": names,
                # which images have a sound (played when clicked on the map)
                "sounds": [i >= 0 and sound_of(paths[i]) is not None for i in ids],
                "class_names": class_names,         # classifiers only, else None
                "anchor_logit": ANCHOR_LOGIT,
                "xy": np.round(xy, 5).tolist(),
                "cell": cell,                       # grid only: size of a cell, as xy
                "labels": labels.tolist(),
                "dimensions": raw.shape[1],
                # per dimension, for drawing an image's vector relative to the dataset
                "dim_mean": np.round(raw.mean(axis=0), 5).tolist(),
                "dim_std": np.round(raw.std(axis=0), 5).tolist(),
                "seconds": {k: round(v, 2) for k, v in timings.items()},
            }
            job["stage"] = "done"
    except Cancelled:
        job["stage"] = "cancelled"
    except Exception as e:
        import traceback
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
    model_list = [{**{k: m[k] for k in ("id", "label", "kind")}, "default_prep": default_prep(m["id"])}
                  for m in models.values()]
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
        "clustering": body.get("clustering", "kmeans"),
        "cluster_on": body.get("cluster_on", "embedding"),
        "k": int(body.get("k", 8)),
        "eps": float(body.get("eps", 0.5)),
        "min_samples": int(body.get("min_samples", 5)),
        "max_images": int(body.get("max_images", MAX_IMAGES)),
    }
    if p["model"] not in models:
        abort(404, f"no model {p['model']}")
    if p["dataset"] not in datasets:
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
    if dataset_id not in datasets or not 0 <= i < len(datasets[dataset_id]):
        abort(404)
    return datasets[dataset_id][i]


@app.get("/api/thumbnail")
def thumbnail():
    dataset_id, i = request.args["dataset"], int(request.args["i"])
    key = (dataset_id, i)
    if key not in thumbnails:
        thumbnails[key] = jpeg(open_image(dataset_image(dataset_id, i)), THUMBNAIL_SIZE)
    response = send_file(io.BytesIO(thumbnails[key]), mimetype="image/jpeg")
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
    path = dataset_image(request.args["dataset"], int(request.args["i"]))
    return send_file(io.BytesIO(jpeg(open_image(path), PREVIEW_SIZE)), mimetype="image/jpeg")


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
    path = dataset_image(request.args["dataset"], int(request.args["i"]))
    if models.get(model_id, {}).get("kind") != "autoencoder":
        abort(400, "only autoencoders reconstruct")
    with model_lock:
        load_model(model_id)
        image = loaded["embedder"].reconstruct([open_image(path)])[0]
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
        ends = autoencoder.embed([open_image(dataset_image(dataset_id, i)) for i in (a, b)])
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
    if s["kind"] == "classifier":
        s["dataset_path"] = str(Path(ROOT_DATASETS_FOLDER) / s["dataset"])
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
