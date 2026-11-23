"""Classifier map: a web page that places images on a map according to how a
trained classifier scores them.

Two projections (computed in the browser, web/classifier_web_app.js):

  biaxial   x = score of one class, y = score of another (picked in the page)
  radial    each of the C classes gets an axis, all at equal angles (3 classes:
            120 degrees apart). An image with scores s_1 ... s_C is placed at

                center + s_1 * axis_1 + s_2 * axis_2 + ... + s_C * axis_C

            so an image strong in one class sits near the tip of its axis, and
            an image that is not strong in any of them (or equally strong in
            all) stays near the center.

Scores are softmax (add up to 1, classes compete) or sigmoid (independent).

Images get into the map by uploading files, or by drawing on the small canvas
and pressing "embed". The drawing is also projected live (every time the mouse
is released) without being added.

The server keeps the images in memory and returns, for every image, the raw
scores (logits) of the selected model. The browser turns them into softmax or
sigmoid and positions; so switching projection or classes doesn't need the
server, only switching model does (every image is classified again).

Run it and open http://127.0.0.1:5001
"""

import base64
import io
import itertools
import threading
from pathlib import Path

import torch
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image

from classifier_model import Classifier

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Every subfolder with a config.json is a model that can be picked in the page
ROOT_MODEL_FOLDER = str(Path(__file__).resolve().parents[2] / "models")
DEFAULT_MODEL = "art_classifier_quick_CLIP_DATACOMP"

# Images already on the map when the page opens (None = start empty)
INITIAL_IMAGES_FOLDER = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STORED_SIZE = 512           # images are kept at most this big (longest side)
THUMBNAIL_SIZE = 160        # size sent to the browser for the map
BATCH_SIZE = 32
PORT = 5001

app = Flask(__name__)

# One model is loaded at a time. The lock stops two requests from using or
# swapping it at the same time (flask answers requests in parallel threads).
lock = threading.Lock()
state = {"name": None, "classifier": None}

# id -> {"name": file name, "image": PIL image, "thumbnail": JPEG bytes}
images = {}
# (model name, image id) -> list of logits, so switching back to a model is instant
logits_cache = {}
next_id = itertools.count(1)
next_drawing = itertools.count(1)


# ---------------------------------------------------------------------------
# Models and images
# ---------------------------------------------------------------------------

def list_models() -> list[str]:
    root = Path(ROOT_MODEL_FOLDER)
    return sorted(p.name for p in root.iterdir() if (p / "config.json").exists())


def load_model(name: str) -> None:
    """Called with the lock held."""
    if state["name"] == name:
        return
    if name not in list_models():
        abort(404, f"no model '{name}' in {ROOT_MODEL_FOLDER}")
    state["classifier"] = None          # free the previous one first
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    print(f"loading {name}")
    state["classifier"] = Classifier.from_folder(Path(ROOT_MODEL_FOLDER) / name).to(DEVICE).eval()
    state["name"] = name
    print(state["classifier"])


def classify(pil_images: list[Image.Image]) -> list[list[float]]:
    """Logits of the current model for each image. Called with the lock held."""
    out = []
    for start in range(0, len(pil_images), BATCH_SIZE):
        out += state["classifier"].predict(pil_images[start:start + BATCH_SIZE], "logits").tolist()
    return out


def logits_for(ids: list[int]) -> dict[int, list[float]]:
    """Logits of the current model for these stored images, computing only the
    ones not in the cache. Called with the lock held."""
    missing = [i for i in ids if (state["name"], i) not in logits_cache]
    for i, logits in zip(missing, classify([images[i]["image"] for i in missing])):
        logits_cache[(state["name"], i)] = logits
    return {i: logits_cache[(state["name"], i)] for i in ids}


def shrink(image: Image.Image, size: int) -> Image.Image:
    image = image.copy()
    image.thumbnail((size, size))
    return image


def add_image(image: Image.Image, name: str) -> int:
    image = shrink(image.convert("RGB"), STORED_SIZE)
    buffer = io.BytesIO()
    shrink(image, THUMBNAIL_SIZE).save(buffer, "JPEG", quality=85)
    image_id = next(next_id)
    images[image_id] = {"name": name, "image": image, "thumbnail": buffer.getvalue()}
    return image_id


def describe(ids: list[int]) -> list[dict]:
    """What the browser needs to place these images. Called with the lock held."""
    logits = logits_for(ids)
    return [{"id": i, "name": images[i]["name"], "thumbnail": f"/api/thumbnail/{i}", "logits": logits[i]}
            for i in ids]


def decode_data_url(data_url: str) -> Image.Image:
    """'data:image/png;base64,....' (what canvas.toDataURL gives) -> PIL image"""
    encoded = data_url.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    if image.mode in ("RGBA", "LA", "P"):
        # transparent pixels become white instead of black
        background = Image.new("RGB", image.size, "white")
        background.paste(image.convert("RGBA"), mask=image.convert("RGBA").split()[-1])
        return background
    return image.convert("RGB")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(WEB, "classifier_web_app.html")


@app.route("/<path:name>")
def static_file(name):
    return send_from_directory(WEB, name)


@app.get("/api/models")
def models():
    names = list_models()
    default = DEFAULT_MODEL if DEFAULT_MODEL in names else (names[0] if names else None)
    return jsonify({"models": names, "default": default})


@app.post("/api/model")
def select_model():
    """Switch model. Returns its classes and every image's logits for it."""
    name = request.get_json()["name"]
    with lock:
        load_model(name)
        return jsonify({"name": name,
                        "class_names": state["classifier"].class_names,
                        "images": describe(list(images))})


@app.post("/api/images")
def upload_images():
    """Files from an <input type=file> or dropped on the page."""
    ids = []
    with lock:
        for file in request.files.getlist("files"):
            try:
                ids.append(add_image(Image.open(file.stream), file.filename))
            except Exception as e:      # not an image, skip it
                print(f"skipping {file.filename}: {e}")
        return jsonify({"images": describe(ids)})


@app.post("/api/classify")
def classify_drawing():
    """Logits for the canvas drawing, without adding it to the map."""
    image = decode_data_url(request.get_json()["image"])
    with lock:
        return jsonify({"logits": classify([image])[0]})


@app.post("/api/embed")
def embed_drawing():
    """Add the canvas drawing to the map as a new image."""
    image = decode_data_url(request.get_json()["image"])
    with lock:
        image_id = add_image(image, f"drawing {next(next_drawing)}")
        return jsonify({"images": describe([image_id])})


@app.post("/api/clear")
def clear_images():
    with lock:
        images.clear()
        logits_cache.clear()
    return jsonify({"images": []})


@app.get("/api/thumbnail/<int:image_id>")
def thumbnail(image_id: int):
    if image_id not in images:
        abort(404)
    return send_file(io.BytesIO(images[image_id]["thumbnail"]), mimetype="image/jpeg")


if __name__ == "__main__":
    if INITIAL_IMAGES_FOLDER:
        for path in sorted(Path(INITIAL_IMAGES_FOLDER).iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                add_image(Image.open(path), path.name)
        print(f"{len(images)} images from {INITIAL_IMAGES_FOLDER}")
    # no debug reloader: it would start the app (and load the model) twice
    app.run(port=PORT, threaded=True)
