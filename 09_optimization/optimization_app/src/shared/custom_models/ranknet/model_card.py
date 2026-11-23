"""A RankNet as one self-describing JSON file (a "model card"): what it is,
how it was trained, and its weights. Every save of a RankNet folder also
writes one next to config.json (<folder name>.ranknet.json), to copy, share or
archive the model as a single file. A card found anywhere under the models
folder loads like a folder.

No torch here (the weights are encoded with the standard library), so a
program can read and check a card without it.

    { "schema":  "diffdraw.taste.model/1",       # DiffDrawing's taste card: it reads these
      "format":  {"name": "ranknet-card", "version": "1.1", "min_reader": "1.0"},
      "producer": {"app": "substack custom_models", "ranknet_version": "1.0"},
      "created", "model_ids", "normalize", "embed_dim", "hidden_dim", "dropout",
      "head": "mlp-128-64-1",
      "layers": [{"name": "net.0", "kind": "linear", "in": 896, "out": 128}, ...],
      "preprocess": {"resize": 224, "crop": "center", "normalize": "per backbone"},
      "trained_on", "training": {...},           # history summary, pairs, agreement...
      "weights": {"net.0.weight": {"shape", "dtype": "float32", "encoding", "data"}, ...} }

Versioning: "format.version" is MAJOR.MINOR. A reader accepts a card whose
major version it knows; a higher minor only adds fields it may ignore. A
breaking change bumps the major and sets "min_reader" to it, so an old reader
refuses the card instead of misreading it.

Weights: float32 little-endian, zlib-compressed, base64 ("encoding":
"f32le+zlib+base64"): about 4 times smaller than a JSON list of numbers, and
exact.
"""

import base64
import json
import sys
import time
import zlib
from array import array
from pathlib import Path

SCHEMA = "diffdraw.taste.model/1"
FORMAT_NAME = "ranknet-card"
FORMAT_VERSION = "1.1"
READS_MAJOR = 1                 # the major format version this code reads
ENCODING = "f32le+zlib+base64"
SUFFIX = ".ranknet.json"


def card_path(folder) -> Path:
    folder = Path(folder)
    return folder / f"{folder.name}{SUFFIX}"


def encode_array(shape: list[int], values) -> dict:
    buffer = array("f", values)
    if sys.byteorder != "little":
        buffer.byteswap()
    return {"shape": list(shape), "dtype": "float32", "encoding": ENCODING,
            "data": base64.b64encode(zlib.compress(buffer.tobytes(), 6)).decode("ascii")}


def decode_array(entry: dict) -> tuple[list[int], array]:
    if entry.get("dtype", "float32") != "float32" or entry.get("encoding", ENCODING) != ENCODING:
        raise ValueError(f"unsupported weights: {entry.get('dtype')} {entry.get('encoding')}")
    buffer = array("f")
    buffer.frombytes(zlib.decompress(base64.b64decode(entry["data"])))
    if sys.byteorder != "little":
        buffer.byteswap()
    shape = [int(d) for d in entry["shape"]]
    expected = 1
    for d in shape:
        expected *= d
    if len(buffer) != expected:
        raise ValueError(f"a tensor has {len(buffer)} numbers, its shape {shape} needs {expected}")
    return shape, buffer


def check(card: dict) -> dict:
    """Raise if this code can't read the card. Cards without "format" are
    DiffDrawing's (version 1.0)."""
    if not isinstance(card, dict) or card.get("schema") != SCHEMA:
        raise ValueError(f"not a RankNet card (schema {card.get('schema') if isinstance(card, dict) else None!r})")
    form = card.get("format", {"version": "1.0", "min_reader": "1.0"})
    major = int(str(form.get("version", "1.0")).split(".")[0])
    min_reader = int(str(form.get("min_reader", "1.0")).split(".")[0])
    if major != READS_MAJOR or min_reader > READS_MAJOR:
        raise ValueError(f"RankNet card format {form.get('version')} needs a newer reader "
                         f"(this one reads {READS_MAJOR}.x)")
    if not card.get("model_ids") or not isinstance(card.get("weights"), dict) or not card["weights"]:
        raise ValueError("RankNet card without model_ids or weights")
    return card


def make_card(config: dict, state: dict, training: dict = None) -> dict:
    """config: RankNet.config(); state: the head's state_dict as
    {name: (shape, flat list of floats)}"""
    layers = []
    for name, (shape, _) in state.items():
        if name.endswith(".weight"):
            layers.append({"name": name[:-len(".weight")], "kind": "linear", "in": shape[1], "out": shape[0]})
    return {
        "schema": SCHEMA,
        "format": {"name": FORMAT_NAME, "version": FORMAT_VERSION, "min_reader": f"{READS_MAJOR}.0"},
        "producer": {"app": "substack custom_models", "ranknet_version": config.get("version")},
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_ids": config["model_ids"],
        "normalize": config["normalize"],
        "embed_dim": config["embed_dim"],
        "hidden_dim": config["hidden_dim"],
        "dropout": config.get("dropout"),
        "head": f"mlp-{config['hidden_dim']}-64-1",
        "layers": layers,
        "preprocess": {"resize": 224, "crop": "center", "normalize": "per backbone (CLIP or ImageNet statistics)"},
        "trained_on": config.get("trained_on", {}),
        "training": training or {},
        "weights": {name: encode_array(shape, values) for name, (shape, values) in state.items()},
    }


def write_card(path, card: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(card), encoding="utf-8")
    tmp.replace(path)


def read_card(path) -> dict:
    return check(json.loads(Path(path).read_text(encoding="utf-8")))
