"""Train an autoencoder offline, without the web app.

Drives the app's own training_engine.TrainingRun with exactly the defaults the
Training tab shows, so the model it writes is the same one the tab would write
and the Embedding Map picks it up straight away. Used instead of the tab
because the server keeps a single global training slot: starting one there
would clobber whatever run is already being watched in the browser.

    uv run --project . --env-file .env python train_dataset.py \
        topography/cell_depth topo_cell_depth_cnn

Settings are passed as name=value after the model name, and override the
defaults below:

    ... topography/cell_depth topo_cell_cnn epochs=80 latent_dim=256
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "06_autoencoders"))

import torch

import training_engine
from training_engine import TrainingRun

ROOT_DATASETS = Path("/Users/ruyiyang/Documents/courseware/2026_Fall/"
                     "SCI6492_Quantitative Aesthetics/sci6492_code/05_classifiers/datasets")
ROOT_MODELS = Path("/Users/ruyiyang/Documents/courseware/2026_Fall/"
                   "SCI6492_Quantitative Aesthetics/sci6492_code/05_classifiers/models")

# The Training tab's defaults (web/app.html), autoencoder side
DEFAULTS = {
    "epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 64,
    "eval_percent": 10,
    "save_every": 10,
    "type": "cnn",
    "image_size": 64,
    "latent_dim": 128,
    # cnn/resnet options. steps is how many times the image is halved, and it
    # sets the bottleneck with image_size: the encoder flattens
    # channels[-1] x (image_size / 2**steps)^2 features down to latent_dim.
    # Raising image_size without raising steps makes that compression tighter,
    # not looser.
    "base_channels": 32,
    "steps": 4,
    # augmentation block
    "flip": 1,
    "rotation": 0,
    "jitter": 0,
    "crop": 0.6,
}


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    dataset, name, rest = sys.argv[1], sys.argv[2], sys.argv[3:]

    settings = dict(DEFAULTS)
    for arg in rest:
        key, _, value = arg.partition("=")
        if key not in DEFAULTS:
            raise SystemExit(f"unknown setting {key} (have {', '.join(DEFAULTS)})")
        settings[key] = type(DEFAULTS[key])(value)

    folder = ROOT_DATASETS / dataset
    paths = training_engine.find_images(folder)
    if not paths:
        raise SystemExit(f"no images in {folder}")

    s = {
        "kind": "autoencoder",
        "name": name,
        "continue": False,
        "dataset": dataset,
        "dataset_path": str(folder),
        "image_paths": [str(p) for p in paths],
        "epochs": settings["epochs"],
        "learning_rate": settings["learning_rate"],
        "batch_size": settings["batch_size"],
        "eval_percent": settings["eval_percent"],
        "save_every": settings["save_every"],
        "augment": {"enabled": True, "flip": bool(settings["flip"]),
                    "rotation": settings["rotation"], "jitter": settings["jitter"],
                    "crop": settings["crop"]},
        "autoencoder": {"type": settings["type"], "image_size": settings["image_size"],
                        "latent_dim": settings["latent_dim"],
                        "options": {"base_channels": settings["base_channels"],
                                    "steps": settings["steps"]}},
    }

    out = ROOT_MODELS / name
    dev = device()
    print(f"{len(paths)} images from {folder}")
    print(f"{settings['type']} autoencoder, {settings['image_size']}px, "
          f"{settings['latent_dim']} latent, {settings['epochs']} epochs on {dev}")
    print(f"-> {out}\n")

    run = TrainingRun(s, out, dev)
    run.start()

    # TrainingRun.write_log already prints every epoch, so just wait here
    while run.active or run.state == "starting":
        time.sleep(2)

    print(f"\nstate: {run.state}  {run.message}")
    if run.state == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
