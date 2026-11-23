"""Convert a folder of images offline, without the web app.

The same conversions the Dataset tab offers (dataset_tools.CONVERSIONS):
grayscale, canny, depth, depth_edges. Every image of the source folder is
converted and written into the target folder under the same name.

    uv run --project . --env-file .env python convert_dataset.py \
        ../../05_classifiers/datasets/frames \
        ../../05_classifiers/datasets/frames_edges \
        canny

Conversion parameters are passed as name=value after the conversion:

    ... canny blur=2 low=100 high=200
    ... depth depth_model=small
"""
import sys
import time
from pathlib import Path

# the app's package folder starts with a digit, so it cannot be imported as a
# package: put it on the path and import the module itself
sys.path.insert(0, str(Path(__file__).parent / "06_autoencoders"))

import torch
from PIL import Image

import dataset_tools

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH = 8


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse(args: list[str], conversion: str) -> dict:
    """name=value pairs -> params, cast to the type of the conversion's default"""
    defaults = dataset_tools.CONVERSIONS[conversion]["params"]
    params = {}
    for arg in args:
        name, _, value = arg.partition("=")
        if name not in defaults:
            raise SystemExit(f"{conversion} has no parameter {name} "
                             f"(it has {', '.join(defaults) or 'none'})")
        default = defaults[name]
        if isinstance(default, bool):
            params[name] = value.lower() in ("1", "true", "yes")
        elif isinstance(default, (int, float)):
            params[name] = type(default)(value)
        else:
            params[name] = value
    return params


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    source, target, conversion = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    if conversion not in dataset_tools.CONVERSIONS:
        raise SystemExit(f"unknown conversion {conversion} "
                         f"(one of {', '.join(dataset_tools.CONVERSIONS)})")
    params = parse(sys.argv[4:], conversion)

    paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"no images in {source}")
    target.mkdir(parents=True, exist_ok=True)

    dev = device()
    # only the depth conversions run a network; grayscale and canny are OpenCV
    on = dev if conversion.startswith("depth") else "the cpu (opencv)"
    print(f"{len(paths)} images, {conversion} {params or ''} on {on}")
    print(f"{source} -> {target}")

    started = time.time()
    for start in range(0, len(paths), BATCH):
        batch = paths[start:start + BATCH]
        images = [Image.open(p).convert("RGB") for p in batch]
        for path, out in zip(batch, dataset_tools.convert(images, conversion, params, dev)):
            # keep the folder structure of the source, write everything as png
            name = path.relative_to(source).with_suffix(".png")
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            out.save(target / name)
        done = start + len(batch)
        per = (time.time() - started) / done
        print(f"\r{done}/{len(paths)}  {per:.2f}s per image, "
              f"{(len(paths) - done) * per / 60:.1f} min left   ", end="", flush=True)
    print(f"\n{len(paths)} images in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
