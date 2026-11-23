"""Train an autoencoder on a folder of images.

No labels needed: every image in IMAGES_FOLDER (and its subfolders) is both
the input and the wanted output. EVALUATION_PERCENT of the images are held
back, the model never trains on them, they show how well it reconstructs
images it has not seen. The model is saved every SAVE_EVERY epochs and after
the last one, with a PDF snapshot of the preview window.

Which kind of autoencoder is set with AUTOENCODER_TYPE (see autoencoder_model.py),
the size of the latent vector with LATENT_DIM, and the options of that kind
with OPTIONS (empty = its defaults, Autoencoder.DEFAULTS of each class).
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from autoencoder_model import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Which autoencoder, one of MLP, CNN, RESNET, VGG, ATTENTION, VAE, SPARSE, VAE_GAN
AUTOENCODER_TYPE = AutoencoderType.VAE_GAN
IMAGE_SIZE = 64             # images are resized and cropped to IMAGE_SIZE x IMAGE_SIZE
LATENT_DIM = 256            # numbers in the latent vector

# The options of each kind, filled in with their defaults (Autoencoder.DEFAULTS
# of each class in autoencoder_model.py). Only the ones of AUTOENCODER_TYPE are
# used; an option a kind doesn't have is an error.
#   base_channels    channels after the first halving, doubled after each next one (max 512)
#   steps            how many times the image is halved (IMAGE_SIZE must be a multiple of 2**steps)
TYPE_OPTIONS = {
    AutoencoderType.MLP: {
        "hidden_sizes": [1024, 512],    # sizes of the layers between the pixels and the latent vector
    },
    AutoencoderType.CNN: {
        "base_channels": 64,
        "steps": 4,
    },
    AutoencoderType.RESNET: {
        "base_channels": 32,
        "steps": 4,
        "residual_blocks": 2,           # residual blocks after every halving / doubling
    },
    AutoencoderType.VGG: {
        "freeze_vgg": True,             # keep the pretrained VGG-16 layers as they are
        "base_channels": 32,            # of the decoder (the encoder is VGG-16, halves 5 times)
    },
    AutoencoderType.ATTENTION: {
        "patch_size": 8,                # the image is cut into patch_size x patch_size patches
        "width": 256,                   # numbers per patch token
        "heads": 4,                     # attention heads (width must be a multiple of heads)
        "layers": 4,                    # transformer layers, in the encoder and in the decoder
    },
    AutoencoderType.VAE: {
        "base_channels": 32,
        "steps": 4,
        "beta": 1.0,                    # weight of the KL loss: higher = smoother latent space, blurrier images
    },
    AutoencoderType.SPARSE: {
        "base_channels": 32,
        "steps": 4,
        "sparsity": 1e-3,               # weight of the L1 penalty: higher = fewer active latent numbers
    },
    AutoencoderType.VAE_GAN: {
        "base_channels": 32,            # of encoder, decoder and discriminator
        "steps": 4,
        "beta": 1.0,                    # weight of the KL loss, as in the VAE
        "feature": 1.0,                 # weight of matching the discriminator's features of the input
        "adversarial": 0.01,            # weight of fooling the discriminator: higher = sharper but less faithful
    },
}
OPTIONS = TYPE_OPTIONS[AUTOENCODER_TYPE]

IMAGES_FOLDER = r"D:\Dropbox\ACADEMIC\GSD\2026_Fall_QA\05_classifiers\datasets\train\B_Baroque"

MODELS_FOLDER = r"D:\ML\QA2026\models"
MODEL_NAME = "baroque_ae_vae_gan_256"

# Load the saved model and keep training it, instead of starting from scratch.
# The saved model must be the same kind, with the same sizes and options.
CONTINUE_TRAINING = False

# Save the model every SAVE_EVERY epochs (and after the last one). With PREVIEW
# each save also writes MODELS_FOLDER/MODEL_NAME/snapshots/epoch_NNN.pdf
SAVE_EVERY = 100

if AUTOENCODER_TYPE == AutoencoderType.ATTENTION:
    LEARNING_RATE = 0.0001
elif AUTOENCODER_TYPE == AutoencoderType.VAE_GAN:
    LEARNING_RATE = 0.0002      # the usual for GANs: two networks pulling against each other
else:
    LEARNING_RATE = 0.003

BATCH_SIZE = 64
NUM_EPOCHS = 50
NUM_WORKERS = 0             # processes loading images in parallel (0 = in the main process)

# After every epoch show a few evaluation images, their reconstruction and
# latent vector, and the loss curves in a live window.
PREVIEW = True
PREVIEW_SAMPLES = 6

# Applied to training images only. Resizing, cropping and converting to a
# tensor are added by the autoencoder.
RANDOM_TRANSFORMATIONS = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
]

EVALUATION_PERCENT = 10.0   # % of the images held back to evaluate the model
SPLIT_SEED = 0              # same seed -> same images held back every run

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFiles(Dataset):
    """Image files, each turned into a tensor by transform. No labels."""

    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def load(self, i: int) -> Image.Image:
        return Image.open(self.paths[i]).convert("RGB")

    def __getitem__(self, i):
        return self.transform(self.load(i))


def find_images(folder: str) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def split(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    order = torch.randperm(len(paths), generator=generator).tolist()
    n_eval = round(len(paths) * EVALUATION_PERCENT / 100)
    return [paths[i] for i in order[n_eval:]], [paths[i] for i in order[:n_eval]]


def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
                      pin_memory=DEVICE.type == "cuda")


def build_autoencoder(model_folder: Path) -> Autoencoder:
    wanted = Autoencoder.create(AUTOENCODER_TYPE, IMAGE_SIZE, LATENT_DIM, **OPTIONS)
    if not (CONTINUE_TRAINING and (model_folder / "config.json").exists()):
        return wanted

    saved = read_config(model_folder)
    for key in ("model_type", "autoencoder_type", "image_size", "latent_dim", "options"):
        if saved.get(key) != wanted.config()[key]:
            raise ValueError(
                f"CONTINUE_TRAINING: the saved model in {model_folder} has {key} {saved.get(key)}, "
                f"but this script asks for {wanted.config()[key]}. Change the settings, pick another "
                f"MODEL_NAME, or set CONTINUE_TRAINING = False to start over (overwrites it).")
    print(f"continuing from {model_folder}")
    return Autoencoder.from_folder(model_folder)


def run_epoch(model: Autoencoder, loader: DataLoader, optimizers=None, after_batch=None) -> dict:
    """One pass over the loader. Trains if optimizers are given (from
    model.make_optimizers), otherwise only evaluates. Returns the average of
    every part of the loss ("loss" = the total). after_batch, if given, is
    called after every batch (keeps the preview window responsive)."""
    training = optimizers is not None
    model.train(training)
    totals, count = {}, 0
    with torch.set_grad_enabled(training):
        for x in tqdm(loader, desc="train" if training else "eval", leave=False):
            x = x.to(DEVICE)
            if training:
                # the model does its own step: most kinds have one loss and one
                # optimizer, the VAE-GAN alternates with its discriminator
                parts = model.train_step(x, optimizers)
            else:
                loss, parts = model.loss(x)
                parts = {"loss": loss.item(), **parts}
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + value * len(x)
            count += len(x)
            if after_batch:
                after_batch()
    return {name: total / max(count, 1) for name, total in totals.items()}


def describe(averages: dict) -> str:
    return " ".join(f"{name} {value:.4f}" for name, value in averages.items())


def main():
    print(f"device: {DEVICE}")
    model_folder = Path(MODELS_FOLDER) / MODEL_NAME

    paths = find_images(IMAGES_FOLDER)
    if not paths:
        raise ValueError(f"no images in {IMAGES_FOLDER}")
    train_paths, eval_paths = split(paths)
    print(f"{len(train_paths)} training images, {len(eval_paths)} evaluation images")

    autoencoder = build_autoencoder(model_folder).to(DEVICE)
    print(autoencoder)

    train_images = ImageFiles(train_paths, autoencoder.preprocess(RANDOM_TRANSFORMATIONS))
    eval_images = ImageFiles(eval_paths, autoencoder.preprocess())
    train_loader = make_loader(train_images, shuffle=True)
    eval_loader = make_loader(eval_images, shuffle=False)

    optimizers = autoencoder.make_optimizers(LEARNING_RATE)

    # history.json keeps the losses of every epoch, across continued runs
    history_file = model_folder / "history.json"
    history = json.loads(history_file.read_text()) if CONTINUE_TRAINING and history_file.exists() else []

    preview = None
    if PREVIEW:
        from autoencoder_training_preview import TrainingPreview
        # show evaluation images if there are any, so they are unseen ones
        preview = TrainingPreview(autoencoder, eval_images if len(eval_images) else
                                  ImageFiles(train_paths, autoencoder.preprocess()),
                                  PREVIEW_SAMPLES, DEVICE)
    after_batch = preview.keep_responsive if preview else None

    for epoch in range(1, NUM_EPOCHS + 1):
        train = run_epoch(autoencoder, train_loader, optimizers, after_batch)
        line = f"epoch {epoch:3d}/{NUM_EPOCHS}  train {describe(train)}"
        evaluation = None
        if len(eval_images):
            evaluation = run_epoch(autoencoder, eval_loader, after_batch=after_batch)
            line += f"  |  eval {describe(evaluation)}"

        history.append({"train": train, "eval": evaluation})
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history_file.write_text(json.dumps(history, indent=2))

        # epochs are counted across continued runs, so snapshots don't overwrite each other
        total_epochs = len(history)
        if preview:
            preview.update(total_epochs, history)

        if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
            autoencoder.save(model_folder)
            line += "  -> saved"
            if preview:
                preview.save_pdf(model_folder / "snapshots" / f"epoch_{total_epochs:03d}.pdf")
        print(line)

    print(f"model in {model_folder}")
    if preview:
        preview.keep_open()


if __name__ == "__main__":
    # The guard is required on Windows: each DataLoader worker process imports
    # this file again, and must not start its own training.
    main()
