"""Train a classifier on a folder of images sorted into one subfolder per class.

    CLASSES_FOLDERS/
        A_Expressionism/  img1.jpg img2.jpg ...
        B_Baroque/        ...

The subfolder names become the class names. EVALUATION_PERCENT of the images
are held back: the model never trains on them, they only measure how well it
does on images it has not seen. The model is saved every SAVE_EVERY epochs and
after the last one, with a PDF snapshot of the preview window.

Two ways to train:

  FREEZE_BACKBONE = True    only the head learns. If RANDOM_TRANSFORMATIONS is
                            empty every image always produces the same features,
                            so they are computed once and cached, and each epoch
                            takes a fraction of a second.
  FREEZE_BACKBONE = False   the whole network learns (fine-tuning). The backbone
                            uses a smaller learning rate so the pretrained
                            features are adjusted gently instead of destroyed.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from classifier_model import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Which backbone, e.g.
#ARCHITECTURE = ClassifierID.imagenet(ImageNetArch.RESNET50)
#ARCHITECTURE = ClassifierID.imagenet(ImageNetArch.CONVNEXT_TINY)
#ARCHITECTURE = ClassifierID.imagenet(ImageNetArch.VIT_B_32)
#ARCHITECTURE = ClassifierID.imagenet(ImageNetArch.ALEXNET)
#ARCHITECTURE = ClassifierID.dinov2(DINOV2Arch.VITB14)
ARCHITECTURE = ClassifierID.clip(CLIPArch.VITB32, CLIPPretrained.DATACOMP_XL_S13B_B90K)
#ARCHITECTURE = ClassifierID.clip(CLIPArch.VITB32, CLIPPretrained.OPENAI)



CLASSES_FOLDERS = str(Path(__file__).resolve().parents[2] / "datasets" / "train_art_quick")

MODELS_FOLDER = str(Path(__file__).resolve().parents[2] / "models")
MODEL_NAME = "art_classifier_quick_CLIP_DATACOMP"


# Load the saved model and keep training it, instead of starting from scratch.
# The saved model must have the same backbone, head and classes.
CONTINUE_TRAINING = True

# Save the model every SAVE_EVERY epochs (and after the last one). With PREVIEW
# each save also writes MODELS_FOLDER/MODEL_NAME/snapshots/epoch_NNN.pdf
SAVE_EVERY = 10

LEARNING_RATE = 0.0001

# After every epoch show a few random training images (original, what the
# model receives, its scores), the loss curves and the evaluation images the
# model gets most wrong in a live window.
PREVIEW = True
PREVIEW_SAMPLES = 5
PREVIEW_MISCLASSIFIED = 3   # per class


# Applied to training images only, a different random version every epoch, so
# the model can't memorize exact pixels. Resizing, cropping to the backbone's
# image size and normalizing are added by the classifier. Leave the list empty
# to train without them (and, with a frozen backbone, cache the features).
RANDOM_TRANSFORMATIONS = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
]

EVALUATION_PERCENT = 20.0   # % of the images held back to evaluate the model
SPLIT_SEED = 0              # same seed -> same images held back every run


BACKBONE_LEARNING_RATE_FACTOR = 0.1     # fine-tuning only: backbone lr = LEARNING_RATE * this

BATCH_SIZE = 32

NUM_EPOCHS = 50

NUM_WORKERS = 0            # processes loading images in parallel (0 = in the main process)

# leave the backbone frozen during training (don;t try to fine-tune it)
FREEZE_BACKBONE = True

# Head configuration
HEAD = "linear"             # "linear" or "mlp"
HIDDEN_SIZE = 512           # mlp only: size of the hidden layer
DROPOUT = 0.2               # mlp only: fraction of hidden values zeroed while training


class ImageSubset(Dataset):
    """Some of the images of an ImageFolder, with their own transform (the
    training and evaluation images are preprocessed differently)."""

    def __init__(self, images: datasets.ImageFolder, indices: list[int], transform):
        self.images = images
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.images.samples[self.indices[i]]
        image = self.images.loader(path)
        return self.transform(image), label


def split_indices(n: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    order = torch.randperm(n, generator=generator).tolist()
    n_eval = round(n * EVALUATION_PERCENT / 100)
    return order[n_eval:], order[:n_eval]


def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
                      pin_memory=DEVICE.type == "cuda")


def build_classifier(class_names: list[str], model_folder: Path) -> Classifier:
    if CONTINUE_TRAINING and (model_folder / "config.json").exists():
        classifier = Classifier.from_folder(model_folder)
        expected = {
            "architecture": (ARCHITECTURE, classifier.classifier_id),
            "class names": (class_names, classifier.class_names),
            "head": (HEAD, classifier.head_type),
        }
        if HEAD == "mlp":
            expected["hidden size"] = (HIDDEN_SIZE, classifier.hidden_size)
        for name, (wanted, saved) in expected.items():
            if wanted != saved:
                raise ValueError(
                    f"CONTINUE_TRAINING: the saved model in {model_folder} has {name} {saved}, "
                    f"but this script asks for {wanted}. Change the settings, pick another "
                    f"MODEL_NAME, or set CONTINUE_TRAINING = False to start over (overwrites it).")
        classifier.dropout = DROPOUT
        classifier.set_freeze_backbone(FREEZE_BACKBONE)
        print(f"continuing from {model_folder}")
        return classifier

    return Classifier(ARCHITECTURE, class_names, head=HEAD, hidden_size=HIDDEN_SIZE,
                      dropout=DROPOUT, freeze_backbone=FREEZE_BACKBONE)


@torch.no_grad()
def compute_features(classifier: Classifier, loader: DataLoader) -> TensorDataset:
    classifier.eval()
    all_features, all_labels = [], []
    for images, labels in tqdm(loader, desc="computing features"):
        all_features.append(classifier.features(images.to(DEVICE)).cpu())
        all_labels.append(labels)
    return TensorDataset(torch.cat(all_features), torch.cat(all_labels))


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn, optimizer=None) -> tuple[float, float, torch.Tensor]:
    """One pass over the loader. Trains if an optimizer is given, otherwise only
    evaluates. Returns (average loss, accuracy, logits). The logits of every
    image are only collected when evaluating (None when training)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    all_logits = []
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                all_logits.append(logits.cpu())
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            count += len(labels)
    logits = torch.cat(all_logits) if all_logits else None
    return total_loss / max(count, 1), correct / max(count, 1), logits


def main():
    print(f"device: {DEVICE}")
    model_folder = Path(MODELS_FOLDER) / MODEL_NAME

    images = datasets.ImageFolder(CLASSES_FOLDERS)
    class_names = images.classes
    train_indices, eval_indices = split_indices(len(images))
    print(f"{len(class_names)} classes: {class_names}")
    print(f"{len(train_indices)} training images, {len(eval_indices)} evaluation images")

    classifier = build_classifier(class_names, model_folder).to(DEVICE)
    print(classifier)

    train_images = ImageSubset(images, train_indices, classifier.preprocess(RANDOM_TRANSFORMATIONS))
    eval_images = ImageSubset(images, eval_indices, classifier.preprocess())

    if FREEZE_BACKBONE and not RANDOM_TRANSFORMATIONS:
        # The backbone never changes and neither do the images, so run every
        # image through the backbone once and train the head on the stored
        # features. The "model" of each epoch is then just the head.
        train_loader = DataLoader(compute_features(classifier, make_loader(train_images, False)),
                                  batch_size=BATCH_SIZE, shuffle=True)
        eval_loader = DataLoader(compute_features(classifier, make_loader(eval_images, False)),
                                 batch_size=BATCH_SIZE)
        model = classifier.head
    else:
        train_loader = make_loader(train_images, shuffle=True)
        eval_loader = make_loader(eval_images, shuffle=False)
        model = classifier

    parameter_groups = [{"params": classifier.head.parameters(), "lr": LEARNING_RATE}]
    if not FREEZE_BACKBONE:
        parameter_groups.append({"params": classifier.backbone.parameters(),
                                 "lr": LEARNING_RATE * BACKBONE_LEARNING_RATE_FACTOR})
    optimizer = torch.optim.AdamW(parameter_groups)
    loss_fn = nn.CrossEntropyLoss()

    # history.json keeps loss and accuracy of every epoch, across continued runs
    history_file = model_folder / "history.json"
    history = json.loads(history_file.read_text()) if CONTINUE_TRAINING and history_file.exists() else []
    has_eval = len(eval_indices) > 0

    preview = None
    if PREVIEW:
        from classifier_training_preview import TrainingPreview
        preview = TrainingPreview(classifier, train_images, eval_images, PREVIEW_SAMPLES,
                                  PREVIEW_MISCLASSIFIED, DEVICE)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy, _ = run_epoch(model, train_loader, loss_fn, optimizer)
        line = f"epoch {epoch:3d}/{NUM_EPOCHS}  train loss {train_loss:.4f} acc {train_accuracy:6.1%}"

        eval_loss, eval_accuracy, eval_logits = None, None, None
        if has_eval:
            eval_loss, eval_accuracy, eval_logits = run_epoch(model, eval_loader, loss_fn)
            line += f"  |  eval loss {eval_loss:.4f} acc {eval_accuracy:6.1%}"

        history.append({"train_loss": train_loss, "train_accuracy": train_accuracy,
                        "eval_loss": eval_loss, "eval_accuracy": eval_accuracy})
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history_file.write_text(json.dumps(history, indent=2))

        # epochs are counted across continued runs, so snapshots don't overwrite each other
        total_epochs = len(history)
        if preview:
            preview.update(total_epochs, history, eval_logits)

        if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
            classifier.save(model_folder)
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
