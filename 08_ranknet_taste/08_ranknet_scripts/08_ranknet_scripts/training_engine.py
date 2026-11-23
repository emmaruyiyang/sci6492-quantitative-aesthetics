"""Training runs started from the web page (model_central_app.py).

The same training as autoencoder_training.py and 05_b_classifiers/
classifier_training.py, but driven by a dictionary of settings instead of
constants at the top of a script, running in a background thread, and
reporting as it goes: status() returns the progress, the losses of every
epoch and a preview (a few images with their reconstruction or class scores)
for the page to draw. A stop can be requested at any moment (it is checked
after every batch), with or without saving the model first.

Settings (what the page sends):

  kind            "autoencoder", "classifier" or "ranknet"
  name            folder name under the models folder
  continue        load the saved model in that folder and keep training it
  dataset_path    autoencoder: every image under it (or image_paths, a list of
                  files, if given); classifier: one subfolder per class;
                  ranknet: image_paths and image_names (their names in the
                  ranking file) and ranking_path, a ranking file
                  (ranknet/rankings.py)
  epochs, learning_rate, batch_size, eval_percent, save_every
  augment         {"enabled": bool, "flip": bool, "rotation": degrees, "jitter": strength,
                   "crop": min scale}, random changes to the training images (both kinds)
  autoencoder     {"type", "image_size", "latent_dim", "options"}
  classifier      {"backbone": {"backbone", "arch", "pretrained"}, "head", "hidden_size",
                   "dropout", "freeze_backbone", "backbone_lr_factor"}
  ranknet         {"backbones": [ModelID strings], "hidden_dim", "dropout", "weight_decay"}
                  (the pairs are split into training / evaluation, not the
                  images; no augmentation: every image is embedded once)

With "continue" the model's architecture comes from its saved config.json and
the architecture settings above are ignored; the training settings are used.
"""

import base64
import io
import json
import threading
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

from autoencoder.autoencoder_model import Autoencoder, AutoencoderType, folder_model_type
from classifier.classifier_model import Classifier, ClassifierID
from ranknet import rankings
from ranknet.ranknet_model import RankNet, pair_accuracy, ranknet_loss
from model_types import ModelID

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PREVIEW_SAMPLES = 6
PREVIEW_SIZE = 128          # classifier preview images (longest side)
AUGMENTED_VIEWS = 4         # classifier preview: augmented versions of each sample
AUGMENTED_SIZE = 64
MISCLASSIFIED_PER_CLASS = 3
RANKED_SHOWN = 8            # ranknet preview: best and worst images
DISAGREEMENTS_SHOWN = 6     # ranknet preview: evaluation pairs the model gets most wrong
EMBED_BATCH = 32
SPLIT_SEED = 0              # same seed -> same images held back every run
LOG_LINES = 200


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def find_images(folder) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def split_indices(n: int, eval_percent: float) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    order = torch.randperm(n, generator=generator).tolist()
    n_eval = round(n * eval_percent / 100)
    return order[n_eval:], order[:n_eval]


def augmentations(augment: dict, crop_size: int) -> list:
    """The random transformations, applied to training images only (none when
    augment["enabled"] is false)"""
    a, t = augment or {}, []
    if not a.get("enabled", True):
        return t
    if a.get("flip"):
        t.append(transforms.RandomHorizontalFlip())
    if a.get("rotation"):
        t.append(transforms.RandomRotation(float(a["rotation"])))
    if a.get("jitter"):
        j = float(a["jitter"])
        t.append(transforms.ColorJitter(brightness=j, contrast=j, saturation=j, hue=min(0.5, j / 4)))
    if float(a.get("crop", 1.0)) < 1.0:
        t.append(transforms.RandomResizedCrop(crop_size, scale=(float(a["crop"]), 1.0)))
    return t


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


class ImageSubset(Dataset):
    """Some of the images of an ImageFolder (with labels), with their own transform"""

    def __init__(self, images: datasets.ImageFolder, indices: list[int], transform):
        self.images = images
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def load(self, i: int) -> tuple[Image.Image, int]:
        path, label = self.images.samples[self.indices[i]]
        return self.images.loader(path).convert("RGB"), label

    def __getitem__(self, i):
        image, label = self.load(i)
        return self.transform(image), label


def data_url(image: Image.Image, size: int = None, fmt: str = "PNG") -> str:
    if size:
        image = image.copy()
        image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, fmt, **({"quality": 85} if fmt == "JPEG" else {}))
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buffer.getvalue()).decode()


def spread(n: int, count: int) -> list[int]:
    """count indices evenly spread over range(n)"""
    count = min(count, n)
    return [k * n // count for k in range(count)]


# ---------------------------------------------------------------------------
# A training run
# ---------------------------------------------------------------------------

class Stopped(Exception):
    pass


class TrainingRun:

    def __init__(self, settings: dict, folder: Path, device: torch.device, on_saved=None):
        """on_saved(folder) is called (in the training thread) after every save."""
        self.settings = settings
        self.folder = Path(folder)
        self.device = device
        self.on_saved = on_saved
        self.state = "starting"         # starting, running, stopping, done, stopped, error
        self.stage = ""
        self.message = ""
        self.epoch = 0                  # epochs completed in this run
        self.total_epochs = int(settings["epochs"])
        self.batch, self.batches = 0, 0
        self.history = []               # every epoch, including earlier runs when continuing
        self.preview, self.preview_version = None, 0
        self.log = []
        self.stop_request = None        # None, "save" or "discard"
        self.saved = False
        self.started = time.time()
        self.epoch_seconds = []
        self.description = ""
        self.model = None               # the Autoencoder or Classifier being trained
        self.thread = threading.Thread(target=self.run, daemon=True)

    # --- control ---------------------------------------------------------------

    def start(self) -> None:
        self.thread.start()

    @property
    def active(self) -> bool:
        return self.state in ("starting", "running", "stopping")

    def request_stop(self, save: bool) -> None:
        if self.active:
            self.stop_request = "save" if save else "discard"
            self.state = "stopping"
            self.write_log("stop requested" + (", saving first" if save else ", without saving"))

    def write_log(self, text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {text}"
        print(f"[training {self.settings['name']}] {text}")
        self.log = (self.log + [line])[-LOG_LINES:]

    def status(self, preview_version: int = -1) -> dict:
        done = len(self.epoch_seconds)
        remaining = self.total_epochs - done
        eta = sum(self.epoch_seconds) / done * remaining if done else None
        status = {
            "name": self.settings["name"], "kind": self.settings["kind"],
            "description": self.description,
            "state": self.state, "stage": self.stage, "message": self.message,
            "epoch": self.epoch, "total_epochs": self.total_epochs,
            "batch": self.batch, "batches": self.batches,
            "elapsed": round(time.time() - self.started, 1),
            "eta": round(eta, 1) if eta is not None and self.active else None,
            "history": self.history, "log": self.log[-40:], "saved": self.saved,
            "preview_version": self.preview_version,
        }
        if preview_version != self.preview_version:
            status["preview"] = self.preview
        return status

    def check_stop(self) -> None:
        """Called after every batch: ends the run if a stop was requested."""
        if self.stop_request:
            if self.stop_request == "save":
                self.save()
            raise Stopped()

    # --- the run ---------------------------------------------------------------

    def run(self) -> None:
        try:
            self.state = "running"
            if self.settings["kind"] == "autoencoder":
                self.train_autoencoder()
            elif self.settings["kind"] == "ranknet":
                self.train_ranknet()
            else:
                self.train_classifier()
            self.state = "done"
            self.write_log("finished")
        except Stopped:
            self.state = "stopped"
            self.write_log("stopped" + (", model saved" if self.stop_request == "save" else ", not saved"))
        except Exception as e:
            self.state = "error"
            self.message = f"{type(e).__name__}: {e}"
            self.write_log(self.message)
            traceback.print_exc()
        finally:
            self.stage = ""
            self.model = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=int(self.settings["batch_size"]), shuffle=shuffle,
                          pin_memory=self.device.type == "cuda")

    def continuing(self) -> bool:
        return bool(self.settings.get("continue")) and (self.folder / "config.json").exists()

    def load_history(self) -> list:
        file = self.folder / "history.json"
        return json.loads(file.read_text()) if self.continuing() and file.exists() else []

    def epochs(self):
        """Yields the epoch numbers (counted across continued runs), timing them.
        self.epoch is the epoch of this run in progress (1, 2, ...)."""
        for _ in range(self.total_epochs):
            start = time.time()
            self.epoch += 1
            yield len(self.history) + 1
            self.epoch_seconds.append(time.time() - start)

    def should_save(self) -> bool:
        every = max(1, int(self.settings.get("save_every") or self.total_epochs))
        return self.epoch % every == 0 or self.epoch == self.total_epochs

    def save(self) -> None:
        self.model.save(self.folder)
        (self.folder / "history.json").write_text(json.dumps(self.history, indent=2))
        self.saved = True
        self.write_log(f"saved to {self.folder}")
        if self.on_saved:
            self.on_saved(self.folder)

    def batches_of(self, loader: DataLoader, stage: str):
        """Yields the batches of a loader, counting them and checking for stops"""
        self.stage = stage
        for batch in loader:
            yield batch
            self.batch += 1
            self.check_stop()

    # --- autoencoder -----------------------------------------------------------

    def train_autoencoder(self) -> None:
        s = self.settings
        # the page sends the exact list; otherwise every image under the folder
        paths = [Path(p) for p in s["image_paths"]] if s.get("image_paths") else find_images(s["dataset_path"])
        if not paths:
            raise ValueError(f"no images in {s['dataset_path']}")
        train_indices, eval_indices = split_indices(len(paths), float(s["eval_percent"]))
        train_paths, eval_paths = [paths[i] for i in train_indices], [paths[i] for i in eval_indices]

        if self.continuing():
            if folder_model_type(self.folder) != "autoencoder":
                raise ValueError(f"{self.folder} is not an autoencoder, can't continue it as one")
            autoencoder = Autoencoder.from_folder(self.folder)
            self.write_log("continuing the saved model")
        else:
            a = s["autoencoder"]
            autoencoder = Autoencoder.create(AutoencoderType(a["type"]), int(a["image_size"]),
                                             int(a["latent_dim"]), **a.get("options", {}))
        self.model = autoencoder.to(self.device)
        self.description = str(autoencoder)
        self.write_log(self.description)
        self.write_log(f"{len(train_paths)} training images, {len(eval_paths)} evaluation images")

        train_images = ImageFiles(train_paths, autoencoder.preprocess(
            augmentations(s.get("augment"), autoencoder.image_size)))
        eval_images = ImageFiles(eval_paths, autoencoder.preprocess())
        train_loader, eval_loader = self.loader(train_images, True), self.loader(eval_images, False)
        optimizers = autoencoder.make_optimizers(float(s["learning_rate"]))
        self.history = self.load_history()

        # the preview shows the same images every epoch: unseen ones if there are any
        shown = eval_images if len(eval_images) else ImageFiles(train_paths, autoencoder.preprocess())
        inputs = torch.stack([shown[i] for i in spread(len(shown), PREVIEW_SAMPLES)])
        input_urls = [data_url(autoencoder.to_image(x)) for x in inputs]

        for epoch in self.epochs():
            self.batch, self.batches = 0, len(train_loader) + len(eval_loader)
            train = self.autoencoder_pass(autoencoder, train_loader, optimizers)
            evaluation = self.autoencoder_pass(autoencoder, eval_loader) if len(eval_images) else None
            self.history.append({"train": train, "eval": evaluation})
            self.autoencoder_preview(autoencoder, inputs, input_urls)
            last = evaluation or train
            self.write_log(f"epoch {epoch}: " + "  ".join(f"{k} {v:.4f}" for k, v in last.items()))
            if self.should_save():
                self.save()

    def autoencoder_pass(self, autoencoder: Autoencoder, loader: DataLoader, optimizers=None) -> dict:
        training = optimizers is not None
        autoencoder.train(training)
        totals, count = {}, 0
        with torch.set_grad_enabled(training):
            for x in self.batches_of(loader, "training" if training else "evaluating"):
                x = x.to(self.device)
                if training:
                    parts = autoencoder.train_step(x, optimizers)
                else:
                    loss, parts = autoencoder.loss(x)
                    parts = {"loss": loss.item(), **parts}
                for name, value in parts.items():
                    totals[name] = totals.get(name, 0.0) + value * len(x)
                count += len(x)
        return {name: total / max(count, 1) for name, total in totals.items()}

    @torch.no_grad()
    def autoencoder_preview(self, autoencoder: Autoencoder, inputs: torch.Tensor, input_urls: list) -> None:
        autoencoder.eval()
        reconstructions, z = autoencoder(inputs.to(self.device))
        self.preview = {"kind": "autoencoder", "rows": [
            {"input": url, "output": data_url(autoencoder.to_image(r)), "latent": [round(v, 4) for v in code.tolist()]}
            for url, r, code in zip(input_urls, reconstructions, z.cpu())]}
        self.preview_version += 1

    # --- classifier ------------------------------------------------------------

    def train_classifier(self) -> None:
        s, c = self.settings, self.settings["classifier"]
        images = datasets.ImageFolder(s["dataset_path"], allow_empty=True)
        class_names = images.classes
        if len(class_names) < 2:
            raise ValueError(f"{s['dataset_path']} needs at least 2 subfolders (classes) with images")
        train_indices, eval_indices = split_indices(len(images), float(s["eval_percent"]))
        freeze = bool(c["freeze_backbone"])

        if self.continuing():
            if folder_model_type(self.folder) != "classifier":
                raise ValueError(f"{self.folder} is not a classifier, can't continue it as one")
            classifier = Classifier.from_folder(self.folder)
            if classifier.class_names != class_names:
                raise ValueError(f"the saved model has classes {classifier.class_names}, "
                                 f"the dataset has {class_names}")
            classifier.dropout = float(c["dropout"])
            classifier.set_freeze_backbone(freeze)
            self.write_log("continuing the saved model")
        else:
            self.stage = "loading the backbone"
            classifier = Classifier(ClassifierID.from_dict(c["backbone"]), class_names, head=c["head"],
                                    hidden_size=int(c["hidden_size"]), dropout=float(c["dropout"]),
                                    freeze_backbone=freeze)
        self.model = classifier.to(self.device)
        self.description = str(classifier)
        self.write_log(self.description)
        self.write_log(f"{len(class_names)} classes, {len(train_indices)} training images, "
                       f"{len(eval_indices)} evaluation images")

        random_transformations = augmentations(s.get("augment"), classifier.image_size()[0])
        train_images = ImageSubset(images, train_indices, classifier.preprocess(random_transformations))
        eval_images = ImageSubset(images, eval_indices, classifier.preprocess())
        train_loader, eval_loader = self.loader(train_images, True), self.loader(eval_images, False)
        model = classifier
        if freeze and not random_transformations:
            # Nothing changes the backbone or the images: run every image through
            # it once, then train the head alone on the stored features.
            self.batch, self.batches = 0, len(train_loader) + len(eval_loader)
            train_loader = DataLoader(self.features(classifier, train_loader), shuffle=True,
                                      batch_size=int(s["batch_size"]))
            eval_loader = DataLoader(self.features(classifier, eval_loader), batch_size=int(s["batch_size"]))
            model = classifier.head

        learning_rate = float(s["learning_rate"])
        groups = [{"params": classifier.head.parameters(), "lr": learning_rate}]
        if not freeze:
            groups.append({"params": classifier.backbone.parameters(),
                           "lr": learning_rate * float(c.get("backbone_lr_factor", 0.1))})
        optimizer = torch.optim.AdamW(groups)
        loss_fn = nn.CrossEntropyLoss()
        self.history = self.load_history()

        shown = eval_images if len(eval_images) else ImageSubset(images, train_indices, classifier.preprocess())
        picks = spread(len(shown), PREVIEW_SAMPLES)
        samples = [shown.load(i) for i in picks]
        sample_inputs = torch.stack([shown[i][0] for i in picks])
        sample_urls = [data_url(image, PREVIEW_SIZE, "JPEG") for image, _ in samples]
        eval_labels = torch.tensor([images.samples[i][1] for i in eval_indices])
        self.preview_context = {
            "images": images, "eval_indices": eval_indices, "eval_labels": eval_labels,
            "train_transform": train_images.transform if random_transformations else None,
            "thumbnails": {},       # evaluation image index -> data url (misclassified ones)
        }

        for epoch in self.epochs():
            self.batch, self.batches = 0, len(train_loader) + len(eval_loader)
            train_loss, train_accuracy, _ = self.classifier_pass(model, train_loader, loss_fn, optimizer)
            eval_loss = eval_accuracy = eval_logits = None
            if len(eval_indices):
                eval_loss, eval_accuracy, eval_logits = self.classifier_pass(model, eval_loader, loss_fn)
            # the same format as 05_b_classifiers/classifier_training.py
            self.history.append({"train_loss": train_loss, "train_accuracy": train_accuracy,
                                 "eval_loss": eval_loss, "eval_accuracy": eval_accuracy})
            self.classifier_preview(classifier, class_names, sample_inputs, sample_urls, samples, eval_logits)
            line = f"epoch {epoch}: train loss {train_loss:.4f} acc {train_accuracy:.1%}"
            if eval_loss is not None:
                line += f"  |  eval loss {eval_loss:.4f} acc {eval_accuracy:.1%}"
            self.write_log(line)
            if self.should_save():
                self.save()

    @torch.no_grad()
    def features(self, classifier: Classifier, loader: DataLoader) -> TensorDataset:
        classifier.eval()
        all_features, all_labels = [], []
        for images, labels in self.batches_of(loader, "computing features"):
            all_features.append(classifier.features(images.to(self.device)).cpu())
            all_labels.append(labels)
        if not all_features:
            return TensorDataset(torch.zeros(0, classifier.feature_size), torch.zeros(0, dtype=torch.long))
        return TensorDataset(torch.cat(all_features), torch.cat(all_labels))

    def classifier_pass(self, model: nn.Module, loader: DataLoader, loss_fn, optimizer=None):
        training = optimizer is not None
        model.train(training)
        total_loss, correct, count, all_logits = 0.0, 0, 0, []
        with torch.set_grad_enabled(training):
            for inputs, labels in self.batches_of(loader, "training" if training else "evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
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

    @torch.no_grad()
    def classifier_preview(self, classifier, class_names, inputs, urls, samples, eval_logits):
        """Sample images with their scores and a few augmented versions of each
        (what the network receives while training, new random ones every
        epoch), the confusion matrix and the most misclassified evaluation
        images of every class."""
        context = self.preview_context
        classifier.eval()
        scores = classifier(inputs.to(self.device)).softmax(dim=-1).cpu()

        mean, std = classifier.image_normalization()
        mean, std = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
        def displayable(x):         # normalized tensor -> image (normalization undone)
            return transforms.functional.to_pil_image((x * std + mean).clamp(0, 1))
        augmented = [[] for _ in samples]
        if context["train_transform"] is not None:
            for k, (image, _) in enumerate(samples):
                augmented[k] = [data_url(displayable(context["train_transform"](image)), AUGMENTED_SIZE, "JPEG")
                                for _ in range(AUGMENTED_VIEWS)]

        confusion, misclassified = None, []
        if eval_logits is not None:
            labels = context["eval_labels"]
            # confusion[true][predicted] = number of evaluation images
            n = len(class_names)
            probabilities = eval_logits.softmax(dim=-1)
            predicted = probabilities.argmax(dim=-1)
            confusion = torch.zeros(n, n, dtype=torch.long)
            confusion.index_put_((labels, predicted), torch.ones_like(predicted), accumulate=True)
            confusion = confusion.tolist()
            for c in range(n):
                of_class = labels == c
                wrong = (of_class & (predicted != c)).nonzero().flatten()
                # worst first: the lowest score for the class they really are
                wrong = wrong[probabilities[wrong, c].argsort()][:MISCLASSIFIED_PER_CLASS].tolist()
                misclassified.append({"total": int(of_class.sum()), "wrong": int((of_class & (predicted != c)).sum()),
                                      "images": [{"image": self.eval_thumbnail(i),
                                                  "predicted": int(predicted[i]),
                                                  "score": round(float(probabilities[i, predicted[i]]), 3),
                                                  "true_score": round(float(probabilities[i, c]), 3)}
                                                 for i in wrong]})

        self.preview = {"kind": "classifier", "class_names": class_names, "confusion": confusion,
                        "misclassified": misclassified,
                        "samples": [{"image": url, "label": label, "augmented": views,
                                     "scores": [round(v, 4) for v in s.tolist()]}
                                    for url, (_, label), views, s in zip(urls, samples, augmented, scores)]}
        self.preview_version += 1

    def eval_thumbnail(self, i: int) -> str:
        """data url of the i-th evaluation image (kept, they often stay misclassified)"""
        context = self.preview_context
        if i not in context["thumbnails"]:
            path = context["images"].samples[context["eval_indices"][i]][0]
            context["thumbnails"][i] = data_url(Image.open(path).convert("RGB"), PREVIEW_SIZE, "JPEG")
        return context["thumbnails"][i]

    # --- ranknet ---------------------------------------------------------------

    def train_ranknet(self) -> None:
        s, r = self.settings, self.settings["ranknet"]
        paths, names = [Path(p) for p in s["image_paths"]], s["image_names"]
        ranking_file = Path(s["ranking_path"])
        pairs, repeats, dropped = rankings.remap(rankings.read_ranking(ranking_file), names)
        if len(pairs) < 2:
            raise ValueError(f"{ranking_file.name} has {len(pairs)} usable pairs for this dataset, at least 2 are needed")

        self.stage = "loading the backbones"
        if self.continuing():
            if folder_model_type(self.folder) != "ranknet":
                raise ValueError(f"{self.folder} is not a RankNet, can't continue it as one")
            ranknet = RankNet.from_folder(self.folder)
            self.write_log("continuing the saved model")
        else:
            model_ids = [ModelID.from_string(m) for m in r["backbones"]]
            if not model_ids:
                raise ValueError("pick at least one backbone")
            ranknet = RankNet(model_ids, int(r["hidden_dim"]), float(r["dropout"]))
        ranknet.trained_on = {"dataset": s["dataset"], "ranking": ranking_file.name}
        self.model = ranknet.to(self.device)
        self.description = str(ranknet)
        self.write_log(self.description)
        consistency = rankings.self_consistency(repeats)
        self.write_log(f"{ranking_file.name}: {len(pairs)} pairs ({sum(p[2] != 0 for p in pairs)} with a preference)"
                       + (f", {dropped} dropped (images not in this dataset)" if dropped else "")
                       + (f", self-consistency {consistency['rate']:.0%} over {consistency['total']} repeats"
                          if consistency else ""))

        # Only the images in some pair are needed, each embedded once: the
        # backbones are frozen and nothing changes the images.
        used = sorted({i for p in pairs for i in p[:2]})
        row = {i: k for k, i in enumerate(used)}
        self.batch, self.batches = 0, -(-len(used) // EMBED_BATCH)
        self.stage = "embedding the images"
        parts = []
        for start in range(0, len(used), EMBED_BATCH):
            images = [Image.open(paths[i]).convert("RGB") for i in used[start:start + EMBED_BATCH]]
            parts.append(ranknet.embed(images))
            self.batch += 1
            self.check_stop()
        embeddings = torch.cat(parts)
        self.write_log(f"{len(used)} images embedded, {embeddings.shape[1]} features")

        index = torch.tensor([[row[p[0]], row[p[1]]] for p in pairs], device=self.device)
        pref = torch.tensor([float(p[2]) for p in pairs], device=self.device)
        train_pairs, eval_pairs = split_indices(len(pairs), float(s["eval_percent"]))
        if not train_pairs:
            train_pairs, eval_pairs = eval_pairs, []
        train_pairs = torch.tensor(train_pairs, device=self.device)
        eval_pairs = torch.tensor(eval_pairs, device=self.device, dtype=torch.long)
        self.write_log(f"{len(train_pairs)} training pairs, {len(eval_pairs)} evaluation pairs")

        optimizer = torch.optim.Adam(ranknet.head.parameters(), lr=float(s["learning_rate"]),
                                     weight_decay=float(r.get("weight_decay", 1e-3)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.total_epochs)
        batch_size = int(s["batch_size"])
        self.history = self.load_history()
        self.preview_context = {"paths": paths, "names": names, "used": used, "pairs": len(pairs),
                                "eval_pairs": eval_pairs.tolist(), "consistency": consistency,
                                "dropped": dropped, "thumbnails": {}}

        for epoch in self.epochs():
            self.batch, self.batches = 0, -(-len(train_pairs) // batch_size)
            self.stage = "training"
            ranknet.train()
            shuffled = train_pairs[torch.randperm(len(train_pairs), device=self.device)]
            total_loss, scores_a, scores_b, prefs = 0.0, [], [], []
            for start in range(0, len(shuffled), batch_size):
                batch = shuffled[start:start + batch_size]
                a, b = ranknet(embeddings[index[batch, 0]]), ranknet(embeddings[index[batch, 1]])
                loss = ranknet_loss(a, b, pref[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
                scores_a.append(a.detach())
                scores_b.append(b.detach())
                prefs.append(pref[batch])
                self.batch += 1
                self.check_stop()
            scheduler.step()
            train_accuracy = pair_accuracy(torch.cat(scores_a), torch.cat(scores_b), torch.cat(prefs))

            ranknet.eval()
            eval_loss = eval_accuracy = None
            with torch.no_grad():
                scores = ranknet(embeddings)
                if len(eval_pairs):
                    a, b = scores[index[eval_pairs, 0]], scores[index[eval_pairs, 1]]
                    eval_loss = ranknet_loss(a, b, pref[eval_pairs]).item()
                    eval_accuracy = pair_accuracy(a, b, pref[eval_pairs])
            # the same keys as a classifier's history: accuracy = share of the
            # pairs (with a preference) that the scores order as answered
            self.history.append({"train_loss": total_loss / len(train_pairs), "train_accuracy": train_accuracy,
                                 "eval_loss": eval_loss, "eval_accuracy": eval_accuracy})
            self.ranknet_preview(scores.cpu(), index.cpu(), pref.cpu())
            line = f"epoch {epoch}: train loss {total_loss / len(train_pairs):.4f}"
            if train_accuracy is not None:
                line += f" agreement {train_accuracy:.1%}"
            if eval_loss is not None:
                line += f"  |  eval loss {eval_loss:.4f}"
                if eval_accuracy is not None:
                    line += f" agreement {eval_accuracy:.1%}"
            self.write_log(line)
            # written into the model's card (model_card.py) with every save
            ranknet.training = {
                **ranknet.trained_on, "pairs": len(pairs), "strict_pairs": sum(p[2] != 0 for p in pairs),
                "dropped_pairs": dropped, "train_pairs": len(train_pairs), "eval_pairs": len(eval_pairs),
                "epochs": len(self.history), "self_consistency": consistency,
                **{k: self.history[-1][k] for k in ("train_loss", "train_accuracy", "eval_loss", "eval_accuracy")},
                "learning_rate": float(s["learning_rate"]), "batch_size": batch_size,
                "weight_decay": float(r.get("weight_decay", 1e-3)), "eval_percent": float(s["eval_percent"]),
            }
            if self.should_save():
                self.save()

    def ranknet_thumbnail(self, i: int) -> str:
        """data url of the dataset's i-th image (kept: the same ones come back)"""
        thumbnails = self.preview_context["thumbnails"]
        if i not in thumbnails:
            thumbnails[i] = data_url(Image.open(self.preview_context["paths"][i]).convert("RGB"), PREVIEW_SIZE, "JPEG")
        return thumbnails[i]

    def ranknet_preview(self, scores: torch.Tensor, index: torch.Tensor, pref: torch.Tensor) -> None:
        """The best and worst of the ranked images by the current scores, and
        the evaluation pairs the model disagrees with most."""
        context = self.preview_context
        used, names = context["used"], context["names"]
        order = scores.argsort(descending=True).tolist()

        def shown(rows):
            return [{"image": self.ranknet_thumbnail(used[k]), "name": names[used[k]],
                     "score": round(float(scores[k]), 3)} for k in rows]

        disagreements = []
        for p in context["eval_pairs"]:
            if pref[p] == 0:
                continue
            a, b = index[p].tolist()
            # how likely the model finds the other answer: P(the one not chosen is better)
            wrong = float(torch.sigmoid((scores[b] - scores[a]) * pref[p]))
            if wrong > 0.5:
                disagreements.append((wrong, a, b, int(pref[p])))
        disagreements.sort(reverse=True)
        self.preview = {
            "kind": "ranknet", "pairs": context["pairs"], "images": len(used),
            "dropped": context["dropped"], "consistency": context["consistency"],
            "best": shown(order[:RANKED_SHOWN]), "worst": shown(order[-RANKED_SHOWN:]),
            "disagreements": [{"a": self.ranknet_thumbnail(used[a]), "b": self.ranknet_thumbnail(used[b]),
                               "answer": answer, "model": round(1 - wrong, 3)}
                              for wrong, a, b, answer in disagreements[:DISAGREEMENTS_SHOWN]],
        }
        self.preview_version += 1
