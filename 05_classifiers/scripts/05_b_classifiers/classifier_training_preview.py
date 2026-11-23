"""Live matplotlib window for classifier_training.py, refreshed after every epoch.

Left, for a few random training images:

  original          the image file as it is
  model input       after the random transformations, resize, crop: what the
                    network actually receives (normalization undone so it can
                    be displayed)
  scores            softmax score of each class, the true class in green

Middle, the training and evaluation loss of every epoch so far.

Right, the most misclassified evaluation images: one row per true class, the
images of that class the model got wrong, worst first (lowest score for the
class they really belong to). Each title says what the model took it for.

save_pdf() writes the window into a PDF.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator

TRUE_COLOR = "tab:green"
OTHER_COLOR = "tab:gray"
FONT = "Courier New"
THUMBNAIL_SIZE = 256        # misclassified images are shrunk to this before drawing


class TrainingPreview:

    def __init__(self, classifier, train_subset, eval_subset, num_samples: int, num_misclassified: int,
                 device: torch.device):
        """train_subset, eval_subset: the ImageSubsets of the training script.
        The training one's transform includes the random transformations."""
        self.classifier = classifier
        self.train_subset = train_subset
        self.eval_subset = eval_subset
        self.num_samples = min(num_samples, len(train_subset))
        self.num_misclassified = num_misclassified
        self.device = device

        plt.rcParams["font.family"] = FONT
        plt.ion()   # interactive mode: drawing doesn't block the training

        # The misclassified column needs evaluation images
        n_classes = len(classifier.class_names)
        show_errors = len(eval_subset) > 0 and num_misclassified > 0
        samples_width = 11
        errors_width = 1.8 * num_misclassified + 1.2 if show_errors else 0
        height = max(2.2 * self.num_samples, 1.9 * n_classes if show_errors else 0)

        # "constrained" layout makes room for the (long) class names
        self.fig = plt.figure(figsize=(samples_width + errors_width, height), layout="constrained")
        self.fig.canvas.manager.set_window_title("training preview")

        # The samples and the misclassified images have different numbers of
        # rows, so each gets its own sub-figure with its own grid, side by side.
        if show_errors:
            samples_part, errors_part = self.fig.subfigures(1, 2, width_ratios=[samples_width, errors_width])
        else:
            samples_part, errors_part = self.fig, None

        grid = samples_part.add_gridspec(self.num_samples, 4, width_ratios=[1, 1, 2, 2.5])
        self.rows = [[samples_part.add_subplot(grid[r, c]) for c in range(3)] for r in range(self.num_samples)]
        self.loss_ax = samples_part.add_subplot(grid[:, 3])

        self.errors_part = errors_part
        if show_errors:
            self.errors_axes = errors_part.subplots(n_classes, num_misclassified, squeeze=False)
            self.eval_labels = torch.tensor([eval_subset.images.samples[i][1] for i in eval_subset.indices])

    def to_displayable(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized tensor [3, H, W] -> image [H, W, 3] with values 0..1"""
        mean, std = self.classifier.image_normalization()
        x = x.cpu() * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        return x.clamp(0, 1).permute(1, 2, 0)

    def update(self, epoch: int, history: list[dict], eval_logits: torch.Tensor = None) -> None:
        """eval_logits: the model's logits for every evaluation image, in the
        order of eval_subset (None if there are no evaluation images)."""
        self.draw_samples(epoch, history)
        if self.errors_part is not None and eval_logits is not None:
            self.draw_misclassified(epoch, eval_logits)
        plt.pause(0.001)    # give the window a moment to redraw

    @torch.no_grad()
    def draw_samples(self, epoch: int, history: list[dict]) -> None:
        picks = random.sample(range(len(self.train_subset)), self.num_samples)
        originals, inputs, labels = [], [], []
        for i in picks:
            path, label = self.train_subset.images.samples[self.train_subset.indices[i]]
            original = self.train_subset.images.loader(path)
            originals.append(original)
            inputs.append(self.train_subset.transform(original))
            labels.append(label)

        self.classifier.eval()
        scores = self.classifier(torch.stack(inputs).to(self.device)).softmax(dim=-1).cpu()

        class_names = self.classifier.class_names
        for row, (original, x, label, s) in enumerate(zip(originals, inputs, labels, scores)):
            ax_original, ax_input, ax_scores = self.rows[row]
            for ax in self.rows[row]:
                ax.clear()

            ax_original.imshow(original)
            ax_input.imshow(self.to_displayable(x))
            for ax in (ax_original, ax_input):
                ax.set_xticks([])
                ax.set_yticks([])

            colors = [TRUE_COLOR if c == label else OTHER_COLOR for c in range(len(class_names))]
            ax_scores.barh(range(len(class_names)), s.numpy(), color=colors)
            ax_scores.set_yticks(range(len(class_names)), class_names, fontsize=8)
            ax_scores.invert_yaxis()    # first class on top
            ax_scores.set_xlim(0, 1)
            ax_scores.tick_params(axis="x", labelsize=7)

        self.rows[0][0].set_title("original", fontsize=10)
        self.rows[0][1].set_title("model input", fontsize=10)
        self.rows[0][2].set_title("scores (green = true class)", fontsize=10)

        self.loss_ax.clear()
        epochs = range(1, len(history) + 1)
        self.loss_ax.plot(epochs, [h["train_loss"] for h in history], label="train")
        if history[-1]["eval_loss"] is not None:
            self.loss_ax.plot(epochs, [h["eval_loss"] for h in history], label="evaluation")
        self.loss_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.loss_ax.set_xlabel("epoch")
        self.loss_ax.set_ylabel("loss")
        self.loss_ax.set_title(f"epoch {epoch}")
        self.loss_ax.legend()
        self.loss_ax.grid(alpha=0.3)

    def draw_misclassified(self, epoch: int, eval_logits: torch.Tensor) -> None:
        class_names = self.classifier.class_names
        scores = eval_logits.softmax(dim=-1)
        predicted = scores.argmax(dim=-1)

        for c, class_name in enumerate(class_names):
            of_class = self.eval_labels == c
            wrong = (of_class & (predicted != c)).nonzero().flatten()
            # worst first: the lowest score for the class they really are
            wrong = wrong[scores[wrong, c].argsort()]

            for col, ax in enumerate(self.errors_axes[c]):
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                if col >= len(wrong):
                    ax.set_frame_on(False)      # fewer mistakes than columns
                    continue
                ax.set_frame_on(True)
                i = wrong[col].item()
                path = self.eval_subset.images.samples[self.eval_subset.indices[i]][0]
                image = self.eval_subset.images.loader(path)
                image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
                ax.imshow(image)
                p = predicted[i].item()
                ax.set_title(f"as {class_names[p]} {scores[i, p]:.2f}\n"
                             f"true class {scores[i, c]:.2f}", fontsize=7)

            self.errors_axes[c, 0].set_ylabel(f"{class_name}\n{len(wrong)}/{of_class.sum().item()} wrong",
                                              fontsize=8)

        self.errors_part.suptitle("most misclassified evaluation images", fontsize=10)

    def save_pdf(self, path: Path) -> None:
        """The window as it is now."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path)

    def keep_open(self) -> None:
        """Called after the last epoch: block until the window is closed."""
        if plt.get_fignums():
            plt.ioff()
            plt.show()
