"""Live matplotlib window for autoencoder_training.py, refreshed after every epoch.

Left, one row per image (always the same images, so you can watch them get
sharper from epoch to epoch):

  input             what the network receives (resized, center crop)
  reconstruction    what the decoder makes of the latent vector
  latent            the latent vector as a barcode, one stripe per number:
                    red positive, blue negative, white 0

Right, the training and evaluation loss of every epoch so far.

save_pdf() writes the window into a PDF.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator

FONT = "Courier New"
BARCODE_COLORMAP = "RdBu_r"
RESPONSIVE_INTERVAL = 0.05  # seconds between handling window events during an epoch


class TrainingPreview:

    def __init__(self, autoencoder, images, num_samples: int, device: torch.device):
        """images: an ImageFiles dataset without random transformations."""
        self.autoencoder = autoencoder
        self.device = device
        self.last_responsive = 0.0
        n = min(num_samples, len(images))
        # evenly spread over the dataset, the same ones every epoch
        self.inputs = torch.stack([images[i * len(images) // n] for i in range(n)])

        plt.rcParams["font.family"] = FONT
        plt.ion()   # interactive mode: drawing doesn't block the training

        self.fig = plt.figure(figsize=(14, max(2.0 * n, 6)), layout="constrained")
        self.fig.canvas.manager.set_window_title("training preview")
        grid = self.fig.add_gridspec(n, 4, width_ratios=[1, 1, 2.2, 3])
        self.rows = [[self.fig.add_subplot(grid[r, c]) for c in range(3)] for r in range(n)]
        self.loss_ax = self.fig.add_subplot(grid[:, 3])

    def update(self, epoch: int, history: list[dict]) -> None:
        self.draw_samples()
        self.draw_losses(epoch, history)
        plt.pause(0.001)    # give the window a moment to redraw

    @torch.no_grad()
    def draw_samples(self) -> None:
        self.autoencoder.eval()
        reconstructions, z = self.autoencoder(self.inputs.to(self.device))
        reconstructions, z = reconstructions.cpu(), z.cpu()
        # one color scale for all rows, symmetric around 0
        scale = z.abs().max().item() or 1.0

        for row, (x, r, code) in enumerate(zip(self.inputs, reconstructions, z)):
            ax_input, ax_output, ax_code = self.rows[row]
            for ax in self.rows[row]:
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
            ax_input.imshow(x.permute(1, 2, 0))
            ax_output.imshow(r.clamp(0, 1).permute(1, 2, 0))
            ax_code.imshow(code.view(1, -1), aspect="auto", cmap=BARCODE_COLORMAP,
                           vmin=-scale, vmax=scale, interpolation="nearest")

        self.rows[0][0].set_title("input", fontsize=10)
        self.rows[0][1].set_title("reconstruction", fontsize=10)
        self.rows[0][2].set_title(f"latent ({z.shape[1]} numbers, |max| {scale:.2f})", fontsize=10)

    def draw_losses(self, epoch: int, history: list[dict]) -> None:
        self.loss_ax.clear()
        epochs = range(1, len(history) + 1)
        self.loss_ax.plot(epochs, [h["train"]["loss"] for h in history], label="train")
        if history[-1]["eval"] is not None:
            self.loss_ax.plot(epochs, [h["eval"]["loss"] for h in history], label="evaluation")
        self.loss_ax.set_yscale("log")
        self.loss_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.loss_ax.set_xlabel("epoch")
        self.loss_ax.set_ylabel("loss")
        last = history[-1]["eval"] or history[-1]["train"]
        parts = "  ".join(f"{k} {v:.4f}" for k, v in last.items() if k != "loss")
        self.loss_ax.set_title(f"epoch {epoch}\n{parts}", fontsize=10)
        self.loss_ax.legend()
        self.loss_ax.grid(alpha=0.3)

    def keep_responsive(self) -> None:
        """Called during an epoch (after every batch) so the window can still be
        moved, resized or zoomed between updates. Only the GUI events are handled,
        nothing is redrawn, and at most every RESPONSIVE_INTERVAL seconds."""
        now = time.monotonic()
        if now - self.last_responsive < RESPONSIVE_INTERVAL:
            return
        self.last_responsive = now
        if plt.fignum_exists(self.fig.number):
            self.fig.canvas.flush_events()

    def save_pdf(self, path: Path) -> None:
        """The window as it is now."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path)

    def keep_open(self) -> None:
        """Called after the last epoch: block until the window is closed."""
        if plt.get_fignums():
            plt.ioff()
            plt.show()
