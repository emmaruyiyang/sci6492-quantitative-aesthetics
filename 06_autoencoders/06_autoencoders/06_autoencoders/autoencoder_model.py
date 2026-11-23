"""Autoencoder = encoder + decoder, trained to give back its own input.

    image  ->  encoder  ->  latent vector  ->  decoder  ->  reconstruction
               (shrinks)    e.g. 128 numbers   (grows)      (same size as image)

Nobody tells the network what the latent numbers should mean. The only thing
it is trained for is that the reconstruction looks like the input. Since the
latent vector is much smaller than the image (64 x 64 x 3 = 12288 numbers
squeezed into 128) the encoder has to find a compact description of what the
images have in common, and the decoder learns to draw an image from it.

Seven kinds, picked with AutoencoderType:

  MLP        fully connected layers on the flattened pixels. Simplest, has no
             idea that neighbouring pixels belong together.
  CNN        convolutions that halve the image size at every step, then a
             fully connected layer to the latent vector. The decoder mirrors it.
  RESNET     the CNN with residual blocks (x + f(x)) after every step: deeper,
             trains more easily.
  VGG        the encoder is the convolutional part of VGG-16, pretrained on
             ImageNet (optionally frozen). Only the decoder has to be learned.
  ATTENTION  the image is cut into patches, a transformer lets every patch look
             at every other one. The latent vector is read from a [CLS] token,
             the decoder is a second transformer that paints the patches back.
  VAE        variational autoencoder. The encoder gives a mean and a spread for
             every latent number, the decoder gets a random sample from them.
             An extra loss (KL divergence) keeps the codes close to a standard
             normal distribution, so the latent space has no holes: any point
             decodes to something image-like.
  SPARSE     CNN whose latent numbers are >= 0 and pushed towards 0 with an L1
             penalty: every image is described by a few active numbers.
  VAE_GAN    VAE + a discriminator, a third network trained to tell real
             images from the decoder's. The decoder is also trained to fool it,
             and reconstructions are compared through the discriminator's
             features, not only pixel by pixel. Sharper than the plain VAE's
             blurry averages. (Larsen et al. 2016, "Autoencoding beyond pixels
             using a learned similarity metric")

Every autoencoder takes RGB images with values 0..1 (no normalization) and its
decoder ends with a sigmoid, so the reconstruction is directly an image.

Saving writes a folder with

  config.json   model_type ("autoencoder"), version, which kind, its options
  weights.pt    all the weights (encoder and decoder)

config.json always has a "model_type" key, so a program that finds a model
folder can tell what it is before loading it: load_model() at the bottom of
this file returns an Autoencoder or a Classifier (classifier_model.py, a copy
of the one in 05_b_classifiers, so this folder works on its own) depending on
it.
"""

import json
import math
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

MODEL_TYPE = "autoencoder"
VERSION = "1.0"
SUPPORTED_VERSIONS = {"1.0"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AutoencoderType(Enum):
    MLP = "mlp"
    CNN = "cnn"
    RESNET = "resnet"
    VGG = "vgg"
    ATTENTION = "attention"
    VAE = "vae"
    SPARSE = "sparse"
    VAE_GAN = "vae_gan"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def channels_per_step(base_channels: int, steps: int, max_channels: int = 512) -> list[int]:
    """Channels after each halving: 32, 64, 128, 256 ... (doubling, capped)"""
    return [min(base_channels * 2 ** i, max_channels) for i in range(steps)]


def down_block(c_in: int, c_out: int) -> nn.Module:
    """Halves width and height: a 4x4 convolution with stride 2."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 4, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.2),
    )


def up_block(c_in: int, c_out: int) -> nn.Module:
    """Doubles width and height. Upsampling + an ordinary convolution, instead
    of a transposed convolution, avoids checkerboard patterns in the output."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )


class ResidualBlock(nn.Module):
    """x + f(x): the block only has to learn a correction to its input, and the
    gradient can flow straight through the "+" to earlier layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.f(x))


def to_rgb(c_in: int) -> nn.Module:
    """Last layer of every decoder: 3 channels with values 0..1"""
    return nn.Sequential(nn.Conv2d(c_in, 3, 3, padding=1), nn.Sigmoid())


def conv_encoder(image_size: int, latent_dim: int, base_channels: int, steps: int,
                 residual_blocks: int = 0) -> nn.Module:
    """[B, 3, S, S] -> halve `steps` times -> flatten -> [B, latent_dim]"""
    channels = channels_per_step(base_channels, steps)
    layers, c_in = [], 3
    for c in channels:
        layers.append(down_block(c_in, c))
        layers += [ResidualBlock(c) for _ in range(residual_blocks)]
        c_in = c
    side = image_size // 2 ** steps
    layers += [nn.Flatten(), nn.Linear(c_in * side * side, latent_dim)]
    return nn.Sequential(*layers)


def conv_decoder(image_size: int, latent_dim: int, base_channels: int, steps: int,
                 residual_blocks: int = 0) -> nn.Module:
    """[B, latent_dim] -> small image of many channels -> double `steps` times -> [B, 3, S, S]"""
    channels = channels_per_step(base_channels, steps)[::-1]
    side = image_size // 2 ** steps
    layers = [nn.Linear(latent_dim, channels[0] * side * side), nn.ReLU(),
              nn.Unflatten(1, (channels[0], side, side))]
    for c_in, c_out in zip(channels, channels[1:] + [base_channels]):
        layers.append(up_block(c_in, c_out))
        layers += [ResidualBlock(c_out) for _ in range(residual_blocks)]
    layers.append(to_rgb(base_channels))
    return nn.Sequential(*layers)


def check_divisible(image_size: int, factor: int, what: str) -> None:
    if image_size % factor:
        raise ValueError(f"image_size {image_size} must be a multiple of {factor} ({what})")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """Base of every kind. A subclass sets TYPE and DEFAULTS (its own options
    and their default values) and builds self.encoder and self.decoder. The
    defaults of encode / decode / forward / loss here work for the plain kinds,
    VAE and SPARSE override some of them."""

    TYPE: AutoencoderType = None
    DEFAULTS: dict = {}

    @staticmethod
    def create(autoencoder_type: AutoencoderType, image_size: int = 64, latent_dim: int = 128,
               **options) -> "Autoencoder":
        return AUTOENCODERS[autoencoder_type](image_size, latent_dim, **options)

    @staticmethod
    def from_config(config: dict) -> "Autoencoder":
        """A new (untrained) autoencoder as described by a config.json"""
        check_config(config)
        return Autoencoder.create(AutoencoderType(config["autoencoder_type"]),
                                  config["image_size"], config["latent_dim"], **config["options"])

    @staticmethod
    def from_folder(folder: str) -> "Autoencoder":
        folder = Path(folder)
        autoencoder = Autoencoder.from_config(read_config(folder))
        autoencoder.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu"))
        return autoencoder

    def __init__(self, image_size: int, latent_dim: int, **options):
        super().__init__()
        unknown = set(options) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"{self.TYPE} autoencoder has no options {sorted(unknown)}. "
                             f"Its options are {self.DEFAULTS}")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.options = {**self.DEFAULTS, **options}

    def __str__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (f"Autoencoder({self.TYPE}, {self.image_size}x{self.image_size} -> {self.latent_dim} latent, "
                f"{n / 1e6:.2f}M parameters, {trainable / 1e6:.2f}M trainable, options {self.options})")

    # --- forward --------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Images [B, 3, S, S] (0..1) -> latent vectors [B, latent_dim]"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent vectors [B, latent_dim] -> images [B, 3, S, S] (0..1)"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (reconstruction, latent)"""
        z = self.encode(x)
        return self.decode(z), z

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """-> (loss to minimize, {name: value} of each of its parts, for printing)

        Mean squared difference between the pixels of input and reconstruction."""
        reconstruction, _ = self(x)
        loss = F.mse_loss(reconstruction, x)
        return loss, {"reconstruction": loss.item()}

    # --- training ---------------------------------------------------------------

    def make_optimizers(self, learning_rate: float) -> list:
        """The optimizers train_step() uses: one for all the trainable weights
        (frozen ones, VGG with freeze_vgg, are left out)."""
        return [torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=learning_rate)]

    def train_step(self, x: torch.Tensor, optimizers: list) -> dict:
        """Learn from one batch -> {name: value} of the loss parts ("loss" = the total)"""
        loss, parts = self.loss(x)
        optimizer = optimizers[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), **parts}

    # --- images <-> tensors ---------------------------------------------------

    def preprocess(self, random_transformations: list = None) -> transforms.Compose:
        """PIL image -> tensor [3, S, S] with values 0..1. The short side is
        resized to S and the center square cropped. Random transformations
        (augmentations, during training) are applied first."""
        return transforms.Compose(list(random_transformations or []) + [
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
        ])

    @staticmethod
    def to_image(x: torch.Tensor) -> Image.Image:
        """Tensor [3, S, S] (0..1) -> PIL image"""
        return transforms.functional.to_pil_image(x.detach().clamp(0, 1).cpu())

    def images_to_tensor(self, images: list[Image.Image]) -> torch.Tensor:
        device = next(self.parameters()).device
        preprocess = self.preprocess()
        return torch.stack([preprocess(img.convert("RGB")) for img in images]).to(device)

    @torch.inference_mode()
    def embed(self, images: list[Image.Image]) -> torch.Tensor:
        """List of PIL images -> latent vectors [B, latent_dim] (on the cpu)"""
        self.eval()
        return self.encode(self.images_to_tensor(images)).cpu()

    @torch.inference_mode()
    def reconstruct(self, images: list[Image.Image]) -> tuple[list[Image.Image], torch.Tensor]:
        """List of PIL images -> (reconstructed PIL images, latent vectors)"""
        self.eval()
        z = self.encode(self.images_to_tensor(images))
        return [self.to_image(x) for x in self.decode(z)], z.cpu()

    # --- saving ---------------------------------------------------------------

    def config(self) -> dict:
        return {
            "model_type": MODEL_TYPE,
            "version": VERSION,
            "autoencoder_type": self.TYPE.value,
            "image_size": self.image_size,
            "latent_dim": self.latent_dim,
            "options": self.options,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }

    def save(self, folder: str) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps(self.config(), indent=2))
        torch.save(self.state_dict(), folder / "weights.pt")


# ---------------------------------------------------------------------------
# The kinds
# ---------------------------------------------------------------------------

class MLPAutoencoder(Autoencoder):
    TYPE = AutoencoderType.MLP
    DEFAULTS = {"hidden_sizes": [1024, 512]}

    def __init__(self, image_size: int, latent_dim: int, **options):
        super().__init__(image_size, latent_dim, **options)
        pixels = 3 * image_size * image_size
        hidden = list(self.options["hidden_sizes"])

        # pixels -> 1024 -> 512 -> latent
        encoder, sizes = [nn.Flatten()], [pixels] + hidden
        for a, b in zip(sizes, sizes[1:]):
            encoder += [nn.Linear(a, b), nn.ReLU()]
        encoder.append(nn.Linear(sizes[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder)

        # mirror: latent -> 512 -> 1024 -> pixels
        decoder, sizes = [], [latent_dim] + hidden[::-1]
        for a, b in zip(sizes, sizes[1:]):
            decoder += [nn.Linear(a, b), nn.ReLU()]
        decoder += [nn.Linear(sizes[-1], pixels), nn.Sigmoid(), nn.Unflatten(1, (3, image_size, image_size))]
        self.decoder = nn.Sequential(*decoder)


class CNNAutoencoder(Autoencoder):
    TYPE = AutoencoderType.CNN
    # steps: how many times the image is halved (64 -> 32 -> 16 -> 8 -> 4)
    DEFAULTS = {"base_channels": 32, "steps": 4}

    def __init__(self, image_size: int, latent_dim: int, **options):
        super().__init__(image_size, latent_dim, **options)
        base, steps = self.options["base_channels"], self.options["steps"]
        blocks = self.options.get("residual_blocks", 0)
        check_divisible(image_size, 2 ** steps, f"it is halved {steps} times")
        self.encoder = conv_encoder(image_size, latent_dim, base, steps, blocks)
        self.decoder = conv_decoder(image_size, latent_dim, base, steps, blocks)


class ResNetAutoencoder(CNNAutoencoder):
    """The CNN, with residual_blocks residual blocks after every halving and
    every doubling."""
    TYPE = AutoencoderType.RESNET
    DEFAULTS = {"base_channels": 32, "steps": 4, "residual_blocks": 2}


class ImageNetNormalize(nn.Module):
    """VGG was trained on ImageNet-normalized images, our inputs are 0..1"""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class VGGAutoencoder(Autoencoder):
    TYPE = AutoencoderType.VGG
    # freeze_vgg: keep the pretrained VGG layers as they are, train only the
    # layer to the latent vector and the decoder.
    DEFAULTS = {"freeze_vgg": True, "base_channels": 32}
    STEPS = 5       # VGG-16 halves the image 5 times (224 -> 7)

    def __init__(self, image_size: int, latent_dim: int, **options):
        super().__init__(image_size, latent_dim, **options)
        check_divisible(image_size, 2 ** self.STEPS, "VGG halves it 5 times")
        # The pretrained weights are downloaded once into the torch cache. When
        # loading a saved model they are replaced by the saved ones anyway.
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        side = image_size // 2 ** self.STEPS
        self.vgg = nn.Sequential(ImageNetNormalize(), vgg)
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(512 * side * side, latent_dim))
        self.encoder = nn.Sequential(self.vgg, self.to_latent)
        self.decoder = conv_decoder(image_size, latent_dim, self.options["base_channels"], self.STEPS)
        for p in self.vgg.parameters():
            p.requires_grad = not self.options["freeze_vgg"]


class AttentionAutoencoder(Autoencoder):
    TYPE = AutoencoderType.ATTENTION
    DEFAULTS = {"patch_size": 8, "width": 256, "heads": 4, "layers": 4}

    def __init__(self, image_size: int, latent_dim: int, **options):
        super().__init__(image_size, latent_dim, **options)
        p, width = self.options["patch_size"], self.options["width"]
        check_divisible(image_size, p, "it is cut into patches")
        self.grid = image_size // p                 # patches per side
        n_patches = self.grid ** 2

        def transformer():
            layer = nn.TransformerEncoderLayer(width, self.options["heads"], dim_feedforward=4 * width,
                                               dropout=0.0, activation="gelu", batch_first=True,
                                               norm_first=True)
            return nn.TransformerEncoder(layer, self.options["layers"], enable_nested_tensor=False)

        # encoder: patches -> tokens (+ where each patch is) -> transformer -> [CLS] -> latent
        self.patchify = nn.Conv2d(3, width, p, stride=p)    # one token per p x p patch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.encoder_positions = nn.Parameter(torch.randn(1, n_patches + 1, width) * 0.02)
        self.encoder_transformer = transformer()
        self.to_latent = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, latent_dim))

        # decoder: one token per patch position, all told the latent vector,
        # transformer -> each token becomes the pixels of its patch
        self.from_latent = nn.Linear(latent_dim, width)
        self.decoder_positions = nn.Parameter(torch.randn(1, n_patches, width) * 0.02)
        self.decoder_transformer = transformer()
        self.to_pixels = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3 * p * p))

    def encode(self, x):
        tokens = self.patchify(x).flatten(2).transpose(1, 2)        # [B, N, width]
        cls = self.cls_token.expand(len(x), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.encoder_positions
        tokens = self.encoder_transformer(tokens)
        return self.to_latent(tokens[:, 0])                          # the [CLS] token

    def decode(self, z):
        p, g = self.options["patch_size"], self.grid
        tokens = self.decoder_positions + self.from_latent(z).unsqueeze(1)
        patches = self.to_pixels(self.decoder_transformer(tokens))  # [B, N, 3*p*p]
        # put the patches back in their places: [B, g, g, 3, p, p] -> [B, 3, g*p, g*p]
        patches = patches.view(len(z), g, g, 3, p, p).permute(0, 3, 1, 4, 2, 5)
        return patches.reshape(len(z), 3, g * p, g * p).sigmoid()


class VAE(Autoencoder):
    TYPE = AutoencoderType.VAE
    # beta: weight of the KL loss. 1 is the classic VAE, more gives a smoother
    # latent space but blurrier reconstructions.
    DEFAULTS = {"base_channels": 32, "steps": 4, "beta": 1.0}

    def __init__(self, image_size: int, latent_dim: int, **options):
        super().__init__(image_size, latent_dim, **options)
        base, steps = self.options["base_channels"], self.options["steps"]
        check_divisible(image_size, 2 ** steps, f"it is halved {steps} times")
        # the encoder outputs 2 numbers per latent dimension: mean and log(variance)
        self.encoder = conv_encoder(image_size, 2 * latent_dim, base, steps)
        self.decoder = conv_decoder(image_size, latent_dim, base, steps)

    def distribution(self, x):
        mean, log_variance = self.encoder(x).chunk(2, dim=1)
        return mean, log_variance.clamp(-10, 10)

    def encode(self, x):
        """The mean: the most likely code, no randomness (for embeddings)"""
        return self.distribution(x)[0]

    def forward(self, x):
        mean, log_variance = self.distribution(x)
        if self.training:
            # "reparameterization trick": a random sample written as
            # mean + std * noise, so the gradient can flow into mean and std
            z = mean + (0.5 * log_variance).exp() * torch.randn_like(mean)
        else:
            z = mean
        return self.decode(z), z

    def loss(self, x):
        mean, log_variance = self.distribution(x)
        z = mean + (0.5 * log_variance).exp() * torch.randn_like(mean)
        reconstruction = F.mse_loss(self.decode(z), x)
        # KL divergence of N(mean, variance) from N(0, 1), summed over the
        # latent dimensions. Divided by the number of pixels so it is on the
        # same per-pixel scale as the mean squared error.
        kl = -0.5 * (1 + log_variance - mean ** 2 - log_variance.exp()).sum(dim=1).mean()
        kl = kl / x[0].numel()
        loss = reconstruction + self.options["beta"] * kl
        return loss, {"reconstruction": reconstruction.item(), "kl": kl.item()}


class SparseAutoencoder(Autoencoder):
    TYPE = AutoencoderType.SPARSE
    # sparsity: weight of the L1 penalty, the mean of |latent|. Higher -> fewer
    # active latent numbers per image, worse reconstructions.
    DEFAULTS = {"base_channels": 32, "steps": 4, "sparsity": 1e-3}

    def __init__(self, image_size: int, latent_dim: int, **options):
        super().__init__(image_size, latent_dim, **options)
        base, steps = self.options["base_channels"], self.options["steps"]
        check_divisible(image_size, 2 ** steps, f"it is halved {steps} times")
        # ReLU: the latent numbers are 0 or positive, so "inactive" is exactly 0
        self.encoder = nn.Sequential(conv_encoder(image_size, latent_dim, base, steps), nn.ReLU())
        self.decoder = conv_decoder(image_size, latent_dim, base, steps)

    def loss(self, x):
        reconstruction, z = self(x)
        mse = F.mse_loss(reconstruction, x)
        l1 = z.abs().mean()
        active = (z > 0).float().mean()
        loss = mse + self.options["sparsity"] * l1
        return loss, {"reconstruction": mse.item(), "l1": l1.item(), "active": active.item()}


class Discriminator(nn.Module):
    """Image -> one score: high for real images, low for the decoder's. The
    feature maps after each halving are returned too, they are used to
    compare a reconstruction with its input. Spectral normalization limits how
    fast each layer can change, which keeps the GAN training stable."""

    def __init__(self, image_size: int, base_channels: int, steps: int):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        blocks, c_in = [], 3
        for c in channels_per_step(base_channels, steps):
            blocks.append(nn.Sequential(sn(nn.Conv2d(c_in, c, 4, stride=2, padding=1)), nn.LeakyReLU(0.2)))
            c_in = c
        self.blocks = nn.ModuleList(blocks)
        side = image_size // 2 ** steps
        self.score = nn.Sequential(nn.Flatten(), sn(nn.Linear(c_in * side * side, 1)))

    def forward(self, x) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """-> (score [B] (a logit: > 0 means "real"), feature maps)"""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return self.score(x).squeeze(1), features


class VAEGAN(VAE):
    """The VAE (same encoder, decoder, KL loss) trained together with a
    Discriminator. Each batch:

      1. encoder + decoder learn to
           - reconstruct the pixels (as the VAE)
           - reconstruct the discriminator's features of the input (the
             "learned similarity": two images that look alike to it count as
             alike even if their pixels are not exactly the same)
           - keep the codes close to N(0, 1) (KL, as the VAE)
           - make the discriminator score their images as real, both the
             reconstructions and images decoded from random codes
      2. the discriminator learns to score real images high and the decoder's
         images (reconstructions and decoded random codes) low

    The discriminator is saved with the rest (weights.pt), so training can
    continue, but encode / decode / embeddings don't use it."""

    TYPE = AutoencoderType.VAE_GAN
    # feature: weight of the discriminator-feature reconstruction loss
    # adversarial: weight of "fool the discriminator". Too high and the images
    #   stop resembling their inputs, too low and it's a plain VAE.
    DEFAULTS = {"base_channels": 32, "steps": 4, "beta": 1.0, "feature": 1.0, "adversarial": 0.01}

    def __init__(self, image_size: int, latent_dim: int, **options):
        super().__init__(image_size, latent_dim, **options)
        self.discriminator = Discriminator(image_size, self.options["base_channels"], self.options["steps"])

    def generator_loss(self, x):
        """Loss of encoder + decoder -> (loss, parts, reconstructions, decoded random codes)"""
        mean, log_variance = self.distribution(x)
        z = mean + (0.5 * log_variance).exp() * torch.randn_like(mean)
        reconstruction = self.decode(z)
        dreamed = self.decode(torch.randn_like(mean))       # random codes from N(0, 1)

        pixels = F.mse_loss(reconstruction, x)
        kl = -0.5 * (1 + log_variance - mean ** 2 - log_variance.exp()).sum(dim=1).mean() / x[0].numel()

        _, real_features = self.discriminator(x)
        reconstruction_score, reconstruction_features = self.discriminator(reconstruction)
        dreamed_score, _ = self.discriminator(dreamed)
        feature = sum(F.mse_loss(r, f.detach()) for r, f in zip(reconstruction_features, real_features))
        feature = feature / len(real_features)
        real = torch.ones_like(reconstruction_score)
        adversarial = (F.binary_cross_entropy_with_logits(reconstruction_score, real) +
                       F.binary_cross_entropy_with_logits(dreamed_score, real)) / 2

        o = self.options
        loss = pixels + o["beta"] * kl + o["feature"] * feature + o["adversarial"] * adversarial
        parts = {"reconstruction": pixels.item(), "kl": kl.item(), "feature": feature.item(),
                 "adversarial": adversarial.item()}
        return loss, parts, reconstruction.detach(), dreamed.detach()

    def loss(self, x):
        loss, parts, _, _ = self.generator_loss(x)
        return loss, parts

    def make_optimizers(self, learning_rate: float) -> list:
        # Adam with beta1 = 0.5 is the usual choice for GANs: momentum that
        # remembers less, the opponent keeps changing.
        generator = [p for name, p in self.named_parameters() if not name.startswith("discriminator.")]
        return [torch.optim.Adam(generator, lr=learning_rate, betas=(0.5, 0.999)),
                torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))]

    def train_step(self, x, optimizers):
        generator_optimizer, discriminator_optimizer = optimizers

        # 1. encoder + decoder (this also leaves gradients in the
        #    discriminator, they are cleared before its own step)
        loss, parts, reconstruction, dreamed = self.generator_loss(x)
        generator_optimizer.zero_grad()
        loss.backward()
        generator_optimizer.step()

        # 2. discriminator: real images -> 1, decoder's images -> 0
        real_score, _ = self.discriminator(x)
        fake_score, _ = self.discriminator(torch.cat([reconstruction, dreamed]))
        d_loss = (F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score)) +
                  F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score)))
        discriminator_optimizer.zero_grad()
        d_loss.backward()
        discriminator_optimizer.step()

        # how sure the discriminator is (0..1): of real images being real, of fakes being real
        parts["discriminator"] = d_loss.item()
        parts["d_real"] = real_score.sigmoid().mean().item()
        parts["d_fake"] = fake_score.sigmoid().mean().item()
        return {"loss": loss.item(), **parts}


AUTOENCODERS = {cls.TYPE: cls for cls in
                (MLPAutoencoder, CNNAutoencoder, ResNetAutoencoder, VGGAutoencoder,
                 AttentionAutoencoder, VAE, SparseAutoencoder, VAEGAN)}


# ---------------------------------------------------------------------------
# config.json: what kind of model is in a folder
# ---------------------------------------------------------------------------

def read_config(folder) -> dict:
    return json.loads((Path(folder) / "config.json").read_text())


def check_config(config: dict) -> None:
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(f"not an autoencoder: model_type is {config.get('model_type')!r}")
    if config.get("version") not in SUPPORTED_VERSIONS:
        raise ValueError(f"autoencoder version {config.get('version')!r} is not supported "
                         f"(supported: {sorted(SUPPORTED_VERSIONS)})")


def model_type_of(config: dict) -> str | None:
    """"autoencoder", "classifier", or None for a config.json of something else.

    Classifiers saved before model_type existed are recognised by their keys."""
    model_type = config.get("model_type")
    if model_type is None and "backbone" in config and "class_names" in config:
        return "classifier"
    if model_type == MODEL_TYPE and config.get("version") not in SUPPORTED_VERSIONS:
        return None
    return model_type if model_type in ("autoencoder", "classifier") else None


def folder_model_type(folder) -> str | None:
    """model_type_of() the config.json in a folder, None if it has none or it
    can't be read."""
    try:
        return model_type_of(read_config(folder))
    except (OSError, ValueError):
        return None


def load_model(folder) -> nn.Module:
    """Rebuild whatever model is saved in the folder: Autoencoder or Classifier."""
    model_type = folder_model_type(folder)
    if model_type == "autoencoder":
        return Autoencoder.from_folder(folder)
    if model_type == "classifier":
        from classifier_model import Classifier
        return Classifier.from_folder(folder)
    raise ValueError(f"{folder} does not contain a model this program knows how to load")


def self_test(autoencoder_type: AutoencoderType, image_size: int = 64, latent_dim: int = 16) -> None:
    """Quick test: build one, run a batch through it, print the shapes."""
    ae = Autoencoder.create(autoencoder_type, image_size, latent_dim)
    x = torch.rand(2, 3, image_size, image_size)
    reconstruction, z = ae(x)
    loss, parts = ae.loss(x)
    assert reconstruction.shape == x.shape and z.shape == (2, latent_dim), (reconstruction.shape, z.shape)
    print(f"{ae}\n    loss {loss.item():.4f} {parts}")


if __name__ == "__main__":
    for t in AutoencoderType:
        self_test(t)
