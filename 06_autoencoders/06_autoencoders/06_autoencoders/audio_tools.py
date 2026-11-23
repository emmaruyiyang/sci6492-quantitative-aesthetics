"""Turn a sound file into a dataset of spectrogram images, one per fragment.

    sound  ->  fragments of `duration` seconds, overlapping by `overlap`
           ->  one spectrogram per fragment: a resolution x resolution grayscale
               image, time from left to right, frequency from bottom (low) to
               top (high), brightness = loudness (in decibels)
           ->  fragment_00000.png + fragment_00000.wav (the sound of that image)

Frequency scales:

  mel      bands spaced the way we hear pitch: narrow at low frequencies, wide
           at high ones (the usual for sound recognition)
  linear   bands of equal width in Hz: most of the image is high frequencies

Brightness normalization:

  fragment   each image uses its own loudest point as white: quiet fragments
             look as bright as loud ones (compares the shape of the sound)
  clip       the loudest point of the whole sound is white in every image:
             quiet fragments are darker (keeps loudness differences)

`dynamic range` decibels below that loudest point become black.
"""

import base64
import io
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULTS = {
    "duration": 1.0,            # seconds per fragment
    "overlap": 25.0,            # percent of a fragment shared with the next one
    "resolution": 256,          # image width and height
    "scale": "mel",             # "mel" or "linear"
    "normalization": "fragment",  # "fragment" or "clip"
    "dynamic_range": 80.0,      # decibels from white to black
    "sample_rate": 22050,       # the sound is resampled to this (Hz)
    "audio_format": "wav",      # "wav" or "mp3", the fragments' sound files
}
SCALES = ["mel", "linear"]
NORMALIZATIONS = ["fragment", "clip"]
SAMPLE_RATES = [16000, 22050, 44100]
AUDIO_FORMATS = ["wav", "mp3"]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac"}
MEL_FFT = 2048
LOG_LINES = 200


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Any sound file -> mono float32 samples at sample_rate"""
    import librosa         # slow to import, only when needed
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return y.astype(np.float32)


def fragment_starts(n_samples: int, sample_rate: int, duration: float, overlap: float) -> list[int]:
    """First sample of every whole fragment (a shorter last piece is dropped)"""
    length = max(1, round(duration * sample_rate))
    step = max(1, round(length * (1 - overlap / 100)))
    if n_samples < length:
        return []
    return list(range(0, n_samples - length + 1, step))


def power_spectrogram(y: np.ndarray, sample_rate: int, resolution: int, scale: str, hop: int) -> np.ndarray:
    """-> power [resolution frequency bands, frames]"""
    import librosa
    if scale == "mel":
        return librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=MEL_FFT, hop_length=hop,
                                              n_mels=resolution, power=2.0)
    # linear: n_fft chosen so there are exactly `resolution` frequency bins
    n_fft = 2 * (resolution - 1)
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2


def spectrogram(fragment: np.ndarray, sample_rate: int, p: dict, top_db: float = None) -> Image.Image:
    """One fragment -> resolution x resolution grayscale image. top_db: the
    decibels drawn as white (None: this fragment's loudest point)."""
    resolution = int(p["resolution"])
    hop = max(1, len(fragment) // (resolution - 1))         # about `resolution` time frames
    power = power_spectrogram(fragment, sample_rate, resolution, p["scale"], hop)
    db = 10 * np.log10(np.maximum(power, 1e-12))
    top = db.max() if top_db is None else top_db
    dynamic = float(p["dynamic_range"])
    gray = np.clip((db - (top - dynamic)) / dynamic, 0, 1)
    image = Image.fromarray((np.flipud(gray) * 255).round().astype(np.uint8), "L")   # low frequencies at the bottom
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.BILINEAR)
    return image


def clip_top_db(y: np.ndarray, sample_rate: int, p: dict) -> float:
    """The loudest point of the whole sound, in decibels (for "clip" normalization),
    measured with the same frequency bands as the fragments"""
    resolution = int(p["resolution"])
    hop = max(1, round(float(p["duration"]) * sample_rate) // (resolution - 1))
    power = power_spectrogram(y, sample_rate, resolution, p["scale"], hop)
    return float(10 * np.log10(np.maximum(power, 1e-12)).max())


def sound_bytes(fragment: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    import soundfile
    buffer = io.BytesIO()
    if fmt == "mp3":
        soundfile.write(buffer, fragment, sample_rate, format="MP3", subtype="MPEG_LAYER_III")
    else:
        soundfile.write(buffer, fragment, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def sound_data_url(fragment: np.ndarray, sample_rate: int) -> str:
    return "data:audio/wav;base64," + base64.b64encode(sound_bytes(fragment, sample_rate, "wav")).decode()


def settings(body: dict) -> dict:
    """The page's settings, checked and with defaults"""
    p = {k: type(v)(body.get(k, v)) for k, v in DEFAULTS.items()}
    p["duration"] = min(max(p["duration"], 0.05), 60.0)
    p["overlap"] = min(max(p["overlap"], 0.0), 95.0)
    p["resolution"] = min(max(p["resolution"], 16), 1024)
    p["dynamic_range"] = max(p["dynamic_range"], 1.0)
    for key, allowed in (("scale", SCALES), ("normalization", NORMALIZATIONS),
                         ("sample_rate", SAMPLE_RATES), ("audio_format", AUDIO_FORMATS)):
        if p[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}")
    return p


class AudioClip:
    """An uploaded sound, decoded once (per sample rate) for previews and the conversion"""

    def __init__(self, path: Path, name: str):
        self.path, self.name = Path(path), name
        self.decoded = {}           # sample rate -> samples
        self.lock = threading.Lock()

    def samples(self, sample_rate: int) -> np.ndarray:
        with self.lock:
            if sample_rate not in self.decoded:
                self.decoded = {sample_rate: load_audio(self.path, sample_rate)}   # keep one
            return self.decoded[sample_rate]


class AudioRun:
    """Writes the fragments in a background thread. Same status() as
    dataset_tools.ConversionRun, so the page shows both the same way."""

    def __init__(self, clip: AudioClip, target: Path, p: dict, on_done=None):
        self.clip, self.target, self.p = clip, Path(target), p
        self.on_done = on_done
        self.state = "running"
        self.done, self.failed, self.total = 0, 0, 0
        self.message = ""
        self.log = []
        self.stop_requested = False
        self.started = time.time()
        self.thread = threading.Thread(target=self.run, daemon=True)

    @property
    def active(self) -> bool:
        return self.state in ("running", "stopping")

    def start(self) -> None:
        self.write_log(f"spectrograms of {self.clip.name} -> {self.target}")
        self.thread.start()

    def stop(self) -> None:
        if self.active:
            self.stop_requested = True
            self.state = "stopping"

    def write_log(self, text: str) -> None:
        print(f"[audio] {text}")
        self.log = (self.log + [f"{time.strftime('%H:%M:%S')}  {text}"])[-LOG_LINES:]

    def status(self) -> dict:
        elapsed = time.time() - self.started
        rate = self.done / elapsed if elapsed > 0 else 0
        remaining = self.total - self.done
        return {"state": self.state, "done": self.done, "failed": self.failed, "total": self.total,
                "target": str(self.target), "message": self.message, "log": self.log[-40:],
                "elapsed": round(elapsed, 1), "eta": round(remaining / rate, 1) if rate and self.active else None,
                "conversion": f"spectrograms of {self.clip.name}", "last": None}

    def run(self) -> None:
        p = self.p
        try:
            sr = int(p["sample_rate"])
            y = self.clip.samples(sr)
            starts = fragment_starts(len(y), sr, float(p["duration"]), float(p["overlap"]))
            self.total = len(starts)
            length = round(float(p["duration"]) * sr)
            self.write_log(f"{len(y) / sr:.1f} s of sound, {len(starts)} fragments of {p['duration']} s, "
                           f"{p['overlap']:.0f}% overlap, {p['resolution']} px, {p['scale']}, "
                           f"normalized per {p['normalization']}")
            top_db = clip_top_db(y, sr, p) if p["normalization"] == "clip" else None
            self.target.mkdir(parents=True, exist_ok=True)
            digits = max(5, len(str(len(starts))))
            for k, start in enumerate(starts):
                if self.stop_requested:
                    self.state = "stopped"
                    self.write_log(f"stopped after {self.done} fragments (they stay in {self.target})")
                    return
                fragment = y[start:start + length]
                stem = self.target / f"fragment_{k:0{digits}d}"
                spectrogram(fragment, sr, p, top_db).save(stem.with_suffix(".png"))
                stem.with_suffix(f".{p['audio_format']}").write_bytes(sound_bytes(fragment, sr, p["audio_format"]))
                self.done += 1
            self.state = "done"
            self.write_log(f"finished: {self.done} spectrograms + sounds in {self.target}")
        except Exception as e:
            self.state = "error"
            self.message = f"{type(e).__name__}: {e}"
            self.write_log(self.message)
            traceback.print_exc()
        finally:
            if self.on_done:
                self.on_done()
