"""A small, well spread subset of a dataset: the images to rank (or anything
else that needs a few hundred images that cover a large collection).

(From DiffDrawing's aesthetic_scoring/taste_sampler.py.)

Why not a random sample: a large collection is very uneven. A random 200 of
WikiArt is about half 19th century European oil paintings, so most of the
ranking would re-confirm one region of the embedding space and say little about
the rest, where the RankNet will later have to guess.

Why not farthest-point sampling (each next image the farthest from the ones
taken): it maximizes coverage, so it picks the outliers first: bad scans,
frames, photographed sculpture, junk.

So: cluster, then sample within the clusters.

    images  ->  model  ->  unit vectors  ->  spherical k-means: K clusters
            ->  near-duplicates removed in each cluster (cosine >= dedup)
            ->  N images shared out between the clusters (about N / K each)
            ->  per cluster: its medoid (the image closest to its center),
                then farthest-point picks inside the cluster

The medoid stands for the cluster; an outlier costs one cluster, not a third of
the budget. N / K is the real choice: images of the same cluster are alike, and
pairs of alike images ("these two are similar, but I prefer this one") are where
most of what people call taste is. With N = K every pair is between two
different regions: the coarse axes (abstract / figurative, dark / bright) are
learned in a few dozen answers, then nothing more.

    K = N / 4     balanced (default)
    K = N / 2     mostly a global good / bad axis
    K = N / 10    fine distinctions within a narrow domain

Use the same embedding model as the RankNet will: the selection covers the
space that model sees.

The new dataset folder gets the images (JPEG, 00000_<name>.jpg ..., in cluster
order) and taste_manifest.json: for every image where it came from and its
cluster. The Ranking page reads the clusters to ask pairs within a cluster.
"""

import json
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

MANIFEST_SCHEMA = "diffdraw.taste.manifest/1"
MANIFEST_FILENAME = "taste_manifest.json"
JPEG_QUALITY = 92
LOG_LINES = 200

DEFAULTS = {
    "count": 200,           # images to select
    "clusters": 50,         # regions to cover
    "dedup": 0.95,          # cosine above which two images count as the same picture (1 = off)
    "group_by": "none",     # none, folder, prefix: what counts as "the same author" for group_cap
    "group_cap": 0,         # at most this many images per group (0 = off)
    "max_side": 512,        # longest side of the written images (0 = keep)
    "max_short_side": 300,  # shortest side (0 = keep): a panorama doesn't become a strip
    "seed": 0,
}
GROUPINGS = {
    "none": "no grouping",
    "folder": "subfolder",
    "prefix": "file name up to the first _ (WikiArt: the artist)",
}


def group_of(name: str, group_by: str) -> str:
    path = Path(name)
    if group_by == "folder":
        return path.parent.as_posix().lower()
    if group_by == "prefix":
        return path.stem.split("_", 1)[0].lower() if "_" in path.stem else path.parent.as_posix().lower()
    return ""


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def spherical_kmeans(features: np.ndarray, k: int, iterations: int = 30, seed: int = 0):
    """k-means on unit vectors, by cosine similarity, k-means++ seeding.
    -> (cluster of every vector, the k unit centers)"""
    n = len(features)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centers = np.empty((k, features.shape[1]), dtype=np.float32)
    centers[0] = features[rng.integers(n)]
    closest = 1.0 - features @ centers[0]
    for c in range(1, k):
        weights = np.maximum(closest, 0.0) ** 2
        total = float(weights.sum())
        centers[c] = features[rng.integers(n) if total <= 0 else rng.choice(n, p=weights / total)]
        closest = np.minimum(closest, 1.0 - features @ centers[c])

    labels = np.full(n, -1, dtype=np.int32)
    for _ in range(iterations):
        similarity = features @ centers.T
        new_labels = similarity.argmax(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = features[labels == c]
            if not len(members):
                # an emptied cluster restarts on the worst covered vector
                centers[c] = features[similarity.max(axis=1).argmin()]
                continue
            summed = members.sum(axis=0)
            norm = float(np.linalg.norm(summed))
            if norm > 0:
                centers[c] = summed / norm
    return (features @ centers.T).argmax(axis=1).astype(np.int32), centers


def dedup(features: np.ndarray, members: list[int], threshold: float) -> list[int]:
    """members without near-duplicates (cosine >= threshold), the first of each kept"""
    if len(members) <= 1 or threshold >= 1.0:
        return list(members)
    block = features[members]
    dropped, kept = set(), []
    for start in range(0, len(members), 256):
        similarity = block[start:start + 256] @ block.T
        for offset in range(similarity.shape[0]):
            i = start + offset
            if i in dropped:
                continue
            kept.append(members[i])
            dropped.update(int(j) for j in np.flatnonzero(similarity[offset] >= threshold) if j > i)
    return kept


def allocate(sizes: list[int], count: int) -> list[int]:
    """How many images each cluster gives, count in total: an even share, the
    rest to the clusters that still have images, largest first."""
    targets = [0] * len(sizes)
    remaining = min(count, sum(sizes))
    while remaining > 0:
        open_ = [c for c in range(len(sizes)) if targets[c] < sizes[c]]
        if not open_:
            break
        share = max(1, remaining // len(open_))
        for c in sorted(open_, key=lambda c: sizes[c], reverse=True):
            if remaining <= 0:
                break
            give = min(share, sizes[c] - targets[c], remaining)
            targets[c] += give
            remaining -= give
    return targets


def pick_in_cluster(features, members, center, target, group, group_counts, group_cap) -> list[tuple[int, bool]]:
    """The medoid, then farthest-point picks within the cluster, at most
    group_cap per group. -> [(image index, is it the medoid)]"""
    if not members or target <= 0:
        return []
    block = features[members]
    chosen = []

    def allowed(k):
        return k not in chosen and (group_cap <= 0 or group_counts.get(group[members[k]], 0) < group_cap)

    def take(k):
        chosen.append(k)
        group_counts[group[members[k]]] = group_counts.get(group[members[k]], 0) + 1

    medoid = next((int(k) for k in (block @ center).argsort()[::-1] if allowed(int(k))), None)
    if medoid is None:
        return []
    take(medoid)
    distance = 1.0 - block @ block[medoid]
    while len(chosen) < target:
        k = next((int(k) for k in distance.argsort()[::-1] if allowed(int(k))), None)
        if k is None:
            break
        take(k)
        distance = np.minimum(distance, 1.0 - block @ block[k])
    return [(members[k], k == medoid) for k in chosen]


def select(features: np.ndarray, names: list[str], p: dict, log=print) -> list[tuple[int, int, bool]]:
    """cluster -> dedup -> allocate -> pick. features: unit vectors.
    -> [(image index, cluster, is medoid)], in cluster order"""
    labels, centers = spherical_kmeans(features, int(p["clusters"]), seed=int(p["seed"]))
    group = [group_of(n, p["group_by"]) for n in names]
    group_counts = {}
    clusters = [dedup(features, np.flatnonzero(labels == c).tolist(), float(p["dedup"])) for c in range(len(centers))]
    dropped = len(names) - sum(len(c) for c in clusters)
    sizes = [len(c) for c in clusters]
    log(f"{len(centers)} clusters of {min(sizes)} to {max(sizes)} images"
        + (f", {dropped} near-duplicates left out" if dropped else ""))
    picked = []
    for c, target in enumerate(allocate(sizes, int(p["count"]))):
        picked += [(i, c, medoid) for i, medoid in
                   pick_in_cluster(features, clusters[c], centers[c], target, group, group_counts, int(p["group_cap"]))]
    if len(picked) < int(p["count"]):
        log(f"only {len(picked)} of {p['count']} images: the group cap or the duplicates left no others")
    return picked


def downscale(image: Image.Image, max_side: int, max_short_side: int) -> Image.Image:
    """Smaller so that the longest side <= max_side and the shortest <= max_short_side
    (0 = no limit), the tighter one wins. Never larger."""
    w, h = image.size
    scale = 1.0
    if max_side:
        scale = min(scale, max_side / max(w, h))
    if max_short_side:
        scale = min(scale, max_short_side / min(w, h))
    if scale >= 1.0:
        return image
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def clusters_of(folder, names: list[str]) -> list[int] | None:
    """The cluster of each of names from the folder's taste_manifest.json, None
    if there is none or it doesn't cover all of them (images added since)."""
    path = Path(folder) / MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        by_name = {r["name"]: int(r["cluster"]) for r in manifest.get("images", [])
                   if "name" in r and r.get("cluster") is not None}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not names or any(n not in by_name for n in names):
        return None
    return [by_name[n] for n in names]


# ---------------------------------------------------------------------------
# A sampling run (background thread), with the status() of a ConversionRun so
# the Dataset tab's monitor shows it
# ---------------------------------------------------------------------------

class SamplerRun:

    def __init__(self, paths: list[Path], names: list[str], source: str, target: Path, p: dict,
                 model_label: str, embed, on_done=None):
        """embed(job) -> unit-normable vectors [len(paths), D] of the images;
        job is a dict with "stage", "progress" (0..1) set by embed, and
        "cancelled", set here to stop it."""
        self.paths, self.names, self.source, self.target = paths, names, source, Path(target)
        self.p, self.model_label, self.embed, self.on_done = p, model_label, embed, on_done
        self.state = "running"
        self.phase = "embedding"
        self.done, self.failed, self.total = 0, 0, len(paths)
        self.message = ""
        self.log = []
        self.stop_requested = False
        self.job = {"stage": "", "progress": 0.0, "cancelled": False}
        self.started = time.time()
        self.phase_started = self.started
        self.thread = threading.Thread(target=self.run, daemon=True)

    @property
    def active(self) -> bool:
        return self.state in ("running", "stopping")

    def start(self) -> None:
        self.write_log(f"sampling {self.p['count']} of {len(self.paths)} images of {self.source} "
                       f"in {self.p['clusters']} clusters, by {self.model_label} -> {self.target}")
        self.thread.start()

    def stop(self) -> None:
        if self.active:
            self.stop_requested = True
            self.job["cancelled"] = True
            self.state = "stopping"

    def write_log(self, text: str) -> None:
        print(f"[sampler] {text}")
        self.log = (self.log + [f"{time.strftime('%H:%M:%S')}  {text}"])[-LOG_LINES:]

    def status(self) -> dict:
        if self.phase == "embedding":
            self.done = round(self.job["progress"] * self.total)
        elapsed = time.time() - self.started
        phase_elapsed = time.time() - self.phase_started
        rate = (self.done + self.failed) / phase_elapsed if phase_elapsed > 0 else 0
        remaining = self.total - self.done - self.failed
        return {"state": self.state, "done": self.done, "failed": self.failed, "total": self.total,
                "target": str(self.target), "message": self.message, "log": self.log[-40:],
                "elapsed": round(elapsed, 1), "eta": round(remaining / rate, 1) if rate and self.active else None,
                "conversion": f"Sampler: {self.phase}", "last": None}

    def run(self) -> None:
        try:
            vectors = np.asarray(self.embed(self.job), dtype=np.float32)
            if self.stop_requested:
                raise InterruptedError()
            features = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            self.write_log(f"{len(features)} images embedded, {features.shape[1]} dimensions")
            self.phase = "selecting"
            picked = select(features, self.names, self.p, self.write_log)
            self.phase, self.phase_started = "writing", time.time()
            self.done, self.total = 0, len(picked)
            self.write_images(picked)
            self.state = "done"
            self.write_log(f"finished: {self.done} images and {MANIFEST_FILENAME} in {self.target}"
                           + (f", {self.failed} could not be read" if self.failed else ""))
        except InterruptedError:
            self.state = "stopped"
            self.write_log("stopped")
        except Exception as e:
            if self.stop_requested:
                self.state = "stopped"
                self.write_log("stopped")
            else:
                self.state = "error"
                self.message = f"{type(e).__name__}: {e}"
                self.write_log(self.message)
                traceback.print_exc()
        finally:
            if self.on_done:
                self.on_done()

    def write_images(self, picked) -> None:
        self.target.mkdir(parents=True, exist_ok=True)
        records = []
        for position, (i, cluster, medoid) in enumerate(picked):
            if self.stop_requested:
                self.write_log(f"stopped after {self.done} images (they stay in {self.target}, without manifest)")
                raise InterruptedError()
            name = f"{position:05d}_{Path(self.names[i]).stem[:60]}.jpg"
            try:
                with Image.open(self.paths[i]) as image:
                    image = downscale(image.convert("RGB"), int(self.p["max_side"]), int(self.p["max_short_side"]))
                    image.save(self.target / name, "JPEG", quality=JPEG_QUALITY)
            except Exception as e:
                self.failed += 1
                self.write_log(f"can't copy {self.names[i]}: {e}")
                continue
            self.done += 1
            records.append({"name": name, "width": image.width, "height": image.height,
                            "source": self.names[i], "cluster": int(cluster), "is_medoid": bool(medoid),
                            "group": group_of(self.names[i], self.p["group_by"])})
        manifest = {"schema": MANIFEST_SCHEMA, "dataset": self.target.name,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "backbone": self.model_label, "sources": [self.source],
                    "selection": {k: self.p[k] for k in DEFAULTS}, "images": records}
        (self.target / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
