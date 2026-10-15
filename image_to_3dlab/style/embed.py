"""Pictures to unit-length vectors, with the two embeddings in the style race.

    embedder = load_embedder("csd")              # or "siglip2"; MissingModel if not downloaded
    vectors = embed_paths(embedder, paths, cache=bible / ".cache" / "embeddings-csd.npz")

`csd` is CSD ViT-L, trained to tell styles apart (its style head); `siglip2` is SigLIP 2's
image tower, a general embedding. Both run on the processor by default: a few hundred
milliseconds a picture, and the GPU stays free for generation.

Vectors are cached by the picture's SHA-256 and the model, so a bible's pictures are
embedded once, and adding one embeds one.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image_to_3dlab.style import models
from image_to_3dlab.style.bible import sha256_file

# Concepts are drawn on light grey; a cut-out (alpha) picture is put back on the same grey,
# or its transparent corners would read as black and change the style vector.
BACKGROUND = (214, 214, 214)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            flat = Image.new("RGB", rgba.size, BACKGROUND)
            flat.paste(rgba, mask=rgba.getchannel("A"))
            return flat
        return image.convert("RGB")


def clip_tensor(image: Image.Image, size: int = 224):
    """CLIP's evaluation preprocessing: shorter side to `size` (bicubic), centre crop, normalise."""
    import torch

    width, height = image.size
    scale = size / min(width, height)
    image = image.resize((max(size, round(width * scale)), max(size, round(height * scale))), Image.BICUBIC)
    left, top = (image.width - size) // 2, (image.height - size) // 2
    image = image.crop((left, top, left + size, top + size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class Embedder:
    key = ""
    record: dict[str, Any] = {}

    def embed(self, images: list[Image.Image]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


def _device(name: str | None) -> str:
    return name or os.environ.get("ASSETFURNACE_STYLE_DEVICE", "cpu")


class CsdEmbedder(Embedder):
    key = "csd"

    def __init__(self, device: str | None = None, hub: Path | None = None):
        import torch

        from image_to_3dlab.style.clip_vit import CSDStyle, csd_state_dict

        path = models.require(models.CSD, hub)["pytorch_model.bin"]
        # The checkpoint also stores its training arguments (an argparse.Namespace) and a few
        # numpy scalars. Allowing exactly those types keeps weights_only: no code from the file runs.
        allowed = [argparse.Namespace, np.dtype, np.core.multiarray.scalar,
                   type(np.dtype("float64")), type(np.dtype("float32")), type(np.dtype("int64"))]
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.device = _device(device)
        self.model = CSDStyle()
        self.model.load_state_dict(csd_state_dict(checkpoint), strict=True)
        self.model.eval().to(self.device)
        self.record = {"id": "csd-vit-l", "repo": models.CSD.repo, "revision": models.CSD.revision,
                       "license": models.CSD.license}

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        batch = torch.stack([clip_tensor(i) for i in images]).to(self.device)
        with torch.inference_mode():
            return self.model(batch).float().cpu().numpy()


class SiglipEmbedder(Embedder):
    key = "siglip2"

    def __init__(self, device: str | None = None, hub: Path | None = None):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        folder = models.require(models.SIGLIP2, hub)["model.safetensors"].parent
        self.device = _device(device)
        self.processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True)
        self.model = SiglipVisionModel.from_pretrained(folder, local_files_only=True, dtype=torch.float32)
        self.model.eval().to(self.device)
        self.record = {"id": "siglip2-base-patch16-224", "repo": models.SIGLIP2.repo,
                       "revision": models.SIGLIP2.revision, "license": models.SIGLIP2.license}

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            pooled = self.model(**inputs).pooler_output
        return torch.nn.functional.normalize(pooled, dim=1).float().cpu().numpy()


EMBEDDERS = {"csd": CsdEmbedder, "siglip2": SiglipEmbedder}


def load_embedder(key: str, device: str | None = None) -> Embedder:
    if key not in EMBEDDERS:
        raise ValueError(f"unknown embedding {key!r}; choose from {', '.join(EMBEDDERS)}")
    return EMBEDDERS[key](device=device)


def _read_cache(cache: Path | None) -> dict[str, np.ndarray]:
    if cache is None or not cache.is_file():
        return {}
    with np.load(cache) as stored:
        return {name: stored[name] for name in stored.files}


def _write_cache(cache: Path, vectors: dict[str, np.ndarray]) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=".emb-", suffix=".npz", dir=cache.parent)
    os.close(handle)
    try:
        np.savez(temp, **vectors)
        os.replace(temp, cache)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def embed_paths(embedder: Embedder, paths: list[Path], cache: Path | None = None,
                batch: int = 8) -> np.ndarray:
    """One unit-length row per picture, in order; pictures already in `cache` are not recomputed."""
    hashes = [sha256_file(p) for p in paths]
    known = _read_cache(cache)
    todo = [(h, p) for h, p in dict(zip(hashes, paths)).items() if h not in known]
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        rows = embedder.embed([open_rgb(p) for _, p in chunk])
        for (digest, _), row in zip(chunk, rows):
            known[digest] = np.asarray(row, dtype=np.float32)
    if todo and cache is not None:
        _write_cache(cache, known)
    if not hashes:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([known[h] for h in hashes])
