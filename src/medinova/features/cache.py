"""On-disk embedding cache: every item is encoded once per corruption variant.

Layout: <cache_dir>/<modality>/<encoder>/<variant>.npz with arrays
  ids (N,), emb (N, D), quality (N, Q)
Variants are "clean" or "<kind>_s<severity>". All variants of one modality share
the same id order, so they can be stacked into a (V, N, D) tensor for fast sampling.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..data.corruptions import corrupt_image, corrupt_text, variant_name
from .quality import image_quality, text_quality


def _item_seed(item_id: str, variant: str) -> int:
    return zlib.crc32(f"{item_id}|{variant}".encode())


IMAGE_SIZE = 224  # corruption levels in data/corruptions.py are defined at this resolution


def prepare_image(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Grayscale, centre-crop to square, resize. Applied before any corruption."""
    img = img.convert("L")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((size, size), Image.Resampling.BICUBIC)


def encode_modality(
    manifest: pd.DataFrame,
    modality: str,
    encoder,
    kinds: list[str],
    severities: list[int],
    out_dir: str | Path,
    image_dir: str | Path | None = None,
    vocab: set[str] | None = None,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Encode the clean variant plus every kind x severity. Returns a timing table."""
    out_dir = Path(out_dir) / modality / encoder.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = manifest["study_id"].astype(str).tolist()
    variants = [("clean", 0)] + [(k, s) for k in kinds for s in severities]
    timings = []
    prepared = {}
    if modality == "image":  # load and downsize each image once, reuse for every variant
        prepared = {
            i: prepare_image(Image.open(Path(image_dir) / f))
            for i, f in zip(ids, manifest["frontal_image"])
        }
    for kind, sev in variants:
        name = variant_name(kind, sev)
        target = out_dir / f"{name}.npz"
        if target.exists():
            continue
        start, embs, quals = time.perf_counter(), [], []
        for b in range(0, len(ids), batch_size):
            chunk = manifest.iloc[b : b + batch_size]
            if modality == "image":
                items = [
                    corrupt_image(prepared[i], kind, sev, _item_seed(i, name))
                    for i in chunk["study_id"].astype(str)
                ]
                quals += [image_quality(x) for x in items]
            else:
                items = [
                    corrupt_text(t, kind, sev, _item_seed(i, name))
                    for i, t in zip(chunk["study_id"].astype(str), chunk["report_text"])
                ]
                quals += [text_quality(x, vocab or set()) for x in items]
            embs.append(encoder.encode(items))
        np.savez_compressed(
            target, ids=np.array(ids), emb=np.concatenate(embs).astype(np.float32),
            quality=np.stack(quals).astype(np.float32),
        )
        secs = time.perf_counter() - start
        timings.append({"variant": name, "items": len(ids), "seconds": secs,
                        "sec_per_item": secs / max(len(ids), 1)})
        print(f"[{modality}/{encoder.name}] {name}: {secs:.1f}s")
    return pd.DataFrame(timings)


@dataclass
class ModalityStore:
    """All cached variants of one modality, stacked: emb (V, N, D), quality (V, N, Q)."""

    variants: list[str]
    ids: np.ndarray
    emb: np.ndarray
    quality: np.ndarray

    @classmethod
    def load(cls, root: str | Path) -> ModalityStore:
        files = sorted(Path(root).glob("*.npz"), key=lambda p: (p.stem != "clean", p.stem))
        if not files:
            raise FileNotFoundError(f"No cached variants in {root}")
        loaded = [np.load(f) for f in files]
        ids = loaded[0]["ids"]
        for f, d in zip(files, loaded):
            if not np.array_equal(d["ids"], ids):
                raise ValueError(f"{f} has a different id order from {files[0]}")
        return cls(
            variants=[f.stem for f in files],
            ids=ids,
            emb=np.stack([d["emb"] for d in loaded]),
            quality=np.stack([d["quality"] for d in loaded]),
        )

    def rows(self, item_ids) -> np.ndarray:
        index = {k: i for i, k in enumerate(self.ids.tolist())}
        return np.array([index[str(i)] for i in item_ids])

    def variant_index(self, name: str) -> int:
        return self.variants.index(name)

    @property
    def dim(self) -> int:
        return self.emb.shape[-1]

    @property
    def q_dim(self) -> int:
        return self.quality.shape[-1]
