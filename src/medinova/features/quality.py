"""Cheap, measurable input-quality features q_m (proposal 6.2)."""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

IMAGE_QUALITY_NAMES = ["laplacian_var", "noise_mad", "contrast_std", "mean_intensity"]
TEXT_QUALITY_NAMES = ["log_tokens", "oov_rate", "n_sentences", "nonalpha_rate"]


def image_quality(img: Image.Image) -> np.ndarray:
    x = np.asarray(img.convert("L").resize((224, 224)), dtype=np.float32) / 255.0
    lap = (
        -4 * x[1:-1, 1:-1] + x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:]
    )
    # Median absolute deviation of the Laplacian: robust noise estimate.
    noise = np.median(np.abs(lap - np.median(lap))) / 0.6745
    return np.array([np.log1p(lap.var() * 1e4), noise, x.std(), x.mean()], dtype=np.float32)


_TOKEN = re.compile(r"[a-z]+")


def build_vocab(texts: list[str], min_count: int = 2) -> set[str]:
    """Vocabulary from training-split reports only (avoid leakage)."""
    counts: dict[str, int] = {}
    for t in texts:
        for w in _TOKEN.findall(t.lower()):
            counts[w] = counts.get(w, 0) + 1
    return {w for w, c in counts.items() if c >= min_count}


def text_quality(text: str, vocab: set[str]) -> np.ndarray:
    words = _TOKEN.findall(text.lower())
    oov = sum(w not in vocab for w in words) / max(len(words), 1)
    sentences = len([s for s in re.split(r"[.;]", text) if s.strip()])
    nonalpha = sum(not (c.isalpha() or c.isspace()) for c in text) / max(len(text), 1)
    return np.array([np.log1p(len(words)), oov, sentences, nonalpha], dtype=np.float32)
