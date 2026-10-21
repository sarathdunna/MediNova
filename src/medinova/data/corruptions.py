"""Controlled image and text corruptions with severities 0-3 (proposal 6.3).

Severity 0 is always the identity. Seen types are used in training; unseen
types are held back to test whether u_m generalises beyond memorised corruptions.
Negation flips and other meaning-changing edits are deliberately absent: those
make a different report, which is a negative, not a corruption.
"""

from __future__ import annotations

import io
import random
import re

import numpy as np
from PIL import Image, ImageFilter

# --- images (PIL, grayscale "L") ------------------------------------------------

_IMAGE_LEVELS = {
    "blur": [0.0, 1.0, 2.5, 5.0],          # Gaussian radius, px at 224
    "noise": [0.0, 0.03, 0.07, 0.15],      # std on [0, 1] intensities
    "contrast": [1.0, 0.7, 0.45, 0.25],    # contrast factor around the mean
    "jpeg": [100, 40, 15, 5],              # JPEG quality
    "exposure": [0.0, 0.12, 0.25, 0.4],    # additive shift on [0, 1]
}


def corrupt_image(img: Image.Image, kind: str, severity: int, seed: int = 0) -> Image.Image:
    if severity == 0:
        return img
    level = _IMAGE_LEVELS[kind][severity]
    img = img.convert("L")
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=level))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(level))
        buf.seek(0)
        return Image.open(buf).convert("L")
    x = np.asarray(img, dtype=np.float32) / 255.0
    if kind == "noise":
        x = x + np.random.default_rng(seed).normal(0.0, level, x.shape)
    elif kind == "contrast":
        x = (x - x.mean()) * level + x.mean()
    elif kind == "exposure":
        x = x + level
    else:
        raise KeyError(kind)
    return Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8), mode="L")


# --- text ---------------------------------------------------------------------

_TEXT_LEVELS = {
    "typo": [0.0, 0.02, 0.06, 0.12],         # share of characters edited
    "token_delete": [0.0, 0.1, 0.25, 0.45],  # share of tokens deleted
    "truncate": [1.0, 0.75, 0.5, 0.25],      # share of sentences kept
    "shuffle": [0.0, 0.2, 0.5, 1.0],         # share of sentences word-shuffled
}
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def corrupt_text(text: str, kind: str, severity: int, seed: int = 0) -> str:
    if severity == 0:
        return text
    level = _TEXT_LEVELS[kind][severity]
    rng = random.Random(seed)
    if kind == "typo":
        chars = list(text)
        letters = [i for i, c in enumerate(chars) if c.isalpha()]
        for i in rng.sample(letters, k=int(len(letters) * level)):
            chars[i] = rng.choice(_ALPHABET)
        return "".join(chars)
    if kind == "token_delete":
        tokens = text.split()
        kept = [t for t in tokens if rng.random() >= level]
        return " ".join(kept or tokens[:1])
    sentences = [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]
    if kind == "truncate":
        return " ".join(sentences[: max(1, int(round(len(sentences) * level)))])
    if kind == "shuffle":
        out = []
        for s in sentences:
            if rng.random() < level:
                words = s.split()
                rng.shuffle(words)
                s = " ".join(words)
            out.append(s)
        return " ".join(out)
    raise KeyError(kind)


IMAGE_KINDS = tuple(_IMAGE_LEVELS)
TEXT_KINDS = tuple(_TEXT_LEVELS)


def variant_name(kind: str, severity: int) -> str:
    return "clean" if severity == 0 else f"{kind}_s{severity}"
