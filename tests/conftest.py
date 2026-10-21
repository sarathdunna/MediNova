"""Synthetic embedding caches that mimic the real layout, so the pipeline runs without data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medinova.config import load_config
from medinova.data.splits import assign_splits
from medinova.features.cache import ModalityStore


def _store(latent, kinds, rng, dim, noise_scale, q_dim=4):
    n = latent.shape[0]
    mix = rng.normal(size=(latent.shape[1], dim))
    clean = latent @ mix
    variants, embs, quals = ["clean"], [clean], [rng.normal(size=(n, q_dim)) * 0.1]
    for k in kinds:
        for s in (1, 2, 3):
            variants.append(f"{k}_s{s}")
            embs.append(clean + rng.normal(size=clean.shape) * noise_scale * s)
            q = rng.normal(size=(n, q_dim)) * 0.1
            q[:, 0] += s  # quality feature tracks severity
            quals.append(q)
    return ModalityStore(variants, np.array([f"S{i}" for i in range(n)]),
                         np.stack(embs).astype(np.float32), np.stack(quals).astype(np.float32))


@pytest.fixture(scope="session")
def synthetic():
    rng = np.random.default_rng(0)
    n = 600
    latent = rng.normal(size=(n, 16))
    images = _store(latent, ["blur", "noise", "contrast", "jpeg", "exposure"], rng, 64, 1.5)
    texts = _store(latent, ["typo", "token_delete", "truncate", "shuffle"], rng, 48, 1.5)
    manifest = pd.DataFrame({"study_id": images.ids, "group_id": images.ids})
    cfg = load_config(overrides={"train.epochs": 15, "train.batch_size": 128,
                                 "evaluation.bootstrap": 50})
    manifest["split"] = assign_splits(manifest, cfg["splits"], 0)
    rows = []
    for split, df in manifest.groupby("split"):
        ids = df.study_id.to_numpy()
        shuffled = np.roll(ids, 1)
        for a, b in zip(ids, shuffled):
            rows += [(a, a, 1, False, split), (a, b, 0, rng.random() < 0.5, split)]
    pairs = pd.DataFrame(rows, columns=["image_study", "report_study", "label", "is_hard", "split"])
    return cfg, pairs, images, texts
