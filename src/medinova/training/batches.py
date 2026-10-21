"""Sample training / evaluation batches straight from stacked cached variants.

Training: each modality of each pair is independently left clean (prob p_clean)
or replaced by a random seen corruption at a random severity 1-3; the severity is
the target for u_m. Evaluation: a fixed variant per modality (a "condition").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..features.cache import ModalityStore


def _parse(variant: str) -> tuple[str, int]:
    if variant == "clean":
        return "clean", 0
    kind, sev = variant.rsplit("_s", 1)
    return kind, int(sev)


class PairSampler:
    def __init__(
        self,
        pairs: pd.DataFrame,
        images: ModalityStore,
        texts: ModalityStore,
        seen_image: list[str],
        seen_text: list[str],
        p_clean: float = 0.4,
    ):
        self.images, self.texts = images, texts
        self.ri = images.rows(pairs["image_study"])
        self.rt = texts.rows(pairs["report_study"])
        self.y = pairs["label"].to_numpy(np.float32)
        self.p_clean = p_clean
        self.seen_i = self._seen_indices(images, seen_image)
        self.seen_t = self._seen_indices(texts, seen_text)

    @staticmethod
    def _seen_indices(store: ModalityStore, kinds: list[str]) -> np.ndarray:
        idx = [i for i, v in enumerate(store.variants) if _parse(v)[0] in kinds]
        if not idx:
            raise ValueError(f"No cached variants for seen kinds {kinds}: {store.variants}")
        return np.array(idx)

    def __len__(self) -> int:
        return len(self.y)

    def _sample_variants(self, n, seen, rng):
        v = rng.choice(seen, size=n)
        return np.where(rng.random(n) < self.p_clean, 0, v)

    def _gather(self, store: ModalityStore, v: np.ndarray, rows: np.ndarray):
        sev = np.array([_parse(store.variants[i])[1] for i in v], dtype=np.float32)
        return (torch.from_numpy(store.emb[v, rows]), torch.from_numpy(store.quality[v, rows]),
                torch.from_numpy(sev))

    def _mild(self, store: ModalityStore, v: np.ndarray) -> np.ndarray:
        names = [store.variants[i] for i in v]
        out = []
        for name in names:
            kind, _ = _parse(name)
            out.append(store.variants.index(f"{kind}_s1") if kind != "clean" else 0)
        return np.array(out)

    def train_batches(self, batch_size: int, rng: np.random.Generator, with_mild: bool = False):
        order = rng.permutation(len(self))
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            vi = self._sample_variants(len(idx), self.seen_i, rng)
            vt = self._sample_variants(len(idx), self.seen_t, rng)
            fi, qi, si = self._gather(self.images, vi, self.ri[idx])
            ft, qt, st = self._gather(self.texts, vt, self.rt[idx])
            batch = dict(fi=fi, qi=qi, si=si, ft=ft, qt=qt, st=st, y=torch.from_numpy(self.y[idx]))
            if with_mild:
                clean = np.zeros(len(idx), dtype=int)
                mi = self._mild(self.images, rng.choice(self.seen_i, size=len(idx)))
                mt = self._mild(self.texts, rng.choice(self.seen_t, size=len(idx)))
                batch["clean"] = (*self._gather(self.images, clean, self.ri[idx])[:2],
                                  *self._gather(self.texts, clean, self.rt[idx])[:2])
                batch["mild"] = (*self._gather(self.images, mi, self.ri[idx])[:2],
                                 *self._gather(self.texts, mt, self.rt[idx])[:2])
            yield batch

    def condition(self, image_variant: str = "clean", text_variant: str = "clean") -> dict:
        """Whole split under one fixed variant per modality."""
        vi = np.full(len(self), self.images.variant_index(image_variant))
        vt = np.full(len(self), self.texts.variant_index(text_variant))
        fi, qi, si = self._gather(self.images, vi, self.ri)
        ft, qt, st = self._gather(self.texts, vt, self.rt)
        return dict(fi=fi, qi=qi, si=si, ft=ft, qt=qt, st=st, y=torch.from_numpy(self.y))
