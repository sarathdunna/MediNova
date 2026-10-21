"""Four-way, group-disjoint splits (proposal 5.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "calibration", "test")


def assign_splits(
    manifest: pd.DataFrame, fractions: dict[str, float], seed: int, group_col: str = "group_id"
) -> pd.Series:
    """Assign each row to a split so that no group crosses splits.

    Groups are shuffled with `seed` and cut by cumulative fraction of groups.
    """
    total = sum(fractions[s] for s in SPLITS)
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {total}")
    groups = np.random.default_rng(seed).permutation(manifest[group_col].drop_duplicates().to_numpy())
    bounds = np.cumsum([fractions[s] for s in SPLITS]) * len(groups)
    cut = np.searchsorted(bounds, np.arange(len(groups)), side="right")
    group_to_split = {g: SPLITS[min(i, len(SPLITS) - 1)] for g, i in zip(groups, cut)}
    return manifest[group_col].map(group_to_split).rename("split")
