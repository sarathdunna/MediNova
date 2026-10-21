"""Finding-aware pair construction (proposal 5.2).

For each study in a split: one positive pair (its image, its report) and one
negative pair (its image, another report from the same split) such that

  * the negative report is not in the anchor's template group (near-duplicate text),
  * at most one report per template group is eligible for a given anchor,
  * the negative's finding labels definitely conflict with the anchor's,
  * with probability `hard_fraction` the negative comes from the `hard_top_k`
    lexically closest eligible reports ("hard negatives").
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from .labels import definite_conflict_matrix


def normalize_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def tfidf_similarity(texts: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    x = vec.fit_transform([normalize_text(t) for t in texts])
    return (x @ x.T).toarray()


def template_groups(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of the `sim >= threshold` graph."""
    _, groups = connected_components(sim >= threshold, directed=False)
    return groups


def build_pairs(
    split_df: pd.DataFrame,
    labels: pd.DataFrame,
    seed: int,
    dedup_threshold: float = 0.95,
    hard_fraction: float = 0.5,
    hard_top_k: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """split_df: rows of one split with study_id, report_text. labels: indexed by study_id.

    Returns (pairs, stats). Pairs columns: image_study, report_study, label, is_hard, text_sim.
    """
    df = split_df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    sim = tfidf_similarity(df["report_text"].tolist())
    groups = template_groups(sim, dedup_threshold)
    lab = labels.loc[df["study_id"]].to_numpy()

    rows, no_negative = [], 0
    for i, study in enumerate(df["study_id"]):
        rows.append((study, study, 1, False, 1.0))
        eligible = np.flatnonzero((groups != groups[i]) & definite_conflict_matrix(lab[i], lab))
        if eligible.size == 0:
            no_negative += 1
            continue
        # One random member per template group.
        eligible = rng.permutation(eligible)
        _, first = np.unique(groups[eligible], return_index=True)
        eligible = eligible[first]
        hard_pool = eligible[np.argsort(-sim[i, eligible])[:hard_top_k]]
        is_hard = rng.random() < hard_fraction
        j = rng.choice(hard_pool if is_hard else eligible)
        rows.append((study, df["study_id"].iat[j], 0, bool(is_hard), float(sim[i, j])))

    pairs = pd.DataFrame(rows, columns=["image_study", "report_study", "label", "is_hard", "text_sim"])
    stats = {
        "studies": len(df),
        "template_groups": int(groups.max() + 1) if len(groups) else 0,
        "positives": int((pairs.label == 1).sum()),
        "negatives": int((pairs.label == 0).sum()),
        "hard_negatives": int(pairs.is_hard.sum()),
        "anchors_without_valid_negative": no_negative,
    }
    return pairs, stats
