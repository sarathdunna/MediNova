"""Discrimination, calibration and selective-prediction metrics (proposal 7)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins on P(match)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def confidence(p: np.ndarray) -> np.ndarray:
    return np.abs(p - 0.5)


def risk_coverage(p: np.ndarray, y: np.ndarray, conf: np.ndarray | None = None):
    """Selective risk (error rate) as the least-confident items are rejected.

    Returns (coverage, risk, aurc). Lower AURC is better.
    """
    conf = confidence(p) if conf is None else conf
    order = np.argsort(-conf, kind="stable")
    errors = ((p[order] >= 0.5).astype(int) != y[order]).astype(float)
    n = np.arange(1, len(y) + 1)
    coverage, risk = n / len(y), np.cumsum(errors) / n
    return coverage, risk, float(_trapezoid(risk, coverage))


def selective_risk(p: np.ndarray, y: np.ndarray, keep: np.ndarray) -> tuple[float, float]:
    """(risk on kept items, achieved coverage)."""
    if keep.sum() == 0:
        return float("nan"), 0.0
    err = ((p[keep] >= 0.5).astype(int) != y[keep]).mean()
    return float(err), float(keep.mean())


def summarize(p: np.ndarray, y: np.ndarray, bins: int = 15) -> dict[str, float]:
    y = y.astype(int)
    two_classes = len(np.unique(y)) == 2
    return {
        "n": len(y),
        "auroc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "auprc": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(y, p >= 0.5)) if two_classes else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece(p, y, bins),
        "aurc": risk_coverage(p, y)[2],
    }


def uncertainty_validity(u: np.ndarray, severity: np.ndarray) -> float:
    """Spearman rho between predicted u_m and applied corruption severity (O3)."""
    return float(spearmanr(u, severity).statistic)


def paired_bootstrap(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n: int = 1000, seed: int = 0,
) -> dict[str, float]:
    """95% CI of metric(A) - metric(B) on the same test items."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        diffs.append(metric(y[i], p_a[i]) - metric(y[i], p_b[i]))
    diffs = np.array(diffs)
    return {"diff": float(metric(y, p_a) - metric(y, p_b)),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5))}


def zero_shot_cosine(fi: np.ndarray, ft: np.ndarray) -> np.ndarray:
    """Baseline (a): cosine in a pre-aligned space (e.g. BiomedCLIP), mapped to [0, 1]."""
    fi = fi / np.linalg.norm(fi, axis=1, keepdims=True)
    ft = ft / np.linalg.norm(ft, axis=1, keepdims=True)
    return (np.sum(fi * ft, axis=1) + 1) / 2
