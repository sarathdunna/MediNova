"""Temperature scaling and the abstention rule (proposal 6.5).

Everything here is fitted on the calibration split only.

    p_cal   = sigmoid(logit / T)
    abstain = |p_cal - 0.5| < delta  OR  max(u_I, u_T) > tau
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 200) -> float:
    """Minimise NLL over log T with LBFGS; returns T > 0."""
    z = torch.as_tensor(logits, dtype=torch.float64)
    t = torch.as_tensor(y, dtype=torch.float64)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits / T))


@dataclass
class AbstentionRule:
    target_coverage: float
    delta: float
    tau: float

    def keep(self, p_cal: np.ndarray, u_max: np.ndarray) -> np.ndarray:
        return (np.abs(p_cal - 0.5) >= self.delta) & (u_max <= self.tau)

    def reason(self, p_cal: float, u_max: float) -> str | None:
        if u_max > self.tau:
            return "unreliable_input"
        if abs(p_cal - 0.5) < self.delta:
            return "low_confidence"
        return None

    def to_dict(self) -> dict:
        return asdict(self)


def select_rule(
    p_cal: np.ndarray, u_max: np.ndarray, y: np.ndarray, target_coverage: float, grid: int = 41
) -> AbstentionRule:
    """Lowest selective risk among (delta, tau) whose coverage >= target; ties -> more coverage."""
    deltas = np.unique(np.quantile(np.abs(p_cal - 0.5), np.linspace(0, 1, grid)))
    taus = np.unique(np.append(np.quantile(u_max, np.linspace(0, 1, grid)), np.inf))
    deltas = np.append(0.0, deltas)
    best, best_key = AbstentionRule(target_coverage, 0.0, float("inf")), (np.inf, 0.0)
    correct = (p_cal >= 0.5).astype(int) == y
    for d in deltas:
        conf_ok = np.abs(p_cal - 0.5) >= d
        for t in taus:
            keep = conf_ok & (u_max <= t)
            cov = keep.mean()
            if cov < target_coverage or keep.sum() == 0:
                continue
            key = (1 - correct[keep].mean(), -cov)
            if key < best_key:
                best, best_key = AbstentionRule(target_coverage, float(d), float(t)), key
    return best
