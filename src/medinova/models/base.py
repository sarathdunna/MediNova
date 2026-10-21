from __future__ import annotations

import numpy as np
import torch
from torch import nn


def mlp(d_in: int, hidden: int, d_out: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_out)
    )


class FusionBase(nn.Module):
    """Holds quality-feature standardisation so training and serving agree."""

    def __init__(self, q_dim_i: int, q_dim_t: int):
        super().__init__()
        self.register_buffer("q_mean_i", torch.zeros(q_dim_i))
        self.register_buffer("q_std_i", torch.ones(q_dim_i))
        self.register_buffer("q_mean_t", torch.zeros(q_dim_t))
        self.register_buffer("q_std_t", torch.ones(q_dim_t))

    def set_quality_stats(self, qi: np.ndarray, qt: np.ndarray) -> None:
        """Fit on clean training-split quality features only."""
        for name, q in (("i", qi), ("t", qt)):
            q = torch.as_tensor(q, dtype=torch.float32)
            getattr(self, f"q_mean_{name}").copy_(q.mean(0))
            getattr(self, f"q_std_{name}").copy_(q.std(0).clamp_min(1e-6))

    def norm_q(self, qi: torch.Tensor, qt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (qi - self.q_mean_i) / self.q_std_i, (qt - self.q_mean_t) / self.q_std_t
