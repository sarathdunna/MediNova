"""Task-matched baselines (proposal 7). Same inputs and outputs as U-MERGE.

(a) zero-shot cosine in a pre-aligned space is not a trainable model: see
    `medinova.evaluation.metrics.zero_shot_cosine`.
(f) quality gating without uncertainty is U-MERGE with use_uncertainty=False.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import FusionBase, mlp


class _Projected(FusionBase):
    def __init__(self, dim_i, dim_t, q_dim_i, q_dim_t, d=128):
        super().__init__(q_dim_i, q_dim_t)
        self.proj_i, self.proj_t = nn.Linear(dim_i, d), nn.Linear(dim_t, d)

    def project(self, fi, ft):
        return F.normalize(self.proj_i(fi), dim=-1), F.normalize(self.proj_t(ft), dim=-1)


class ProjectedCosine(_Projected):
    """(b) logit = s * cos(h_I, h_T) + b after trained projections."""

    def __init__(self, dim_i, dim_t, q_dim_i, q_dim_t, d=128, **_):
        super().__init__(dim_i, dim_t, q_dim_i, q_dim_t, d)
        self.scale, self.bias = nn.Parameter(torch.tensor(10.0)), nn.Parameter(torch.tensor(0.0))

    def forward(self, fi, qi, ft, qt):
        hi, ht = self.project(fi, ft)
        return {"logit": self.scale * (hi * ht).sum(-1) + self.bias, "h_i": hi, "h_t": ht}


class ConcatMLP(_Projected):
    """(c) MLP([h_I; h_T])."""

    def __init__(self, dim_i, dim_t, q_dim_i, q_dim_t, d=128, hidden=256, dropout=0.1, **_):
        super().__init__(dim_i, dim_t, q_dim_i, q_dim_t, d)
        self.head = mlp(2 * d, hidden, 1, dropout)

    def forward(self, fi, qi, ft, qt):
        hi, ht = self.project(fi, ft)
        return {"logit": self.head(torch.cat([hi, ht], -1)).squeeze(-1), "h_i": hi, "h_t": ht}


class BilinearScore(_Projected):
    """(d) score-level fusion: logit = h_I^T W h_T + b."""

    def __init__(self, dim_i, dim_t, q_dim_i, q_dim_t, d=128, **_):
        super().__init__(dim_i, dim_t, q_dim_i, q_dim_t, d)
        self.W = nn.Parameter(torch.eye(d) * 10.0)
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, fi, qi, ft, qt):
        hi, ht = self.project(fi, ft)
        logit = torch.einsum("bd,de,be->b", hi, self.W, ht) + self.bias
        return {"logit": logit, "h_i": hi, "h_t": ht}


class CrossAttentionFusion(_Projected):
    """(e) lightweight attention: each embedding split into k tokens, one transformer layer."""

    def __init__(self, dim_i, dim_t, q_dim_i, q_dim_t, d=128, hidden=256, dropout=0.1,
                 tokens=8, **_):
        super().__init__(dim_i, dim_t, q_dim_i, q_dim_t, d)
        self.k, self.e = tokens, d // tokens
        self.modality = nn.Parameter(torch.zeros(2, 1, self.e))
        self.layer = nn.TransformerEncoderLayer(
            self.e, nhead=2, dim_feedforward=4 * self.e, dropout=dropout, batch_first=True
        )
        self.head = mlp(self.e, hidden, 1, dropout)

    def forward(self, fi, qi, ft, qt):
        hi, ht = self.project(fi, ft)
        b = hi.shape[0]
        ti = hi.view(b, self.k, self.e) + self.modality[0]
        tt = ht.view(b, self.k, self.e) + self.modality[1]
        pooled = self.layer(torch.cat([ti, tt], 1)).mean(1)
        return {"logit": self.head(pooled).squeeze(-1), "h_i": hi, "h_t": ht}
