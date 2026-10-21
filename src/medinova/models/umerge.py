"""U-MERGE head (proposal 6.2, 6.3).

    h_m  = normalize(P_m f_m)                              m in {I, T}
    a_m  = relevance head(h_m, q_m)                        trained by L_BCE only
    u_m  = sigmoid(uncertainty head(h_m, q_m))             supervised by L_unc (target s_m / 3)
    w_m  = softmax_m(a_m - lambda * u_m),  lambda = softplus(.) >= 0
    z    = w_I h_I + w_T h_T
    x    = [z; h_I * h_T; |h_I - h_T|; cos(h_I, h_T); u_I; u_T]
    p    = sigmoid(MLP(x))

Both modalities are always present when the model runs; missing input is a
hard abstention handled outside the model, so there is no modality mask.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .base import FusionBase, mlp


class UMerge(FusionBase):
    def __init__(
        self,
        dim_i: int,
        dim_t: int,
        q_dim_i: int,
        q_dim_t: int,
        d: int = 128,
        hidden: int = 256,
        dropout: float = 0.1,
        lambda_init: float = 1.0,
        use_gate: bool = True,
        use_agreement: bool = True,
        use_uncertainty: bool = True,
        stop_grad_uncertainty: bool = True,
        **_,
    ):
        super().__init__(q_dim_i, q_dim_t)
        self.use_gate, self.use_agreement = use_gate, use_agreement
        self.use_uncertainty, self.stop_grad_uncertainty = use_uncertainty, stop_grad_uncertainty

        self.proj_i, self.proj_t = nn.Linear(dim_i, d), nn.Linear(dim_t, d)
        small = max(hidden // 4, 16)
        if use_gate:
            self.rel_i, self.rel_t = mlp(d + q_dim_i, small, 1), mlp(d + q_dim_t, small, 1)
            # Inverse softplus so softplus(lambda_raw) == lambda_init at start.
            self.lambda_raw = nn.Parameter(torch.tensor(math.log(math.expm1(lambda_init))))
        if use_uncertainty:
            self.unc_i, self.unc_t = mlp(d + q_dim_i, small, 1), mlp(d + q_dim_t, small, 1)

        head_in = d + (2 * d + 1 if use_agreement else 0) + (2 if use_uncertainty else 0)
        self.head = mlp(head_in, hidden, 1, dropout)

    @property
    def lam(self) -> torch.Tensor:
        return F.softplus(self.lambda_raw)

    def forward(self, fi, qi, ft, qt) -> dict[str, torch.Tensor]:
        qi, qt = self.norm_q(qi, qt)
        hi = F.normalize(self.proj_i(fi), dim=-1)
        ht = F.normalize(self.proj_t(ft), dim=-1)
        xi, xt = torch.cat([hi, qi], -1), torch.cat([ht, qt], -1)
        out = {"h_i": hi, "h_t": ht}

        if self.use_uncertainty:
            ul_i, ul_t = self.unc_i(xi).squeeze(-1), self.unc_t(xt).squeeze(-1)
            u_i, u_t = torch.sigmoid(ul_i), torch.sigmoid(ul_t)
            out.update(u_logit_i=ul_i, u_logit_t=ul_t, u_i=u_i, u_t=u_t)
            if self.stop_grad_uncertainty:
                u_i, u_t = u_i.detach(), u_t.detach()
        else:
            u_i = u_t = torch.zeros(hi.shape[0], device=hi.device)

        if self.use_gate:
            a = torch.stack([self.rel_i(xi).squeeze(-1), self.rel_t(xt).squeeze(-1)], -1)
            w = torch.softmax(a - self.lam * torch.stack([u_i, u_t], -1), dim=-1)
        else:
            w = torch.full((hi.shape[0], 2), 0.5, device=hi.device)
        out.update(w_i=w[:, 0], w_t=w[:, 1])

        feats = [w[:, :1] * hi + w[:, 1:] * ht]
        if self.use_agreement:
            prod = hi * ht
            feats += [prod, (hi - ht).abs(), prod.sum(-1, keepdim=True)]
        if self.use_uncertainty:
            feats.append(torch.stack([u_i, u_t], -1))
        out["logit"] = self.head(torch.cat(feats, -1)).squeeze(-1)
        return out
