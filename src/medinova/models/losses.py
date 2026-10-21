"""L = L_BCE + alpha L_unc + beta L_contrast + gamma L_consist (proposal 6.4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def uncertainty_loss(out: dict, sev_i: torch.Tensor, sev_t: torch.Tensor) -> torch.Tensor:
    """BCE between u_m and normalised corruption severity s_m / 3."""
    if "u_logit_i" not in out:
        return out["logit"].new_zeros(())
    return F.binary_cross_entropy_with_logits(out["u_logit_i"], sev_i / 3.0) + \
        F.binary_cross_entropy_with_logits(out["u_logit_t"], sev_t / 3.0)


def info_nce(h_i: torch.Tensor, h_t: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric InfoNCE over the positive pairs in the batch."""
    pos = y > 0.5
    if pos.sum() < 2:
        return h_i.new_zeros(())
    logits = h_i[pos] @ h_t[pos].T / tau
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def consistency_kl(logit_clean: torch.Tensor, logit_mild: torch.Tensor) -> torch.Tensor:
    """KL(Bern(p_clean) || Bern(p_mild)), clean side treated as the target."""
    p = torch.sigmoid(logit_clean).detach().clamp(1e-6, 1 - 1e-6)
    q = torch.sigmoid(logit_mild).clamp(1e-6, 1 - 1e-6)
    return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()


def total_loss(out, y, sev_i, sev_t, loss_cfg: dict, out_mild=None, out_clean=None):
    parts = {"bce": F.binary_cross_entropy_with_logits(out["logit"], y)}
    if loss_cfg.get("alpha", 0):
        parts["unc"] = loss_cfg["alpha"] * uncertainty_loss(out, sev_i, sev_t)
    if loss_cfg.get("beta", 0) and "h_i" in out:
        parts["contrast"] = loss_cfg["beta"] * info_nce(
            out["h_i"], out["h_t"], y, loss_cfg.get("contrast_temperature", 0.07)
        )
    if loss_cfg.get("gamma", 0) and out_mild is not None and out_clean is not None:
        parts["consist"] = loss_cfg["gamma"] * consistency_kl(out_clean["logit"], out_mild["logit"])
    return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}
