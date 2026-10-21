"""U-MERGE and baselines. Every model maps (fi, qi, ft, qt) -> {"logit", ...}."""

from __future__ import annotations

from .baselines import BilinearScore, ConcatMLP, CrossAttentionFusion, ProjectedCosine
from .umerge import UMerge

# name -> (class, forced keyword overrides). Ablations of U-MERGE are just flags.
MODELS = {
    "umerge": (UMerge, {}),
    "quality_gate": (UMerge, {"use_uncertainty": False}),   # baseline (f)
    "umerge_no_gate": (UMerge, {"use_gate": False}),
    "umerge_no_agreement": (UMerge, {"use_agreement": False}),
    "projected_cosine": (ProjectedCosine, {}),               # baseline (b)
    "concat_mlp": (ConcatMLP, {}),                           # baseline (c)
    "bilinear": (BilinearScore, {}),                         # baseline (d)
    "cross_attention": (CrossAttentionFusion, {}),           # baseline (e)
}


def build_model(name: str, dim_i: int, dim_t: int, q_dim_i: int, q_dim_t: int, model_cfg: dict):
    cls, forced = MODELS[name]
    kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
    kwargs.update(forced)
    return cls(dim_i=dim_i, dim_t=dim_t, q_dim_i=q_dim_i, q_dim_t=q_dim_t, **kwargs)
