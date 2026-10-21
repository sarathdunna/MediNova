import pytest
import torch

from medinova.models import MODELS, build_model
from medinova.models.losses import total_loss

CFG = {"d": 32, "hidden": 64, "dropout": 0.0, "lambda_init": 1.0, "use_gate": True,
       "use_agreement": True, "use_uncertainty": True, "stop_grad_uncertainty": True}


def _inputs(b=8):
    return torch.randn(b, 20), torch.randn(b, 4), torch.randn(b, 12), torch.randn(b, 4)


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_runs_and_backprops(name):
    model = build_model(name, 20, 12, 4, 4, CFG)
    out = model(*_inputs())
    assert out["logit"].shape == (8,)
    y, s = torch.randint(0, 2, (8,)).float(), torch.randint(0, 4, (8,)).float()
    loss, _ = total_loss(out, y, s, s, {"alpha": 1.0, "beta": 0.5, "contrast_temperature": 0.07})
    loss.backward()


def test_umerge_gate_weights_sum_to_one_and_lambda_nonnegative():
    model = build_model("umerge", 20, 12, 4, 4, CFG)
    out = model(*_inputs())
    assert torch.allclose(out["w_i"] + out["w_t"], torch.ones(8))
    assert model.lam.item() == pytest.approx(1.0, rel=1e-5)
    assert ((out["u_i"] >= 0) & (out["u_i"] <= 1)).all()


def test_stop_grad_keeps_bce_out_of_uncertainty_head():
    model = build_model("umerge", 20, 12, 4, 4, CFG)
    out = model(*_inputs())
    torch.nn.functional.binary_cross_entropy_with_logits(out["logit"], torch.ones(8)).backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.unc_i.parameters())


def test_ablation_flags_change_head_input():
    full = build_model("umerge", 20, 12, 4, 4, CFG)
    bare = build_model("umerge_no_agreement", 20, 12, 4, 4, CFG)
    assert full.head[0].in_features == 32 + 65 + 2
    assert bare.head[0].in_features == 32 + 2
