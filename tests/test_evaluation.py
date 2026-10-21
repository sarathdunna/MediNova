import numpy as np

from medinova.evaluation.calibration import apply_temperature, fit_temperature, select_rule
from medinova.evaluation.metrics import ece, paired_bootstrap, risk_coverage, summarize


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 2, 5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-true_logit))).astype(int)
    T = fit_temperature(true_logit * 3, y)  # model is 3x overconfident
    assert 2.5 < T < 3.5
    assert ece(apply_temperature(true_logit * 3, T), y) < ece(1 / (1 + np.exp(-true_logit * 3)), y)


def test_perfect_ranking_has_zero_aurc():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.2, 0.8, 0.99])
    assert risk_coverage(p, y)[2] == 0
    assert summarize(p, y)["auroc"] == 1


def test_select_rule_meets_coverage_and_uses_uncertainty():
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.integers(0, 2, n)
    u = rng.random(n)
    # predictions are right when u is low, random when u is high
    p = np.where(u < 0.7, np.where(y == 1, 0.8, 0.2), rng.random(n))
    rule = select_rule(p, u, y, target_coverage=0.7)
    keep = rule.keep(p, u)
    assert keep.mean() >= 0.7
    assert rule.tau < 1.0
    assert ((p[keep] >= 0.5) != y[keep]).mean() < ((p >= 0.5) != y).mean()


def test_paired_bootstrap_ci_contains_diff():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    a, b = np.clip(y + rng.normal(0, .4, 300), 0, 1), rng.random(300)
    from sklearn.metrics import roc_auc_score
    r = paired_bootstrap(roc_auc_score, y, a, b, n=200)
    assert r["ci_low"] <= r["diff"] <= r["ci_high"] and r["ci_low"] > 0
