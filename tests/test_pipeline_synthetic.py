"""End-to-end train -> calibrate -> evaluate on synthetic caches (no real data needed)."""

import json

from medinova.training.train import calibrate_run, evaluate_run, train_run


def test_umerge_pipeline_end_to_end(synthetic, tmp_path):
    cfg, pairs, images, texts = synthetic
    run = train_run(cfg, "umerge", pairs, images, texts, tmp_path / "umerge", seed=0)
    cal = calibrate_run(cfg, run, pairs, images, texts)
    assert cal["temperature"] > 0 and len(cal["rules"]) == 4
    res = evaluate_run(cfg, run, pairs, images, texts)

    assert res["clean"]["calibrated"]["auroc"] > 0.8
    # Synthetic corruption noise scales with severity, so u_m must track it (O3).
    assert res["uncertainty_validity_image"] > 0.5
    assert res["uncertainty_validity_text"] > 0.5
    # Severe corruption hurts; the model should know it is less reliable there.
    assert res["severe_image_blur"]["mean_u_i"] > res["clean"]["mean_u_i"]
    for name in ("metrics.json", "calibration.json", "test_predictions.csv", "model.pt"):
        assert (run / name).exists()
    json.loads((run / "metrics.json").read_text())


def test_baseline_pipeline_runs(synthetic, tmp_path):
    cfg, pairs, images, texts = synthetic
    run = train_run(cfg, "concat_mlp", pairs, images, texts, tmp_path / "concat", seed=0)
    calibrate_run(cfg, run, pairs, images, texts)
    res = evaluate_run(cfg, run, pairs, images, texts)
    assert "uncertainty_validity_image" not in res
    # Plain concatenation has no agreement features; it is expected to be weak at matching.
    assert 0.0 <= res["clean"]["calibrated"]["auroc"] <= 1.0
