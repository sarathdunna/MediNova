"""Training, calibration and test evaluation for one run.

A run directory holds: config.yaml, model.pt, history.csv, calibration.json,
metrics.json and test_predictions.csv. Every number in the thesis should be
regenerable from a run directory plus the cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..config import seed_everything
from ..evaluation.calibration import AbstentionRule, apply_temperature, fit_temperature, select_rule
from ..evaluation.metrics import selective_risk, summarize, uncertainty_validity
from ..features.cache import ModalityStore
from ..models import build_model
from ..models.losses import total_loss
from .batches import PairSampler


def make_samplers(cfg: dict, pairs: pd.DataFrame, images: ModalityStore, texts: ModalityStore):
    c = cfg["corruptions"]
    return {
        split: PairSampler(pairs[pairs.split == split], images, texts,
                           c["image_seen"], c["text_seen"], c["p_clean"])
        for split in ("train", "validation", "calibration", "test")
    }


@torch.no_grad()
def predict(model, batch: dict) -> dict[str, np.ndarray]:
    model.eval()
    out = model(batch["fi"], batch["qi"], batch["ft"], batch["qt"])
    keep = ("logit", "u_i", "u_t", "w_i", "w_t")
    return {k: out[k].numpy() for k in keep if k in out}


def train_run(cfg: dict, model_name: str, pairs: pd.DataFrame, images: ModalityStore,
              texts: ModalityStore, run_dir: str | Path, seed: int) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    samplers = make_samplers(cfg, pairs, images, texts)

    model = build_model(model_name, images.dim, texts.dim, images.q_dim, texts.q_dim, cfg["model"])
    train_rows_i = samplers["train"].ri
    train_rows_t = samplers["train"].rt
    model.set_quality_stats(images.quality[0, train_rows_i], texts.quality[0, train_rows_t])

    tc, lc = cfg["train"], cfg["loss"]
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    val = samplers["validation"].condition()
    best, best_state, stale, history = -np.inf, None, 0, []
    with_mild = bool(lc.get("gamma", 0))

    for epoch in range(tc["epochs"]):
        model.train()
        logs = []
        for b in samplers["train"].train_batches(tc["batch_size"], rng, with_mild):
            out = model(b["fi"], b["qi"], b["ft"], b["qt"])
            out_clean = model(*b["clean"]) if with_mild else None
            out_mild = model(*b["mild"]) if with_mild else None
            loss, parts = total_loss(out, b["y"], b["si"], b["st"], lc, out_mild, out_clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            logs.append(parts)
        p_val = 1 / (1 + np.exp(-predict(model, val)["logit"]))
        val_auroc = summarize(p_val, val["y"].numpy())["auroc"]
        row = {"epoch": epoch, "val_auroc": val_auroc,
               **{k: float(np.mean([log[k] for log in logs])) for k in logs[0]}}
        history.append(row)
        if val_auroc > best:
            best, stale = val_auroc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= tc["patience"]:
                break

    model.load_state_dict(best_state)
    torch.save({"model_name": model_name, "state_dict": model.state_dict(),
                "dims": [images.dim, texts.dim, images.q_dim, texts.q_dim]}, run_dir / "model.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({**cfg, "run": {"model": model_name, "seed": seed}}, f)
    return run_dir


def load_run_model(run_dir: str | Path, cfg: dict):
    ckpt = torch.load(Path(run_dir) / "model.pt", map_location="cpu")
    model = build_model(ckpt["model_name"], *ckpt["dims"], cfg["model"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def _u_max(pred: dict) -> np.ndarray:
    if "u_i" not in pred:
        return np.zeros_like(pred["logit"])
    return np.maximum(pred["u_i"], pred["u_t"])


def calibrate_run(cfg: dict, run_dir: str | Path, pairs, images, texts) -> dict:
    """Fit T and one abstention rule per target coverage on the calibration split."""
    model = load_run_model(run_dir, cfg)
    cal = make_samplers(cfg, pairs, images, texts)["calibration"]
    # Calibrate under the same mix of clean and corrupted inputs the model sees in use.
    rng = np.random.default_rng(cfg["seed"])
    batch = next(cal.train_batches(len(cal), rng))
    pred, y = predict(model, batch), batch["y"].numpy().astype(int)
    T = fit_temperature(pred["logit"], y)
    p_cal = apply_temperature(pred["logit"], T)
    rules = [select_rule(p_cal, _u_max(pred), y, c).to_dict()
             for c in cfg["abstention"]["coverages"]]
    result = {"temperature": T, "rules": rules}
    with open(Path(run_dir) / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def test_conditions(cfg: dict) -> dict[str, tuple[str, str]]:
    """Named (image_variant, text_variant) test conditions (proposal 7, test subsets)."""
    c = cfg["corruptions"]
    conds = {"clean": ("clean", "clean")}
    for s, label in ((1, "mild"), (3, "severe")):
        for k in c["image_seen"]:
            conds[f"{label}_image_{k}"] = (f"{k}_s{s}", "clean")
        for k in c["text_seen"]:
            conds[f"{label}_text_{k}"] = ("clean", f"{k}_s{s}")
    for k in c["image_unseen"]:
        conds[f"unseen_image_{k}"] = (f"{k}_s3", "clean")
    for k in c["text_unseen"]:
        conds[f"unseen_text_{k}"] = ("clean", f"{k}_s3")
    return conds


def evaluate_run(cfg: dict, run_dir: str | Path, pairs, images, texts) -> dict:
    """Single final pass on the test split, every condition, every coverage target."""
    run_dir = Path(run_dir)
    model = load_run_model(run_dir, cfg)
    with open(run_dir / "calibration.json", encoding="utf-8") as f:
        cal = json.load(f)
    test_pairs = pairs[pairs.split == "test"].reset_index(drop=True)
    sampler = make_samplers(cfg, pairs, images, texts)["test"]
    bins = cfg["evaluation"]["ece_bins"]
    results, frames = {}, []

    for name, (vi, vt) in test_conditions(cfg).items():
        if vi not in images.variants or vt not in texts.variants:
            continue
        batch = sampler.condition(vi, vt)
        pred, y = predict(model, batch), batch["y"].numpy().astype(int)
        p_raw = 1 / (1 + np.exp(-pred["logit"]))
        p_cal = apply_temperature(pred["logit"], cal["temperature"])
        res = {"uncalibrated": summarize(p_raw, y, bins), "calibrated": summarize(p_cal, y, bins)}
        if name == "clean":
            hard = test_pairs["is_hard"].to_numpy() | (y == 1)
            res["hard_negative_subset"] = summarize(p_cal[hard], y[hard], bins)
        for rule in cal["rules"]:
            r = AbstentionRule(**rule)
            risk, cov = selective_risk(p_cal, y, r.keep(p_cal, _u_max(pred)))
            res[f"coverage_target_{rule['target_coverage']}"] = {"risk": risk, "coverage": cov}
        if "u_i" in pred:
            res["mean_u_i"], res["mean_u_t"] = float(pred["u_i"].mean()), float(pred["u_t"].mean())
        results[name] = res
        frames.append(pd.DataFrame({"condition": name, "y": y, "p_cal": p_cal, "u_max": _u_max(pred),
                                    "image_study": test_pairs.image_study,
                                    "report_study": test_pairs.report_study}))

    # O3: does u_m track severity across all seen and unseen image/text variants?
    if "u_i" in predict(model, sampler.condition()):
        for modality, store in (("image", images), ("text", texts)):
            us, sevs = [], []
            for v in store.variants:
                b = sampler.condition(v, "clean") if modality == "image" else sampler.condition("clean", v)
                pred = predict(model, b)
                us.append(pred["u_i" if modality == "image" else "u_t"])
                sevs.append((b["si"] if modality == "image" else b["st"]).numpy())
            results[f"uncertainty_validity_{modality}"] = uncertainty_validity(
                np.concatenate(us), np.concatenate(sevs))

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    pd.concat(frames).to_csv(run_dir / "test_predictions.csv", index=False)
    return results
