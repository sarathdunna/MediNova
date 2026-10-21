"""MediNova X research console API (proposal 8). Research and education only.

    uvicorn api.main:app --reload --port 8000

Endpoints
  GET  /health
  GET  /runs                     runs with their clean-test summary
  GET  /runs/{name}/metrics      full metrics.json
  POST /predict                  multipart: run, image (file), report (text)

Missing or unreadable input is a hard abstention, decided before the model runs.
"""

from __future__ import annotations

import io
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from medinova.evaluation.calibration import AbstentionRule, apply_temperature
from medinova.features.encoders import get_encoder
from medinova.features.quality import image_quality, text_quality
from medinova.training.train import load_run_model

RUNS_DIR = Path(os.environ.get("MEDINOVA_RUNS", "runs"))
DISCLAIMER = "Research prototype. Not for clinical use; output is not a diagnosis."

app = FastAPI(title="MediNova X research API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("MEDINOVA_CORS", "http://localhost:3000").split(","),
    allow_methods=["*"], allow_headers=["*"],
)


def _run_dir(name: str) -> Path:
    path = (RUNS_DIR / name).resolve()
    if RUNS_DIR.resolve() not in path.parents or not (path / "model.pt").exists():
        raise HTTPException(404, f"Unknown run {name!r}")
    return path


@lru_cache(maxsize=4)
def _load(name: str):
    run = _run_dir(name)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    cal = json.loads((run / "calibration.json").read_text(encoding="utf-8"))
    vocab_file = Path(cfg["paths"]["interim_dir"]) / "vocab.txt"
    vocab = set(vocab_file.read_text(encoding="utf-8").split()) if vocab_file.exists() else set()
    return cfg, cal, load_run_model(run, cfg), vocab


@lru_cache(maxsize=4)
def _encoder(name: str):
    return get_encoder(name)


@app.get("/health")
def health():
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.get("/runs")
def runs():
    out = []
    for d in sorted(RUNS_DIR.glob("*/metrics.json")):
        m = json.loads(d.read_text(encoding="utf-8"))
        out.append({"name": d.parent.name, "clean": m.get("clean", {}).get("calibrated", {})})
    return out


@app.get("/runs/{name}/metrics")
def run_metrics(name: str):
    return json.loads((_run_dir(name) / "metrics.json").read_text(encoding="utf-8"))


def _abstain(reason: str):
    return {"abstain": True, "reason": reason, "p_match": None, "disclaimer": DISCLAIMER}


@app.post("/predict")
async def predict(
    run: str = Form(...),
    report: str = Form(""),
    image: UploadFile | None = File(None),
    coverage: float = Form(0.9),
):
    if image is None or not report.strip():
        return _abstain("missing_input")
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("L")
    except UnidentifiedImageError:
        return _abstain("unreadable_image")

    cfg, cal, model, vocab = _load(run)
    fi = _encoder(cfg["encoders"]["image"]).encode([img])
    ft = _encoder(cfg["encoders"]["text"]).encode([report])
    qi, qt = image_quality(img)[None], text_quality(report, vocab)[None]
    with torch.no_grad():
        out = model(*(torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in (fi, qi, ft, qt)))
    p = float(apply_temperature(out["logit"].numpy(), cal["temperature"])[0])
    u_i = float(out["u_i"][0]) if "u_i" in out else None
    u_t = float(out["u_t"][0]) if "u_t" in out else None
    rule = min(cal["rules"], key=lambda r: abs(r["target_coverage"] - coverage))
    reason = AbstentionRule(**rule).reason(p, max(u_i or 0.0, u_t or 0.0))
    return {
        "abstain": reason is not None, "reason": reason, "p_match": p,
        "u_image": u_i, "u_text": u_t,
        "w_image": float(out["w_i"][0]) if "w_i" in out else None,
        "w_text": float(out["w_t"][0]) if "w_t" in out else None,
        "rule": rule, "disclaimer": DISCLAIMER,
    }
