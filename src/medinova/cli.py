"""Command-line pipeline. Run steps in order:

    medinova manifest            parse IU X-Ray reports -> data/interim/manifest.csv
    medinova label [--csv F]     finding labels (CheXbert CSV, or dev keyword fallback)
    medinova split               four-way study-disjoint splits
    medinova pairs               finding-aware positives / negatives per split
    medinova encode --modality image|text [--encoder NAME]
    medinova train --model umerge --seed 13
    medinova calibrate --run runs/umerge_s13
    medinova evaluate  --run runs/umerge_s13

All steps read configs/default.yaml, then --config, then --set key=value overrides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .config import load_config


def _paths(cfg):
    p = {k: Path(v) for k, v in cfg["paths"].items()}
    p["interim_dir"].mkdir(parents=True, exist_ok=True)
    return p


def _stores(cfg):
    from .features.cache import ModalityStore

    cache = Path(cfg["paths"]["cache_dir"])
    return (ModalityStore.load(cache / "image" / cfg["encoders"]["image"]),
            ModalityStore.load(cache / "text" / cfg["encoders"]["text"]))


def _pairs(cfg):
    return pd.read_csv(_paths(cfg)["interim_dir"] / "pairs.csv",
                       dtype={"image_study": str, "report_study": str})


def cmd_manifest(cfg, args):
    from .data.iu_xray import build_manifest, build_manifest_from_hf

    p = _paths(cfg)
    if args.source == "hf":
        manifest, excluded = build_manifest_from_hf(p["hf_dir"], p["raw_dir"] / "images")
    else:
        manifest, excluded = build_manifest(p["raw_dir"])
    manifest.to_csv(p["interim_dir"] / "manifest.csv", index=False)
    excluded.to_csv(p["interim_dir"] / "exclusions.csv", index=False)
    print(f"{len(manifest)} studies kept, {len(excluded)} excluded")
    if len(excluded):
        print(excluded["reason"].value_counts().to_string())


def cmd_label(cfg, args):
    from .data.labels import label_reports, load_label_csv

    p = _paths(cfg)
    manifest = pd.read_csv(p["interim_dir"] / "manifest.csv", dtype={"study_id": str})
    if args.chexbert:
        from .data.chexbert import CheXbertLabeler

        labels = CheXbertLabeler(args.chexbert).label(manifest.set_index("study_id")["report_text"])
    elif args.csv:
        labels = load_label_csv(args.csv)
    else:
        print("WARNING: using the development keyword labeler; not valid for reported results.")
        labels = label_reports(manifest.set_index("study_id")["report_text"])
    labels.to_csv(p["interim_dir"] / "labels.csv", index_label="study_id")


def cmd_split(cfg, args):
    from .data.splits import assign_splits

    p = _paths(cfg)
    manifest = pd.read_csv(p["interim_dir"] / "manifest.csv", dtype={"study_id": str, "group_id": str})
    manifest["split"] = assign_splits(manifest, cfg["splits"], cfg["seed"])
    manifest.to_csv(p["interim_dir"] / "manifest.csv", index=False)
    print(manifest["split"].value_counts().to_string())


def cmd_pairs(cfg, args):
    from .data.negatives import build_pairs

    p = _paths(cfg)
    manifest = pd.read_csv(p["interim_dir"] / "manifest.csv", dtype={"study_id": str})
    labels = pd.read_csv(p["interim_dir"] / "labels.csv", dtype={"study_id": str}).set_index("study_id")
    n, frames, stats = cfg["negatives"], [], {}
    for i, (split, df) in enumerate(manifest.groupby("split")):
        pairs, s = build_pairs(df, labels, cfg["seed"] + i, n["dedup_threshold"],
                               n["hard_fraction"], n["hard_top_k"])
        frames.append(pairs.assign(split=split))
        stats[split] = s
    pd.concat(frames).to_csv(p["interim_dir"] / "pairs.csv", index=False)
    with open(p["interim_dir"] / "pairs_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


def cmd_encode(cfg, args):
    from .features.cache import encode_modality
    from .features.encoders import get_encoder
    from .features.quality import build_vocab

    p = _paths(cfg)
    manifest = pd.read_csv(p["interim_dir"] / "manifest.csv", dtype={"study_id": str})
    c = cfg["corruptions"]
    name = args.encoder or cfg["encoders"][args.modality]
    kinds = c[f"{args.modality}_seen"] + c[f"{args.modality}_unseen"]
    vocab = build_vocab(manifest.loc[manifest.split == "train", "report_text"].tolist())
    (p["interim_dir"] / "vocab.txt").write_text("\n".join(sorted(vocab)), encoding="utf-8")
    timings = encode_modality(
        manifest, args.modality, get_encoder(name), kinds, c["severities"], p["cache_dir"],
        image_dir=p["raw_dir"] / "images", vocab=vocab, batch_size=args.batch_size,
    )
    timings.to_csv(p["cache_dir"] / f"timings_{args.modality}_{name}.csv", index=False)


def cmd_train(cfg, args):
    from .training.train import train_run

    images, texts = _stores(cfg)
    run_dir = Path(cfg["paths"]["runs_dir"]) / (args.name or f"{args.model}_s{args.seed}")
    train_run(cfg, args.model, _pairs(cfg), images, texts, run_dir, args.seed)
    print(f"saved {run_dir}")


def cmd_calibrate(cfg, args):
    from .training.train import calibrate_run

    images, texts = _stores(cfg)
    print(json.dumps(calibrate_run(cfg, args.run, _pairs(cfg), images, texts), indent=2))


def cmd_evaluate(cfg, args):
    from .training.train import evaluate_run

    images, texts = _stores(cfg)
    res = evaluate_run(cfg, args.run, _pairs(cfg), images, texts)
    print(json.dumps(res["clean"], indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="medinova", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides, e.g. --set loss.alpha=0 model.d=64")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest").add_argument(
        "--source", choices=["hf", "openi"], default="hf",
        help="hf: Hugging Face mirror parquet (paths.hf_dir); openi: NLM XML + PNG (paths.raw_dir)")
    lab = sub.add_parser("label")
    lab.add_argument("--chexbert", default=None, help="path to chexbert.pth: run CheXbert here")
    lab.add_argument("--csv", default=None, help="labels already produced by CheXbert/CheXpert")
    sub.add_parser("split")
    sub.add_parser("pairs")
    enc = sub.add_parser("encode")
    enc.add_argument("--modality", choices=["image", "text"], required=True)
    enc.add_argument("--encoder", default=None)
    enc.add_argument("--batch-size", type=int, default=16)
    tr = sub.add_parser("train")
    tr.add_argument("--model", default="umerge")
    tr.add_argument("--seed", type=int, default=13)
    tr.add_argument("--name", default=None)
    for name in ("calibrate", "evaluate"):
        sub.add_parser(name).add_argument("--run", required=True)

    args = ap.parse_args(argv)
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load_config(args.config, overrides)
    globals()[f"cmd_{args.cmd}"](cfg, args)


if __name__ == "__main__":
    main()
