# Data

Nothing under `raw/`, `interim/` or `cache/` is committed. Keep restricted files on this machine.

## IU X-Ray / Open-I (primary)

Confirm the licence and terms of use before downloading (proposal week 1–4). Then place files as:

```
data/raw/iu_xray/
  reports/*.xml          Open-I "ecgen-radiology" XML, one file per study
  images/*.png           PNGs named by the XML's parentImage id (e.g. CXR1_1_IM-0001-3001.png)
  projections.csv        optional: uid, filename, projection (Frontal / Lateral)
```

Without `projections.csv`, the first linked image is used and `frontal_source` in the manifest
says so; check a sample by hand.

## Finding labels

Run CheXbert (or the CheXpert labeler) on `report_text` from `data/interim/manifest.csv` and save a CSV
with a `study_id` column plus the 14 CheXpert label columns (1 / 0 / -1 / blank). Then:

```
medinova label --csv path/to/chexbert_labels.csv
```

`medinova label` with no `--csv` uses a keyword fallback for development only.

## Generated files

| File | Made by |
| --- | --- |
| `interim/manifest.csv`, `interim/exclusions.csv` | `medinova manifest` (+ `split` column from `medinova split`) |
| `interim/labels.csv` | `medinova label` |
| `interim/pairs.csv`, `interim/pairs_stats.json` | `medinova pairs` |
| `interim/vocab.txt` | `medinova encode --modality text` (training split only) |
| `cache/<modality>/<encoder>/<variant>.npz` | `medinova encode` |
| `cache/timings_*.csv` | `medinova encode` (CPU cost per variant, for proposal section 9) |
