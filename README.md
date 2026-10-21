# MediNova X: U-MERGE

Uncertainty-aware image–report correspondence for chest X-rays, built to run on a laptop CPU.
Research and education only: not a medical device, not for clinical use.

Proposal (revision 2): https://claude.ai/artifact/6498XZ933acQU98WzCfUEW

## What is here

| Path | Proposal section | Contents |
| --- | --- | --- |
| `src/medinova/data/iu_xray.py` | 5.1, 5.3 | Open-I XML → study manifest, with an exclusion log |
| `src/medinova/data/labels.py` | 5.2 | CheXbert CSV loader, dev keyword labeler, finding-conflict test |
| `src/medinova/data/negatives.py` | 5.2 | Template dedup, finding-aware and hard negatives |
| `src/medinova/data/splits.py` | 5.3 | Four-way group-disjoint splits |
| `src/medinova/data/corruptions.py` | 6.3 | Seen / unseen image and text corruptions, severity 0–3 |
| `src/medinova/features/` | 6.1, 9 | Frozen encoders, quality features q_m, embedding cache |
| `src/medinova/models/umerge.py` | 6.2, 6.3 | U-MERGE head |
| `src/medinova/models/baselines.py` | 7 | Projected cosine, concat MLP, bilinear, cross-attention |
| `src/medinova/models/losses.py` | 6.4 | BCE, L_unc, InfoNCE, consistency KL |
| `src/medinova/evaluation/` | 6.5, 7 | Temperature scaling, abstention rule, metrics, bootstrap |
| `src/medinova/training/` | 7 | Batch sampling from the cache; train / calibrate / evaluate |
| `api/` | 8 | FastAPI research API |
| `web/` | 8 | Next.js research console |
| `configs/default.yaml` | all | Every experiment setting |

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[dev,api]"
.venv\Scripts\python -m pytest -q
```

Add the encoders when you are ready to embed real data (large downloads on first use):

```powershell
.venv\Scripts\python -m pip install -e ".[encoders]"
```

## Pipeline

```powershell
medinova manifest                       # data/raw/iu_xray -> data/interim/manifest.csv
medinova label --csv chexbert.csv       # or no --csv for the dev-only keyword labeler
medinova split
medinova pairs
medinova encode --modality text
medinova encode --modality image        # slowest step; timings saved to data/cache/
medinova train --model umerge --seed 13
medinova calibrate --run runs/umerge_s13
medinova evaluate  --run runs/umerge_s13
```

Ablations and baselines are config overrides or model names:

```powershell
medinova train --model concat_mlp
medinova train --model umerge --name umerge_unsupervised_u --set loss.alpha=0 model.stop_grad_uncertainty=false
medinova train --model umerge --name umerge_generic --set encoders.image=mobilenet_v3_small encoders.text=minilm
```

## Console

```powershell
.venv\Scripts\uvicorn api.main:app --reload --port 8000
cd web; npm install; npm run dev        # http://localhost:3000
```

Or `docker compose up --build`. Use synthetic examples only in any public deployment.

## Status

- Core logic (splits, negatives, corruptions, models, losses, calibration, metrics, train / calibrate /
  evaluate) is covered by tests, including an end-to-end run on synthetic caches.
- The encoder wrappers and the IU X-Ray parser are written against their libraries and data format but
  have not been run here: no data or weights are downloaded yet. Verify them in week 5.
