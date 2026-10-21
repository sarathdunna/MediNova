# Data

Nothing under `raw/`, `interim/` or `cache/` is committed. Keep restricted files on this machine.

## IU X-Ray / Open-I (primary)

Default source (`medinova manifest`): the Hugging Face mirror
[`ykumards/open-i`](https://huggingface.co/datasets/ykumards/open-i), CC BY-NC-ND 4.0
(non-commercial; do not redistribute modified images). Download its parquet files to
`data/raw/hf_open_i/`; frontal images are extracted to `data/raw/iu_xray/images/`.

Alternative (`medinova manifest --source openi`), the NLM Open-I files, when available:

```
data/raw/iu_xray/
  reports/*.xml          Open-I "ecgen-radiology" XML, one file per study
  images/*.png           PNGs named by the XML's parentImage id (e.g. CXR1_1_IM-0001-3001.png)
  projections.csv        optional: uid, filename, projection (Frontal / Lateral)
```

Without `projections.csv`, the first linked image is used and `frontal_source` in the manifest
says so; check a sample by hand.

## Finding labels

Reported results use CheXbert, run in-package on `report_text` (findings + impression):

```
git clone --depth 1 https://github.com/stanfordmlgroup/CheXbert tools/CheXbert   # reference samples for the test
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('StanfordAIMI/RRG_scorers', 'chexbert.pth', local_dir='tools/weights')"
medinova label --chexbert tools/weights/chexbert.pth
```

The checkpoint (1.3 GB) is used under Stanford's CheXbert research licence (`tools/CheXbert/LICENSE.pdf`);
the original Box link in the CheXbert README is dead. `tests/test_chexbert.py` checks that the port
reproduces CheXbert's own labelled samples exactly.

Alternatives: `medinova label --csv labels.csv` reads labels produced elsewhere (`study_id` + 14 CheXpert
columns, 1 / 0 / -1 / blank). `medinova label` with neither option uses a keyword fallback for
development only.

## Generated files

| File | Made by |
| --- | --- |
| `interim/manifest.csv`, `interim/exclusions.csv` | `medinova manifest` (+ `split` column from `medinova split`) |
| `interim/labels.csv` | `medinova label` |
| `interim/pairs.csv`, `interim/pairs_stats.json` | `medinova pairs` |
| `interim/vocab.txt` | `medinova encode --modality text` (training split only) |
| `cache/<modality>/<encoder>/<variant>.npz` | `medinova encode` |
| `cache/timings_*.csv` | `medinova encode` (CPU cost per variant, for proposal section 9) |
