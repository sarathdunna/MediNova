"""Finding labels per report (proposal 5.2, step 1).

The intended labeler is CheXbert or the CheXpert rule-based labeler, run outside
this package; its CSV output is read by `load_label_csv`. `simple_labeler` is a
small keyword + negation fallback for development and tests only. It is not a
substitute for a validated labeler and must not be used for reported results.

Encoding per finding: 1 positive, 0 negative, -1 uncertain, NaN not mentioned.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]


def load_label_csv(path: str, id_column: str = "study_id") -> pd.DataFrame:
    """Read labeler output: one row per report, one column per CheXpert label."""
    df = pd.read_csv(path)
    missing = [c for c in CHEXPERT_LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"Label CSV is missing columns: {missing}")
    return df.set_index(id_column)[CHEXPERT_LABELS].astype(float)


_KEYWORDS = {
    "Cardiomegaly": r"cardiomegaly|enlarged heart|heart (size )?is enlarged",
    "Lung Opacity": r"opacit",
    "Lung Lesion": r"nodule|mass|lesion",
    "Edema": r"edema",
    "Consolidation": r"consolidation",
    "Pneumonia": r"pneumonia",
    "Atelectasis": r"atelecta",
    "Pneumothorax": r"pneumothora",
    "Pleural Effusion": r"effusion",
    "Fracture": r"fracture",
    "Support Devices": r"catheter|tube|pacemaker|line in place|sternotomy",
}
_NEGATION = re.compile(r"\b(no|without|negative for|free of|clear of|resolved)\b[^.;]*$")
_UNCERTAIN = re.compile(r"\b(may|possible|possibly|cannot exclude|questionable|suggest)\b[^.;]*$")


def simple_labeler(text: str) -> np.ndarray:
    """Development-only keyword labeler with sentence-local negation detection."""
    out = np.full(len(CHEXPERT_LABELS), np.nan)
    sentences = re.split(r"[.;\n]", text.lower())
    for name, pattern in _KEYWORDS.items():
        idx = CHEXPERT_LABELS.index(name)
        for s in sentences:
            m = re.search(pattern, s)
            if not m:
                continue
            prefix = s[: m.start()]
            if _NEGATION.search(prefix):
                value = 0.0
            elif _UNCERTAIN.search(prefix):
                value = -1.0
            else:
                value = 1.0
            # A positive mention anywhere wins over a negative one.
            out[idx] = value if np.isnan(out[idx]) else max(out[idx], value)
    findings = np.delete(out, 0)
    out[0] = 0.0 if np.any(findings == 1) else 1.0
    return out


def label_reports(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        np.stack([simple_labeler(t) for t in texts]), index=texts.index, columns=CHEXPERT_LABELS
    )


def definite_conflict_matrix(anchor: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """True where a candidate report's findings conflict with the anchor's.

    Not-mentioned is read as negative; uncertain values never count as a conflict.
    anchor: (L,), candidates: (N, L) -> (N,) bool
    """
    a = np.nan_to_num(anchor, nan=0.0)
    c = np.nan_to_num(candidates, nan=0.0)
    comparable = (a != -1) & (c != -1)
    return np.any(comparable & (a != c), axis=1)
