"""Our CheXbert port must reproduce the reference labels shipped with the original repo.

Needs tools/CheXbert (git clone) and tools/weights/chexbert.pth; skipped otherwise (e.g. CI).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "tools" / "weights" / "chexbert.pth"
SAMPLES = ROOT / "tools" / "CheXbert" / "src"

pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and (SAMPLES / "labeled_reports.csv").exists()),
    reason="CheXbert checkpoint or reference samples not present",
)


def test_port_matches_reference_labels():
    from medinova.data.chexbert import CHEXBERT_CONDITIONS, CheXbertLabeler

    reports = pd.read_csv(SAMPLES / "sample_reports.csv")["Report Impression"]
    expected = pd.read_csv(SAMPLES / "labeled_reports.csv")[CHEXBERT_CONDITIONS].to_numpy()
    got = CheXbertLabeler(str(CKPT)).label(reports)[CHEXBERT_CONDITIONS].to_numpy()
    assert np.array_equal(np.isnan(got), np.isnan(expected))
    assert np.array_equal(np.nan_to_num(got, nan=9), np.nan_to_num(expected, nan=9))
