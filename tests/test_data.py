import numpy as np
import pandas as pd
from PIL import Image

from medinova.data.corruptions import IMAGE_KINDS, TEXT_KINDS, corrupt_image, corrupt_text
from medinova.data.labels import (
    CHEXPERT_LABELS,
    definite_conflict_matrix,
    label_reports,
    simple_labeler,
)
from medinova.data.negatives import build_pairs
from medinova.data.splits import SPLITS, assign_splits

REPORTS = [
    "The heart is normal in size. The lungs are clear. No pleural effusion or pneumothorax.",
    "Heart size normal. Lungs are clear. No effusion. No pneumothorax.",
    "The heart is normal in size. The lungs are clear. No pleural effusion or pneumothorax.",
    "Cardiomegaly. Small left pleural effusion. No pneumothorax.",
    "Right lower lobe consolidation concerning for pneumonia. No effusion.",
    "Large right pneumothorax. Heart size normal.",
    "Mild cardiomegaly with pulmonary edema. Bilateral pleural effusions.",
    "Heart size normal. Left basilar atelectasis. No effusion.",
]


def test_splits_are_group_disjoint_and_complete():
    m = pd.DataFrame({"study_id": range(1000), "group_id": np.arange(1000) // 2})
    s = assign_splits(m, {"train": .6, "validation": .15, "calibration": .1, "test": .15}, seed=1)
    assert set(s) == set(SPLITS)
    assert (m.assign(split=s).groupby("group_id").split.nunique() == 1).all()
    assert abs((s == "train").mean() - 0.6) < 0.02


def test_simple_labeler_negation():
    v = simple_labeler("No pleural effusion. Small right pneumothorax.")
    assert v[CHEXPERT_LABELS.index("Pleural Effusion")] == 0
    assert v[CHEXPERT_LABELS.index("Pneumothorax")] == 1
    assert v[CHEXPERT_LABELS.index("No Finding")] == 0


def test_conflict_ignores_uncertain_and_treats_absent_as_negative():
    nan = np.nan
    a = np.array([1, nan, 0, -1])
    c = np.array([[1, 0, 0, 1], [1, 1, 0, -1], [1, nan, nan, 0]])
    assert definite_conflict_matrix(a, c).tolist() == [False, True, False]


def test_pairs_never_use_template_duplicates_or_agreeing_findings():
    df = pd.DataFrame({"study_id": [f"S{i}" for i in range(len(REPORTS))], "report_text": REPORTS})
    labels = label_reports(df.set_index("study_id").report_text)
    pairs, stats = build_pairs(df, labels, seed=0, hard_fraction=1.0)
    assert stats["positives"] == len(REPORTS)
    neg = pairs[pairs.label == 0]
    # S0 and S2 are identical text: never a negative for each other.
    assert not ((neg.image_study == "S0") & (neg.report_study == "S2")).any()
    for _, r in neg.iterrows():
        assert definite_conflict_matrix(
            labels.loc[r.image_study].to_numpy(), labels.loc[[r.report_study]].to_numpy()
        )[0]


def test_corruptions_identity_at_zero_and_change_otherwise():
    img = Image.fromarray((np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8), "L")
    for k in IMAGE_KINDS:
        assert np.array_equal(np.asarray(corrupt_image(img, k, 0)), np.asarray(img))
        assert not np.array_equal(np.asarray(corrupt_image(img, k, 3, seed=1)), np.asarray(img))
    for k in TEXT_KINDS:
        assert corrupt_text(REPORTS[3], k, 0) == REPORTS[3]
        assert corrupt_text(REPORTS[3], k, 3, seed=1) != REPORTS[3]
    assert corrupt_text(REPORTS[3], "typo", 2, seed=5) == corrupt_text(REPORTS[3], "typo", 2, seed=5)
