"""Parse IU X-Ray / Open-I reports into a study manifest (proposal 5.1, 5.3).

Expected raw layout (not committed, see data/README.md):

    data/raw/iu_xray/reports/*.xml      Open-I "ecgen-radiology" XML, one per study
    data/raw/iu_xray/images/*.png       images named by parentImage id
    data/raw/iu_xray/projections.csv    optional: columns uid, filename, projection

Without a projections file the first linked image is used as the frontal view and
this is logged, because view labels then have to be confirmed by hand.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = [
    "study_id", "group_id", "findings", "impression", "report_text",
    "image_ids", "frontal_image", "frontal_source",
]


def _clean(text: str | None) -> str:
    text = (text or "").strip()
    return re.sub(r"\s+", " ", text)


def parse_report(path: Path) -> dict:
    root = ET.parse(path).getroot()
    sections: dict[str, str] = {}
    for node in root.iter("AbstractText"):
        label = (node.get("Label") or "").upper()
        sections[label] = _clean("".join(node.itertext()))
    uid = root.find(".//uId")
    study_id = uid.get("id") if uid is not None and uid.get("id") else path.stem
    image_ids = [n.get("id") for n in root.iter("parentImage") if n.get("id")]
    return {
        "study_id": study_id,
        "findings": sections.get("FINDINGS", ""),
        "impression": sections.get("IMPRESSION", ""),
        "image_ids": image_ids,
    }


def build_manifest_from_hf(
    parquet_dir: str | Path, image_out_dir: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Manifest from the Hugging Face mirror `ykumards/open-i` (Kaggle raddar release).

    One row per study `uid`, with findings, impression and the frontal image as encoded
    bytes (views were manually classified as frontal / lateral by the mirror's author).
    Frontal images are written once to `image_out_dir` as <uid>_frontal.png.
    """
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    image_out_dir = Path(image_out_dir)
    image_out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(parquet_dir).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {parquet_dir}")

    rows, excluded = [], []
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=64, columns=["uid", "findings", "impression", "img_frontal"]):
            for rec in batch.to_pylist():
                study = str(rec["uid"])
                findings, impression = _clean(rec["findings"]), _clean(rec["impression"])
                text = " ".join(t for t in (findings, impression) if t)
                if not text:
                    excluded.append({"study_id": study, "reason": "no_findings_or_impression"})
                    continue
                img = rec["img_frontal"]
                if isinstance(img, dict):  # datasets Image feature: {"bytes", "path"}
                    img = img.get("bytes")
                if not img:
                    excluded.append({"study_id": study, "reason": "no_frontal_view"})
                    continue
                name = f"{study}_frontal.png"
                target = image_out_dir / name
                if not target.exists():
                    try:
                        Image.open(io.BytesIO(img)).convert("L").save(target)
                    except OSError as e:
                        excluded.append({"study_id": study, "reason": f"unreadable_image: {e}"})
                        continue
                rows.append({
                    "study_id": study, "group_id": study, "findings": findings,
                    "impression": impression, "report_text": text, "image_ids": name,
                    "frontal_image": name, "frontal_source": "hf_manual_view_label",
                })
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    dupes = manifest.duplicated("study_id", keep="first")
    excluded += [{"study_id": s, "reason": "duplicate_uid"} for s in manifest.loc[dupes, "study_id"]]
    return manifest[~dupes].reset_index(drop=True), pd.DataFrame(excluded)


def _load_projections(path: Path | None) -> dict[str, str]:
    """Map image stem -> projection ('Frontal' / 'Lateral')."""
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    stems = df["filename"].astype(str).str.replace(r"\.dcm\.png$|\.png$", "", regex=True)
    return dict(zip(stems, df["projection"].astype(str)))


def build_manifest(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (manifest, exclusions). Every excluded study is logged with a reason."""
    raw_dir = Path(raw_dir)
    report_dir, image_dir = raw_dir / "reports", raw_dir / "images"
    if not report_dir.exists():
        raise FileNotFoundError(f"No reports directory at {report_dir}; see data/README.md")
    projections = _load_projections(raw_dir / "projections.csv")
    available = {p.stem.replace(".dcm", ""): p.name for p in image_dir.glob("*.png")}

    rows, excluded = [], []
    for xml_path in sorted(report_dir.glob("*.xml")):
        try:
            rec = parse_report(xml_path)
        except ET.ParseError as e:
            excluded.append({"study_id": xml_path.stem, "reason": f"xml_parse_error: {e}"})
            continue
        text = " ".join(t for t in (rec["findings"], rec["impression"]) if t)
        if not text:
            excluded.append({"study_id": rec["study_id"], "reason": "no_findings_or_impression"})
            continue
        images = [i for i in rec["image_ids"] if i in available]
        if not images:
            excluded.append({"study_id": rec["study_id"], "reason": "no_linked_image_on_disk"})
            continue
        frontal = [i for i in images if projections.get(i, "").lower() == "frontal"]
        if projections and not frontal:
            excluded.append({"study_id": rec["study_id"], "reason": "no_frontal_view"})
            continue
        rows.append({
            "study_id": rec["study_id"],
            # IU X-Ray has no patient ids; splits fall back to study-level grouping.
            "group_id": rec["study_id"],
            "findings": rec["findings"],
            "impression": rec["impression"],
            "report_text": text,
            "image_ids": ";".join(images),
            "frontal_image": available[(frontal or images)[0]],
            "frontal_source": "projections_csv" if frontal else "first_linked_image",
        })
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS), pd.DataFrame(excluded)
