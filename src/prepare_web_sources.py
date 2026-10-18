from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

CROSS = RAW / "cross_model_web"
GRECO = RAW / "four_class_web"


def norm(v) -> str:
    if pd.isna(v):
        return ""
    return re.sub(r"\s+", " ", str(v).replace("\x00", " ")).strip()


def sha(v: str) -> str:
    return hashlib.sha256(norm(v).lower().encode("utf-8", "ignore")).hexdigest()


def find_col(cols, names):
    lower = {str(c).strip().lower(): c for c in cols}
    for name in names:
        if name in lower:
            return lower[name]
    for c in cols:
        lc = str(c).lower()
        if any(name in lc for name in names):
            return c
    return None


def schema_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "record_id", "text", "subject", "body", "phishing_label", "origin",
        "generator", "register", "source", "category", "dataset", "license",
        "text_hash", "raw_record_id"
    ])


def load_cross_model() -> pd.DataFrame:
    data_dir = CROSS / "data"
    human = data_dir / "human_corpus_sampled.csv"
    llm = data_dir / "llm_corpus_sampled.csv"
    if not human.exists() or not llm.exists():
        raise FileNotFoundError(
            "Zenodo cross-model files are missing. Run src/download_all_project_data.py first."
        )

    h = pd.read_csv(human, low_memory=False)
    l = pd.read_csv(llm, low_memory=False)

    h2 = schema_frame()
    h2["text"] = h["text"].map(norm)
    h2["subject"] = h.get("subject", "").map(norm) if "subject" in h else ""
    h2["body"] = h.get("body", h["text"]).map(norm)
    h2["phishing_label"] = 1
    h2["origin"] = "human"
    h2["generator"] = "human"
    h2["register"] = "unspecified"
    h2["source"] = h.get("source", "human_cross_model").astype(str)
    h2["category"] = h.get("category", "") if "category" in h else ""
    h2["dataset"] = "cross_model_phishing"
    h2["license"] = "CC BY 4.0"

    l2 = schema_frame()
    l2["text"] = l["text"].map(norm)
    l2["subject"] = l.get("subject", "").map(norm) if "subject" in l else ""
    l2["body"] = l.get("body", l["text"]).map(norm)
    l2["phishing_label"] = 1
    l2["origin"] = "llm"
    model_col = "model" if "model" in l else ("source" if "source" in l else None)
    l2["generator"] = l[model_col].astype(str).str.lower().str.strip() if model_col else "unknown_llm"
    l2["register"] = "unspecified"
    l2["source"] = l.get("source", "llm_cross_model").astype(str)
    l2["category"] = l.get("category", "") if "category" in l else ""
    l2["dataset"] = "cross_model_phishing"
    l2["license"] = "CC BY 4.0"

    out = pd.concat([h2, l2], ignore_index=True)
    return out


def infer_phishing_from_path(p: Path):
    s = str(p).lower()
    if "phishing" in s:
        return 1
    if "legit" in s or "legitimate" in s:
        return 0
    return None


def infer_origin_from_path(p: Path):
    s = str(p).lower()
    if "llm" in s:
        return "llm"
    if "human" in s:
        return "human"
    return "unknown"


def load_greco() -> pd.DataFrame:
    if not GRECO.exists():
        raise FileNotFoundError(
            "Greco Kaggle data are missing. Run src/download_all_project_data.py first."
        )
    frames = []
    for p in GRECO.rglob("*.csv"):
        label = infer_phishing_from_path(p)
        origin = infer_origin_from_path(p)
        if label is None:
            continue
        try:
            d = pd.read_csv(p, low_memory=False)
        except Exception as exc:
            # One public Greco CSV is malformed because commas inside the text
            # are not consistently quoted. Its schema is text,label, so parse
            # each physical line from the RIGHT and keep the final 0/1 field.
            print("Standard CSV parser failed for", p.name, "- using right-split fallback:", exc)
            rows = []
            try:
                with p.open("r", encoding="utf-8", errors="replace") as fh:
                    header = fh.readline()
                    for line in fh:
                        line = line.rstrip("\r\n")
                        if not line:
                            continue
                        left, sep, right = line.rpartition(",")
                        if sep and right.strip() in {"0", "1"}:
                            rows.append({"text": left, "label": int(right.strip())})
                d = pd.DataFrame(rows)
                if d.empty:
                    print("Fallback parser produced no rows; skipping", p)
                    continue
            except Exception as fallback_exc:
                print("Skipping unreadable Greco CSV:", p, fallback_exc)
                continue

        # Prefer a dedicated text/email-text column. If the file exposes
        # separate subject/body fields, combine them exactly as the project
        # inference path does; do not silently use body alone.
        text_col = find_col(d.columns, ["text", "email_text", "email content", "email_content"])
        subject_col = find_col(d.columns, ["subject", "title"])
        body_col = find_col(d.columns, ["body", "content", "message"])

        if text_col:
            text = d[text_col].map(norm)
        elif subject_col or body_col:
            subj = d[subject_col].map(norm) if subject_col else pd.Series([""] * len(d), index=d.index)
            body = d[body_col].map(norm) if body_col else pd.Series([""] * len(d), index=d.index)
            text = (subj + " " + body).map(norm)
        else:
            print("No text column in", p)
            continue

        f = schema_frame()
        f["text"] = text
        f["subject"] = d[subject_col].map(norm) if subject_col else ""
        f["body"] = d[body_col].map(norm) if body_col else text
        f["phishing_label"] = int(label)
        f["origin"] = origin
        f["generator"] = "unknown_llm" if origin == "llm" else ("human" if origin == "human" else "unknown")
        f["register"] = "unspecified"
        f["source"] = p.parent.name
        f["category"] = ""
        f["dataset"] = "four_class_human_llm"
        f["license"] = "CC BY-NC-SA 4.0"
        frames.append(f)

    if not frames:
        raise RuntimeError("No Greco four-class CSV files could be parsed.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    cross = load_cross_model()
    greco = load_greco()
    before = pd.concat([cross, greco], ignore_index=True)
    before["text"] = before["text"].map(norm)
    before = before[before["text"].str.len().gt(0)].copy()
    before["text_hash"] = before["text"].map(sha)
    before["raw_record_id"] = [f"R{i:06d}" for i in range(1, len(before) + 1)]

    # Remove text hashes with conflicting phishing labels rather than guessing.
    conflicts = before.groupby("text_hash")["phishing_label"].nunique()
    bad = set(conflicts[conflicts > 1].index)
    conflict_rows = before[before["text_hash"].isin(bad)].copy()
    conflict_rows.to_csv(PROC / "conflicting_duplicate_labels.csv", index=False, encoding="utf-8-sig")

    clean = before[~before["text_hash"].isin(bad)].copy()
    before_dedup = len(clean)
    clean = clean.drop_duplicates("text_hash", keep="first").copy()
    clean["record_id"] = [f"E{i:06d}" for i in range(1, len(clean) + 1)]

    before.to_csv(PROC / "uploaded_unified_before_dedup.csv", index=False, encoding="utf-8-sig")
    clean.to_csv(PROC / "uploaded_unified_dedup.csv", index=False, encoding="utf-8-sig")

    stats = {
        "raw_rows_total": int(len(before)),
        "dedup_rows_total": int(len(clean)),
        "exact_duplicates_removed": int(before_dedup - len(clean)),
        "conflicting_label_rows_removed": int(len(conflict_rows)),
        "class_counts": {str(k): int(v) for k, v in clean["phishing_label"].value_counts().sort_index().items()},
        "origin_counts": {str(k): int(v) for k, v in clean["origin"].value_counts(dropna=False).items()},
        "generator_counts": {str(k): int(v) for k, v in clean["generator"].value_counts(dropna=False).items()},
        "dataset_counts": {str(k): int(v) for k, v in clean["dataset"].value_counts(dropna=False).items()},
    }
    (PROC / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # Copy original cross-model result artifacts into the project results folder.
    src_results = CROSS / "results"
    dst_results = ROOT / "results" / "original_cross_model"
    if src_results.exists():
        dst_results.mkdir(parents=True, exist_ok=True)
        for p in src_results.rglob("*"):
            if p.is_file():
                target = dst_results / p.relative_to(src_results)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("Prepared:", PROC / "uploaded_unified_dedup.csv")


if __name__ == "__main__":
    main()
