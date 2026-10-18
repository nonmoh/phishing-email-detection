from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import re, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "processed" / "uploaded_unified_dedup.csv"
LARGE_DIR = ROOT / "data" / "raw" / "kaggle_large"
OUT = ROOT / "data" / "processed" / "master_dataset.csv"
STATS = ROOT / "data" / "processed" / "master_dataset_stats.json"

parser = argparse.ArgumentParser()
parser.add_argument("--small-only", action="store_true", help="Do not merge Kaggle large data.")
args = parser.parse_args()

if not BASE.exists():
    raise SystemExit(f"Missing {BASE}")

base_df = pd.read_csv(BASE, low_memory=False)

def norm(s):
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).replace("\x00", " ")).strip()

def text_hash(s):
    return hashlib.sha256(norm(s).lower().encode("utf-8", "ignore")).hexdigest()

def map_label(v):
    s = str(v).strip().lower()
    if s in {"1","phishing","phish","spam","fraud","malicious","spam email","phishing email"}:
        return 1
    if s in {"0","legitimate","legit","ham","safe","benign","safe email"}:
        return 0
    return np.nan

def find_col(cols, candidates):
    low = {str(c).lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for c in cols:
        lc = str(c).lower().strip()
        if any(k in lc for k in candidates):
            return c
    return None

# Keep common schema from the existing cleaned corpus
required_cols = list(base_df.columns)
frames = [base_df.copy()]
source_report = [{"dataset":"uploaded_small","rows":len(base_df),"path":str(BASE)}]

if not args.small_only and LARGE_DIR.exists():
    candidates = []
    for p in LARGE_DIR.rglob("*.csv"):
        try:
            d = pd.read_csv(p, low_memory=False)
        except Exception as e:
            print("skip", p.name, e)
            continue

        text_col = find_col(
            d.columns,
            ["email text","email_text","text_combined","email content","email_content","text","body","message"]
        )
        label_col = find_col(
            d.columns,
            ["email type","email_type","label","class","type","category"]
        )
        if not text_col or not label_col:
            print("No compatible text/label columns:", p.name, list(d.columns)[:12])
            continue

        tmp = pd.DataFrame()
        tmp["text"] = d[text_col].map(norm)
        tmp["phishing_label"] = d[label_col].map(map_label)
        tmp = tmp[tmp["phishing_label"].notna() & tmp["text"].str.len().gt(0)].copy()
        tmp["phishing_label"] = tmp["phishing_label"].astype(int)

        tmp["subject"] = ""
        tmp["body"] = tmp["text"]
        tmp["origin"] = "human_or_legacy"
        tmp["generator"] = "not_applicable"
        tmp["register"] = "unspecified"
        tmp["source"] = p.stem
        tmp["category"] = ""
        tmp["dataset"] = "kaggle_large"
        tmp["license"] = "See Kaggle dataset page and original source-corpus licenses"
        tmp["text_hash"] = tmp["text"].map(text_hash)
        tmp["raw_record_id"] = [f"K{i:07d}" for i in range(1, len(tmp)+1)]
        tmp["record_id"] = ""

        # Align with base schema
        for c in required_cols:
            if c not in tmp.columns:
                tmp[c] = ""
        tmp = tmp[required_cols]
        candidates.append((len(tmp), p, tmp))
        print("Accepted", p.name, len(tmp), "label counts", tmp["phishing_label"].value_counts().to_dict())

    if candidates:
        # The Kaggle package contains multiple source files; use the largest compatible combined file
        # to avoid counting nested source corpora twice.
        candidates.sort(key=lambda x: x[0], reverse=True)
        n, p, best = candidates[0]
        frames.append(best)
        source_report.append({"dataset":"kaggle_large","rows":n,"path":str(p)})
        print("Selected large CSV:", p, "rows:", n)
    else:
        print("WARNING: large data folder exists but no compatible CSV was found.")
else:
    print("Large dataset not merged (folder missing or --small-only used).")

before = pd.concat(frames, ignore_index=True)
before["text"] = before["text"].map(norm)
before["text_hash"] = before["text"].map(text_hash)

# Remove conflicting labels for identical normalized text.
conf = before.groupby("text_hash")["phishing_label"].nunique()
bad_hashes = set(conf[conf > 1].index)
if bad_hashes:
    print("Removing hashes with conflicting labels:", len(bad_hashes))

clean = before[~before["text_hash"].isin(bad_hashes)].copy()
before_dedup = len(clean)
clean = clean.drop_duplicates("text_hash", keep="first").copy()
clean["record_id"] = [f"M{i:07d}" for i in range(1, len(clean)+1)]

# Useful derived fields for EDA/error analysis
clean["char_length"] = clean["text"].astype(str).str.len()
clean["word_length"] = clean["text"].astype(str).str.split().str.len()

OUT.parent.mkdir(parents=True, exist_ok=True)
clean.to_csv(OUT, index=False, encoding="utf-8-sig")

stats = {
    "rows_before_merge_dedup": int(len(before)),
    "rows_after_conflict_filter_before_dedup": int(before_dedup),
    "rows_after_dedup": int(len(clean)),
    "duplicates_removed": int(before_dedup - len(clean)),
    "conflicting_hashes_removed": int(len(bad_hashes)),
    "class_counts": {str(k):int(v) for k,v in clean["phishing_label"].value_counts().sort_index().items()},
    "dataset_counts": {str(k):int(v) for k,v in clean["dataset"].value_counts(dropna=False).items()},
    "origin_counts": {str(k):int(v) for k,v in clean["origin"].value_counts(dropna=False).items()},
    "generator_counts": {str(k):int(v) for k,v in clean["generator"].value_counts(dropna=False).items()},
    "source_inputs": source_report,
}
STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(stats, indent=2, ensure_ascii=False))
print("Saved:", OUT)
