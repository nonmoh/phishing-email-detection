from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, matthews_corrcoef, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "processed" / "master_dataset.csv"
PRED = ROOT / "results" / "baseline" / "baseline_predictions.csv"
OUT = ROOT / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MASTER, low_memory=False)
pred = pd.read_csv(PRED)
test_pred = pred[pred["split"].eq("test")].copy()
merged = test_pred.merge(df, on="record_id", how="left")

def metric_row(g, model, subgroup_name, subgroup_value):
    y = g["true_label"].astype(int)
    p = g["predicted_label"].astype(int)
    pr,rc,f,_ = precision_recall_fscore_support(y,p,average="binary",zero_division=0)
    return {
        "model":model, "subgroup":subgroup_name, "value":str(subgroup_value),
        "n":len(g), "accuracy":accuracy_score(y,p),
        "balanced_accuracy":balanced_accuracy_score(y,p),
        "precision":pr, "recall":rc, "f1":f, "mcc":matthews_corrcoef(y,p)
    }

# Length bands based on words
if "word_length" not in merged.columns:
    merged["word_length"] = merged["text"].astype(str).str.split().str.len()
bins = [-1,50,100,200,500,10**9]
labels = ["<=50","51-100","101-200","201-500",">500"]
merged["length_band"] = pd.cut(merged["word_length"], bins=bins, labels=labels)

subgroup_rows = []
for model, gm in merged.groupby("model"):
    for col in ["origin","dataset","generator","length_band"]:
        if col not in gm.columns:
            continue
        for value, g in gm.groupby(col, dropna=False, observed=False):
            if len(g) < 10:
                continue
            subgroup_rows.append(metric_row(g, model, col, value))
pd.DataFrame(subgroup_rows).to_csv(OUT/"subgroup_metrics.csv", index=False)

# Errors
errors = merged[merged["true_label"] != merged["predicted_label"]].copy()
errors["error_type"] = np.where(
    (errors["true_label"].eq(1)) & (errors["predicted_label"].eq(0)),
    "false_negative",
    "false_positive"
)
keep_cols = [c for c in [
    "record_id","model","true_label","predicted_label","score","error_type",
    "dataset","origin","generator","word_length","subject","text"
] if c in errors.columns]
errors[keep_cols].to_csv(OUT/"error_examples.csv", index=False)

# Aggregate error counts
err_counts = errors.groupby(["model","error_type"]).size().reset_index(name="count")
err_counts.to_csv(OUT/"error_counts.csv", index=False)

# Logistic-regression calibration bins on the held-out test set
lr = merged[merged["model"].eq("logreg")].copy()
if not lr.empty:
    lr["probability"] = lr["score"].clip(0,1)
    lr["prob_bin"] = pd.cut(lr["probability"], bins=np.linspace(0,1,11), include_lowest=True)
    cal = lr.groupby("prob_bin", observed=False).agg(
        n=("true_label","size"),
        mean_predicted_probability=("probability","mean"),
        observed_phishing_rate=("true_label","mean")
    ).reset_index()
    cal["brier_score_overall"] = brier_score_loss(lr["true_label"], lr["probability"])
    cal.to_csv(OUT/"logreg_calibration.csv", index=False)

print("Saved error-analysis outputs to", OUT)
