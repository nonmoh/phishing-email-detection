from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "master_dataset.csv"
BASE_METRICS = ROOT / "results" / "baseline" / "baseline_metrics.csv"
OUT = ROOT / "results" / "baseline" / "cross_generator_phishing_detection.csv"

df = pd.read_csv(DATA, low_memory=False)
df = df[df["text"].notna() & df["phishing_label"].isin([0,1])].copy()
df["phishing_label"] = df["phishing_label"].astype(int)

known = ["gpt-4.1","deepseek3.2","llama-3.3-70b"]

leg = df[df["phishing_label"].eq(0)].sample(frac=1, random_state=42).reset_index(drop=True)
cut = int(len(leg) * 0.70)
leg_train, leg_test = leg.iloc[:cut], leg.iloc[cut:]

def make_model(name):
    common = dict(max_features=80000, ngram_range=(1,2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    if name == "logreg":
        clf = LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)
    elif name == "linear_svm":
        clf = LinearSVC(class_weight="balanced", random_state=42)
    else:
        raise ValueError(name)
    return Pipeline([("tfidf",TfidfVectorizer(**common)),("clf",clf)])

baseline_f1 = {}
if BASE_METRICS.exists():
    bm = pd.read_csv(BASE_METRICS)
    bm = bm[bm["split"].eq("test")]
    baseline_f1 = dict(zip(bm["model"], bm["f1"]))

rows = []
for model_name in ["logreg","linear_svm"]:
    for hold in known:
        test_ph = df[(df["phishing_label"].eq(1)) & (df["generator"].astype(str).str.lower().eq(hold))].copy()
        if test_ph.empty:
            print("No test rows for", hold)
            continue

        train_ph = df[
            (df["phishing_label"].eq(1)) &
            (~df["generator"].astype(str).str.lower().eq(hold))
        ].copy()

        train = pd.concat([leg_train, train_ph], ignore_index=True)
        test = pd.concat([leg_test, test_ph], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

        model = make_model(model_name)
        print("Training", model_name, "| holdout:", hold, "| train:", len(train), "| test:", len(test))
        model.fit(train["text"].astype(str), train["phishing_label"])
        pred = model.predict(test["text"].astype(str))
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(test["text"].astype(str))[:,1]
        else:
            score = model.decision_function(test["text"].astype(str))

        p,r,f,_ = precision_recall_fscore_support(test["phishing_label"], pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(test["phishing_label"], score)
        except Exception:
            auc = np.nan

        id_f1 = baseline_f1.get(model_name, np.nan)
        rel_drop = ((id_f1 - f) / id_f1 * 100.0) if pd.notna(id_f1) and id_f1 != 0 else np.nan

        rows.append({
            "model":model_name,
            "held_out_generator":hold,
            "n_train":len(train),
            "n_test":len(test),
            "n_test_phishing":len(test_ph),
            "accuracy":accuracy_score(test["phishing_label"], pred),
            "balanced_accuracy":balanced_accuracy_score(test["phishing_label"], pred),
            "precision":p,
            "recall":r,
            "f1":f,
            "roc_auc":auc,
            "mcc":matthews_corrcoef(test["phishing_label"], pred),
            "id_test_f1":id_f1,
            "relative_f1_drop_pct":rel_drop
        })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
print("Saved:", OUT)
