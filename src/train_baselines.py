from pathlib import Path
import pandas as pd
import numpy as np
import json, joblib, argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef,
    roc_curve, precision_recall_curve
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "master_dataset.csv"
OUT = ROOT / "results" / "baseline"
SPLITS = ROOT / "data" / "splits"
OUT.mkdir(parents=True, exist_ok=True)
SPLITS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
SEED = args.seed

df = pd.read_csv(DATA, low_memory=False)
df = df[df["text"].notna() & df["phishing_label"].isin([0,1])].copy()
df["phishing_label"] = df["phishing_label"].astype(int)

# 70 / 15 / 15 stratified split
train, temp = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df["phishing_label"]
)
val, test = train_test_split(
    temp, test_size=0.50, random_state=SEED, stratify=temp["phishing_label"]
)

for name, split in [("train",train),("validation",val),("test",test)]:
    split[["record_id"]].to_csv(SPLITS/f"baseline_{name}_ids.csv", index=False)

def tfidf():
    return TfidfVectorizer(
        max_features=80000,
        ngram_range=(1,2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode"
    )

models = {
    "dummy_most_frequent": Pipeline([
        ("tfidf", tfidf()),
        ("clf", DummyClassifier(strategy="most_frequent", random_state=SEED))
    ]),
    "logreg": Pipeline([
        ("tfidf", tfidf()),
        ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=SEED))
    ]),
    "linear_svm": Pipeline([
        ("tfidf", tfidf()),
        ("clf", LinearSVC(class_weight="balanced", random_state=SEED))
    ])
}

metrics_rows, cm_rows, prediction_rows, roc_rows, pr_rows = [], [], [], [], []

def score_vector(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:,1]
    return model.decision_function(x)

def evaluate(model_name, model, split_name, part):
    x = part["text"].astype(str)
    y = part["phishing_label"].to_numpy()
    pred = model.predict(x)
    score = score_vector(model, x)

    p,r,f,_ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y, score)
    except Exception:
        auc = np.nan
    cm = confusion_matrix(y, pred, labels=[0,1])

    metrics_rows.append({
        "model":model_name, "split":split_name, "seed":SEED,
        "n":len(part), "accuracy":accuracy_score(y,pred),
        "balanced_accuracy":balanced_accuracy_score(y,pred),
        "precision":p, "recall":r, "f1":f,
        "roc_auc":auc, "mcc":matthews_corrcoef(y,pred)
    })
    cm_rows.append({
        "model":model_name, "split":split_name,
        "tn":int(cm[0,0]), "fp":int(cm[0,1]),
        "fn":int(cm[1,0]), "tp":int(cm[1,1])
    })
    for rid, true, pp, ss in zip(part["record_id"], y, pred, score):
        prediction_rows.append({
            "record_id":rid, "model":model_name, "split":split_name,
            "true_label":int(true), "predicted_label":int(pp), "score":float(ss)
        })
    if len(np.unique(y)) == 2:
        fpr,tpr,thr = roc_curve(y, score)
        for a,b,c in zip(fpr,tpr,thr):
            roc_rows.append({"model":model_name,"split":split_name,"fpr":float(a),"tpr":float(b),"threshold":float(c)})
        prec,rec,thr2 = precision_recall_curve(y, score)
        thr_pad = list(thr2) + [np.nan]
        for a,b,c in zip(prec,rec,thr_pad):
            pr_rows.append({"model":model_name,"split":split_name,"precision":float(a),"recall":float(b),"threshold":float(c) if not pd.isna(c) else np.nan})

for name, model in models.items():
    print("Training", name, "on", len(train), "emails")
    model.fit(train["text"].astype(str), train["phishing_label"])
    evaluate(name, model, "validation", val)
    evaluate(name, model, "test", test)
    if name != "dummy_most_frequent":
        joblib.dump(model, OUT/f"{name}.joblib")

# Save metrics and predictions
pd.DataFrame(metrics_rows).to_csv(OUT/"baseline_metrics.csv", index=False)
pd.DataFrame(cm_rows).to_csv(OUT/"baseline_confusion_matrices.csv", index=False)
pd.DataFrame(prediction_rows).to_csv(OUT/"baseline_predictions.csv", index=False)
pd.DataFrame(roc_rows).to_csv(OUT/"roc_points.csv", index=False)
pd.DataFrame(pr_rows).to_csv(OUT/"pr_points.csv", index=False)

# Export top weighted terms for the two linear text models
feature_rows = []
for name in ["logreg","linear_svm"]:
    model = models[name]
    vect = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    names = np.array(vect.get_feature_names_out())
    coef = clf.coef_.ravel()
    top_pos = np.argsort(coef)[-30:][::-1]
    top_neg = np.argsort(coef)[:30]
    for rank, idx in enumerate(top_pos, 1):
        feature_rows.append({"model":name,"direction":"phishing","rank":rank,"term":names[idx],"weight":float(coef[idx])})
    for rank, idx in enumerate(top_neg, 1):
        feature_rows.append({"model":name,"direction":"legitimate","rank":rank,"term":names[idx],"weight":float(coef[idx])})
pd.DataFrame(feature_rows).to_csv(OUT/"top_weighted_terms.csv", index=False)

print("\nTest-set metrics")
print(pd.DataFrame(metrics_rows).query("split == 'test'").to_string(index=False))
print("\nSaved models and evaluation artifacts to", OUT)
