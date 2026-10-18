from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "processed" / "master_dataset.csv"
BASE = ROOT / "results" / "baseline"
AN = ROOT / "results" / "analysis"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MASTER, low_memory=False)
if "word_length" not in df.columns:
    df["word_length"] = df["text"].astype(str).str.split().str.len()

def save_bar(series, title, xlabel, ylabel, filename, horizontal=False):
    plt.figure(figsize=(9,5))
    if horizontal:
        series.sort_values().plot(kind="barh")
    else:
        series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIG/filename, dpi=170)
    plt.close()

save_bar(df["phishing_label"].map({0:"Legitimate",1:"Phishing"}).value_counts(),
         "Class distribution","Class","Emails","01_class_distribution.png")
save_bar(df["origin"].fillna("unknown").value_counts().head(10),
         "Origin distribution","Origin","Emails","02_origin_distribution.png", horizontal=True)
save_bar(df["generator"].fillna("unknown").value_counts().head(12),
         "Generator distribution","Generator","Emails","03_generator_distribution.png", horizontal=True)
save_bar(df["dataset"].fillna("unknown").value_counts().head(12),
         "Dataset-source distribution","Dataset","Emails","04_dataset_distribution.png", horizontal=True)

plt.figure(figsize=(9,5))
plt.hist(df["word_length"].clip(upper=df["word_length"].quantile(0.99)), bins=50)
plt.title("Email-length distribution (trimmed at 99th percentile)")
plt.xlabel("Words")
plt.ylabel("Emails")
plt.tight_layout()
plt.savefig(FIG/"05_email_length_histogram.png", dpi=170)
plt.close()

metrics_file = BASE/"baseline_metrics.csv"
if metrics_file.exists():
    m = pd.read_csv(metrics_file)
    m = m[m["split"].eq("test")].copy()
    cols = ["accuracy","precision","recall","f1","roc_auc"]
    plot = m.set_index("model")[cols].T
    ax = plot.plot(kind="bar", figsize=(10,5))
    ax.set_title("Held-out test performance")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0,1.03)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG/"06_baseline_metrics.png", dpi=170)
    plt.close()

cm_file = BASE/"baseline_confusion_matrices.csv"
if cm_file.exists():
    cm = pd.read_csv(cm_file)
    cm = cm[cm["split"].eq("test")]
    for _,r in cm.iterrows():
        mat = np.array([[r["tn"],r["fp"]],[r["fn"],r["tp"]]], dtype=int)
        plt.figure(figsize=(5,4))
        plt.imshow(mat)
        plt.title(f"Confusion matrix — {r['model']}")
        plt.xticks([0,1],["Pred Legit","Pred Phish"])
        plt.yticks([0,1],["True Legit","True Phish"])
        for i in range(2):
            for j in range(2):
                plt.text(j,i,str(mat[i,j]),ha="center",va="center")
        plt.tight_layout()
        plt.savefig(FIG/f"07_confusion_{r['model']}.png", dpi=170)
        plt.close()

roc_file = BASE/"roc_points.csv"
if roc_file.exists():
    roc = pd.read_csv(roc_file)
    roc = roc[roc["split"].eq("test")]
    plt.figure(figsize=(6,5))
    for model,g in roc.groupby("model"):
        plt.plot(g["fpr"],g["tpr"],label=model)
    plt.plot([0,1],[0,1],linestyle="--")
    plt.title("ROC curves — held-out test")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG/"08_roc_curves.png", dpi=170)
    plt.close()

pr_file = BASE/"pr_points.csv"
if pr_file.exists():
    pr = pd.read_csv(pr_file)
    pr = pr[pr["split"].eq("test")]
    plt.figure(figsize=(6,5))
    for model,g in pr.groupby("model"):
        plt.plot(g["recall"],g["precision"],label=model)
    plt.title("Precision–Recall curves — held-out test")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG/"09_precision_recall_curves.png", dpi=170)
    plt.close()

cg_file = BASE/"cross_generator_phishing_detection.csv"
if cg_file.exists():
    cg = pd.read_csv(cg_file)
    if not cg.empty:
        pivot = cg.pivot(index="held_out_generator", columns="model", values="f1")
        ax = pivot.plot(kind="bar", figsize=(9,5))
        ax.set_title("Cross-generator F1")
        ax.set_xlabel("Held-out generator")
        ax.set_ylabel("F1")
        ax.set_ylim(0,1.03)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIG/"10_cross_generator_f1.png", dpi=170)
        plt.close()

        pivot2 = cg.pivot(index="held_out_generator", columns="model", values="relative_f1_drop_pct")
        ax = pivot2.plot(kind="bar", figsize=(9,5))
        ax.set_title("Relative F1 change vs. in-distribution test")
        ax.set_xlabel("Held-out generator")
        ax.set_ylabel("Relative F1 drop (%)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIG/"11_relative_f1_drop.png", dpi=170)
        plt.close()

err_file = AN/"error_counts.csv"
if err_file.exists():
    e = pd.read_csv(err_file)
    if not e.empty:
        pivot = e.pivot(index="model", columns="error_type", values="count").fillna(0)
        ax = pivot.plot(kind="bar", figsize=(8,5))
        ax.set_title("False-positive and false-negative counts")
        ax.set_xlabel("Model")
        ax.set_ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIG/"12_error_counts.png", dpi=170)
        plt.close()

cal_file = AN/"logreg_calibration.csv"
if cal_file.exists():
    c = pd.read_csv(cal_file)
    c = c[c["n"] > 0]
    if not c.empty:
        plt.figure(figsize=(6,5))
        plt.plot(c["mean_predicted_probability"], c["observed_phishing_rate"], marker="o")
        plt.plot([0,1],[0,1],linestyle="--")
        plt.title("Logistic Regression calibration")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed phishing rate")
        plt.tight_layout()
        plt.savefig(FIG/"13_logreg_calibration.png", dpi=170)
        plt.close()

terms_file = BASE/"top_weighted_terms.csv"
if terms_file.exists():
    t = pd.read_csv(terms_file)
    for model in ["logreg","linear_svm"]:
        ph = t[(t["model"].eq(model)) & (t["direction"].eq("phishing"))].head(15)
        if not ph.empty:
            plt.figure(figsize=(9,6))
            plt.barh(ph["term"][::-1], ph["weight"][::-1])
            plt.title(f"Top phishing-associated terms — {model}")
            plt.xlabel("Model weight")
            plt.tight_layout()
            plt.savefig(FIG/f"14_top_terms_{model}.png", dpi=170)
            plt.close()

print("Saved figures to", FIG)
