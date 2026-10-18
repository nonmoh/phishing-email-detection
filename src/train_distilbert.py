from __future__ import annotations

from pathlib import Path
import inspect
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "master_dataset.csv"
if not DATA.exists():
    DATA = ROOT / "data" / "processed" / "uploaded_unified_dedup.csv"
OUT = ROOT / "results" / "distilbert_model"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def main(epochs: float = 2.0, batch_size: int = 8, max_length: int = 256):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(SEED)
    df = pd.read_csv(DATA)
    df = df[df["text"].notna() & df["phishing_label"].isin([0, 1])].copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["phishing_label"].astype(int)

    train_df, temp_df = train_test_split(
        df[["text", "label", "record_id"]],
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    split_dir = ROOT / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df[["record_id"]].to_csv(split_dir / "distilbert_train_ids.csv", index=False)
    val_df[["record_id"]].to_csv(split_dir / "distilbert_validation_ids.csv", index=False)
    test_df[["record_id"]].to_csv(split_dir / "distilbert_test_ids.csv", index=False)

    local_base = ROOT / "models" / "distilbert-base-uncased"
    model_name = str(local_base) if (local_base / "config.json").exists() else "distilbert-base-uncased"
    print("Using DistilBERT base:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def to_ds(frame):
        ds = Dataset.from_pandas(frame[["text", "label"]], preserve_index=False)
        return ds.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
            batched=True,
        )

    train_ds, val_ds, test_ds = map(to_ds, [train_df, val_df, test_df])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        logits = np.asarray(logits)
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = ex[:, 1] / ex.sum(axis=1)
        pred = (prob >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        return {
            "accuracy": accuracy_score(y_true, pred),
            "precision": p,
            "recall": r,
            "f1": f1,
            "roc_auc": roc_auc_score(y_true, prob),
            "pr_auc": average_precision_score(y_true, prob),
        }

    kwargs = dict(
        output_dir=str(OUT / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        report_to=[],
        logging_steps=50,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    args = TrainingArguments(**kwargs)

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    tsig = inspect.signature(Trainer.__init__)
    if "processing_class" in tsig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in tsig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate(test_ds)
    trainer.save_model(str(OUT))
    tokenizer.save_pretrained(str(OUT))

    clean = {k.replace("eval_", ""): float(v) for k, v in metrics.items() if k.startswith("eval_")}
    clean.update({"n_train": len(train_df), "n_validation": len(val_df), "n_test": len(test_df)})
    pd.DataFrame([clean]).to_csv(ROOT / "results" / "distilbert_metrics.csv", index=False)
    (ROOT / "results" / "distilbert_metrics.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    print(json.dumps(clean, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    a = parser.parse_args()
    main(a.epochs, a.batch_size, a.max_length)
