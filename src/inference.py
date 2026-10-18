from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable
import math
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "results" / "baseline"
DISTILBERT_DIR = ROOT / "results" / "distilbert_model"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logreg.joblib",
    "Linear SVM": MODEL_DIR / "linear_svm.joblib",
    "DistilBERT": DISTILBERT_DIR,
}


def model_available(model_name: str) -> bool:
    path = MODEL_FILES.get(model_name)
    if path is None:
        return False
    if model_name == "DistilBERT":
        return path.exists() and (path / "config.json").exists()
    return path.exists()


def load_model(model_name: str):
    path = MODEL_FILES.get(model_name)
    if path is None:
        raise ValueError(f"Unknown model: {model_name}")
    if not model_available(model_name):
        if model_name == "DistilBERT":
            raise FileNotFoundError(
                f"DistilBERT model not found at {path}. Train it from START_HERE.ipynb first."
            )
        raise FileNotFoundError(f"Model file not found: {path}. Train baselines first.")

    if model_name == "DistilBERT":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return {
            "kind": "transformer",
            "name": model_name,
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
        }

    return {"kind": "sklearn", "name": model_name, "model": joblib.load(path)}


def build_email_text(subject: str = "", body: str = "") -> str:
    subject = (subject or "").strip()
    body = (body or "").strip()
    if subject and body:
        return f"{subject} {body}"
    return subject or body


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def predict_texts(loaded_model, texts: Iterable[str]) -> Dict[str, Any]:
    texts = [str(t or "") for t in texts]
    if not texts:
        return {"labels": np.array([], dtype=int), "scores": np.array([], dtype=float), "calibrated": False}

    kind = loaded_model.get("kind") if isinstance(loaded_model, dict) else "sklearn"
    model = loaded_model.get("model") if isinstance(loaded_model, dict) else loaded_model

    if kind == "transformer":
        import torch
        tokenizer = loaded_model["tokenizer"]
        device = loaded_model["device"]
        probs_all = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                probs_all.append(probs)
        scores = np.concatenate(probs_all) if probs_all else np.array([], dtype=float)
        labels = (scores >= 0.5).astype(int)
        return {"labels": labels, "scores": scores, "calibrated": False, "score_type": "softmax confidence"}

    labels = model.predict(texts).astype(int)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(texts)[:, 1].astype(float)
        return {"labels": labels, "scores": scores, "calibrated": True, "score_type": "probability"}
    if hasattr(model, "decision_function"):
        margins = np.asarray(model.decision_function(texts), dtype=float)
        scores = 1.0 / (1.0 + np.exp(-margins))
        return {"labels": labels, "scores": scores, "raw_scores": margins, "calibrated": False, "score_type": "display estimate"}
    return {"labels": labels, "scores": np.full(len(labels), np.nan), "calibrated": False, "score_type": "unavailable"}


def predict_text(loaded_model, text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Email text is empty.")
    out = predict_texts(loaded_model, [text])
    label = int(out["labels"][0])
    score = float(out["scores"][0]) if len(out["scores"]) else None
    raw_score = None
    if "raw_scores" in out and len(out["raw_scores"]):
        raw_score = float(out["raw_scores"][0])
    return {
        "label": label,
        "class_name": "Phishing" if label == 1 else "Legitimate",
        "phishing_probability": score,
        "raw_score": raw_score,
        "calibrated_probability": bool(out.get("calibrated", False)),
        "score_type": out.get("score_type", "score"),
    }
