from __future__ import annotations

from pathlib import Path
import io
import email
import pandas as pd
import streamlit as st

from src.inference import (
    ROOT,
    MODEL_FILES,
    load_model,
    build_email_text,
    predict_text,
    predict_texts,
    model_available,
)

st.set_page_config(
    page_title="AI Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 1.4rem;}
      .hero {
        padding: 1.15rem 1.3rem;
        border: 1px solid rgba(120,120,120,.22);
        border-radius: 16px;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, rgba(30,74,120,.10), rgba(30,120,115,.08));
      }
      .hero h1 {margin: 0; font-size: 2rem;}
      .hero p {margin: .35rem 0 0 0; opacity: .82;}
      .result-card {
        padding: 1.2rem 1.3rem;
        border-radius: 16px;
        border: 1px solid rgba(120,120,120,.22);
        margin-top: .6rem;
      }
      .phishing {background: rgba(200, 40, 40, .08);}
      .legitimate {background: rgba(25, 130, 90, .08);}
      .small-note {font-size: .88rem; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🛡️ AI Phishing Email Detector</h1>
      <p>Interactive interface for testing the CS0617 phishing-detection models.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_model(name: str):
    return load_model(name)


def parse_eml(uploaded_file):
    raw = uploaded_file.getvalue()
    msg = email.message_from_bytes(raw)
    subject = msg.get("subject", "") or ""
    body = ""
    if msg.is_multipart():
        chunks = []
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if ctype == "text/plain" and "attachment" not in disp.lower():
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    chunks.append(payload.decode(charset, errors="replace"))
        body = "\n".join(chunks)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="replace")
    return subject, body


available_models = [name for name in MODEL_FILES if model_available(name)]
if not available_models:
    st.error("No trained model files were found. Run `run_baselines.bat` first.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Model", available_models, index=0)
    st.caption("The current baseline models were trained primarily on English email text.")
    st.divider()
    st.markdown("**Project folder**")
    st.code(str(ROOT), language=None)
    st.markdown("**Model file**")
    st.code(str(MODEL_FILES[selected_model].relative_to(ROOT)), language=None)

main_tab, batch_tab, metrics_tab, about_tab = st.tabs(
    ["Single Email", "Batch CSV", "Model Metrics", "About"]
)

with main_tab:
    left, right = st.columns([1.35, 0.9], gap="large")

    with left:
        st.subheader("Test an email")
        uploaded = st.file_uploader(
            "Optional: upload a .txt or .eml file",
            type=["txt", "eml"],
            help="You can also paste the subject and body manually below.",
        )

        default_subject = ""
        default_body = ""
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".eml"):
                    default_subject, default_body = parse_eml(uploaded)
                else:
                    default_body = uploaded.getvalue().decode("utf-8", errors="replace")
            except Exception as exc:
                st.warning(f"Could not parse the uploaded file: {exc}")

        subject = st.text_input("Subject", value=default_subject, placeholder="e.g. Urgent account verification")
        body = st.text_area(
            "Email body",
            value=default_body,
            height=270,
            placeholder="Paste the complete email body here...",
        )

        c1, c2 = st.columns([1, 1])
        analyze = c1.button("Analyze email", type="primary", use_container_width=True)
        clear_hint = c2.button("Clear", use_container_width=True)
        if clear_hint:
            st.rerun()

    with right:
        st.subheader("Prediction")
        if analyze:
            text = build_email_text(subject, body)
            if not text.strip():
                st.warning("Enter a subject, an email body, or upload a file first.")
            else:
                try:
                    model = get_model(selected_model)
                    result = predict_text(model, text)
                    css_class = "phishing" if result["label"] == 1 else "legitimate"
                    icon = "⚠️" if result["label"] == 1 else "✅"
                    st.markdown(
                        f'<div class="result-card {css_class}"><h2>{icon} {result["class_name"]}</h2></div>',
                        unsafe_allow_html=True,
                    )

                    prob = result.get("phishing_probability")
                    if prob is not None:
                        st.metric("Phishing score", f"{prob * 100:.1f}%")
                        st.progress(min(max(float(prob), 0.0), 1.0))
                        if result["calibrated_probability"]:
                            st.caption("Probability from Logistic Regression `predict_proba`.")
                        else:
                            st.caption(
                                "For Linear SVM, this is a sigmoid-transformed decision margin for display only; "
                                "it is not a calibrated probability."
                            )

                    st.markdown("**Input length**")
                    st.write(f"{len(text):,} characters · {len(text.split()):,} words")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
        else:
            st.info("Enter an email on the left and click **Analyze email**.")

    st.divider()
    st.subheader("Compare both baseline models")
    if st.button("Compare models", use_container_width=False):
        text = build_email_text(subject, body)
        if not text.strip():
            st.warning("Enter an email first.")
        else:
            rows = []
            for name in available_models:
                try:
                    res = predict_text(get_model(name), text)
                    rows.append(
                        {
                            "Model": name,
                            "Prediction": res["class_name"],
                            "Phishing score": None
                            if res["phishing_probability"] is None
                            else round(res["phishing_probability"], 4),
                            "Score type": "probability"
                            if res["calibrated_probability"]
                            else "display estimate",
                        }
                    )
                except Exception as exc:
                    rows.append({"Model": name, "Prediction": f"Error: {exc}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with batch_tab:
    st.subheader("Batch testing")
    st.write(
        "Upload a CSV containing either a `text` column, or `subject` and `body` columns. "
        "The app will append model predictions and let you download the result."
    )
    batch_file = st.file_uploader("CSV file", type=["csv"], key="batch_csv")
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            if "text" in batch_df.columns:
                texts = batch_df["text"].fillna("").astype(str)
            elif {"subject", "body"}.issubset(batch_df.columns):
                texts = (
                    batch_df["subject"].fillna("").astype(str).str.strip()
                    + " "
                    + batch_df["body"].fillna("").astype(str).str.strip()
                ).str.strip()
            else:
                st.error("CSV must contain `text`, or both `subject` and `body` columns.")
                texts = None

            if texts is not None:
                st.dataframe(batch_df.head(20), use_container_width=True)
                if st.button("Run batch prediction", type="primary"):
                    model = get_model(selected_model)
                    out = batch_df.copy()
                    pred_out = predict_texts(model, texts.tolist())
                    pred = pred_out["labels"]
                    out["prediction_label"] = pred.astype(int)
                    out["prediction"] = pd.Series(pred).map({0: "Legitimate", 1: "Phishing"})
                    out["phishing_score"] = pred_out["scores"]
                    out["score_type"] = pred_out.get("score_type", "score")
                    st.success(f"Analyzed {len(out):,} rows.")
                    st.dataframe(out.head(50), use_container_width=True)
                    csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "Download predictions CSV",
                        data=csv_bytes,
                        file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Could not process the CSV: {exc}")

with metrics_tab:
    st.subheader("Saved baseline metrics")
    metrics_file = ROOT / "results" / "baseline" / "baseline_metrics.csv"
    cm_file = ROOT / "results" / "baseline" / "baseline_confusion_matrices.csv"
    cross_file = ROOT / "results" / "baseline" / "cross_generator_phishing_detection.csv"

    if metrics_file.exists():
        mdf = pd.read_csv(metrics_file)
        st.dataframe(mdf, use_container_width=True, hide_index=True)
    else:
        st.info("Baseline metrics have not been generated yet.")

    if cm_file.exists():
        st.markdown("**Confusion matrices**")
        st.dataframe(pd.read_csv(cm_file), use_container_width=True, hide_index=True)

    if cross_file.exists():
        st.markdown("**Cross-generator evaluation**")
        st.dataframe(pd.read_csv(cross_file), use_container_width=True, hide_index=True)

with about_tab:
    st.subheader("What this interface tests")
    st.markdown(
        """
        - Binary classification: **Phishing vs. Legitimate**.
        - Baseline models: **TF-IDF + Logistic Regression** and **TF-IDF + Linear SVM**.
        - **DistilBERT** appears automatically after it has been trained from `START_HERE.ipynb`.
        - The interface loads the exact saved model artifacts produced by this project.
        - Subject and body are concatenated in the same style used by the project dataset.
        - This is a research/demo interface, not a production email-security gateway.
        """
    )
    st.warning(
        "Do not describe the Linear SVM display score as a calibrated probability. "
        "Also, current data do not yet support a completed adversarial-register experiment."
    )
