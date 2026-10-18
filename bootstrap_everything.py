from __future__ import annotations

from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
import webbrowser

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
IS_WIN = os.name == "nt"
VENV_PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def run(cmd, *, check=True, cwd=ROOT):
    print("\n>", " ".join(str(x) for x in cmd))
    return subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=check)


def in_project_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == VENV_PY.resolve()
    except Exception:
        return False


def create_venv_and_relaunch(args):
    if not VENV_PY.exists():
        print("Creating virtual environment:", VENV)
        try:
            run([sys.executable, "-m", "venv", str(VENV)])
        except subprocess.CalledProcessError:
            print("First venv attempt failed; retrying with --clear ...")
            run([sys.executable, "-m", "venv", "--clear", str(VENV)])

    if not VENV_PY.exists():
        raise SystemExit("Could not create .venv. Install Python 3.11+ with the venv component and rerun.")

    cmd = [str(VENV_PY), str(Path(__file__).resolve()), "--inside-venv"]
    if args.force_data:
        cmd.append("--force-data")
    if args.preload_distilbert:
        cmd.append("--preload-distilbert")
    if args.train:
        cmd.append("--train")
    if args.open_notebook:
        cmd.append("--open-notebook")
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


def install_all():
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements_all.txt")])


def register_kernel():
    run([
        sys.executable, "-m", "ipykernel", "install", "--user",
        "--name", "ai-phishing-project",
        "--display-name", "AI Phishing Project (.venv)"
    ], check=False)


def download_and_prepare(force=False):
    cmd = [sys.executable, str(ROOT / "src" / "download_all_project_data.py")]
    if force:
        cmd.append("--force")
    run(cmd)
    run([sys.executable, str(ROOT / "src" / "prepare_web_sources.py")])
    run([sys.executable, str(ROOT / "src" / "build_master_dataset.py")])


def preload_distilbert():
    print("\nPreloading the base DistilBERT model from Hugging Face (not phishing-trained)...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dest = ROOT / "models" / "distilbert-base-uncased"
    dest.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    tok.save_pretrained(dest)
    model.save_pretrained(dest)
    print("Base model saved to:", dest)


def run_training_pipeline():
    run([sys.executable, str(ROOT / "src" / "train_baselines.py"), "--seed", "42"])
    run([sys.executable, str(ROOT / "src" / "evaluate_cross_generator.py")])
    run([sys.executable, str(ROOT / "src" / "error_analysis.py")])
    run([sys.executable, str(ROOT / "src" / "generate_figures.py")])
    run([sys.executable, str(ROOT / "src" / "export_visual_ready.py")])


def verify():
    mods = [
        "pandas", "numpy", "scipy", "sklearn", "matplotlib", "joblib", "yaml",
        "requests", "kagglehub", "streamlit", "jupyterlab", "torch", "transformers",
        "datasets", "accelerate"
    ]
    failures = []
    for mod in mods:
        r = subprocess.run([sys.executable, "-c", f"import {mod}; print('{mod}: OK')"], cwd=str(ROOT))
        if r.returncode:
            failures.append(mod)
    if failures:
        raise SystemExit("Import verification failed: " + ", ".join(failures))

    required = [
        ROOT / "data" / "raw" / "kaggle_large",
        ROOT / "data" / "raw" / "four_class_web",
        ROOT / "data" / "raw" / "cross_model_web",
        ROOT / "data" / "processed" / "uploaded_unified_dedup.csv",
        ROOT / "data" / "processed" / "master_dataset.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Data verification failed. Missing:\n" + "\n".join(missing))


def open_jupyter():
    notebook = ROOT / "START_HERE_LARGE_10WEEKS.ipynb"
    print("Opening JupyterLab with:", notebook)
    subprocess.Popen([
        sys.executable, "-m", "jupyter", "lab", str(notebook),
        "--notebook-dir", str(ROOT)
    ], cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="One-click full setup for the AI Phishing project.")
    ap.add_argument("--inside-venv", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--force-data", action="store_true", help="Redownload all public datasets.")
    ap.add_argument("--preload-distilbert", action="store_true", help="Also download base DistilBERT weights.")
    ap.add_argument("--train", action="store_true", help="Train/evaluate classical models after setup.")
    ap.add_argument("--open-notebook", action="store_true", help="Open JupyterLab at the end.")
    args = ap.parse_args()

    print("=" * 72)
    print("AI PHISHING PROJECT — COMPLETE INTERNET BOOTSTRAP")
    print("Project folder:", ROOT)
    print("Python:", sys.executable)
    print("=" * 72)

    if not args.inside_venv and not in_project_venv():
        create_venv_and_relaunch(args)

    install_all()
    register_kernel()
    download_and_prepare(force=args.force_data)

    if args.preload_distilbert:
        preload_distilbert()

    if args.train:
        run_training_pipeline()

    verify()

    print("\n" + "=" * 72)
    print("SETUP COMPLETE")
    print("Virtual environment:", VENV)
    print("Master dataset:", ROOT / "data" / "processed" / "master_dataset.csv")
    print("Notebook:", ROOT / "START_HERE_LARGE_10WEEKS.ipynb")
    print("Interface command:")
    print(f'  "{VENV_PY}" -m streamlit run "{ROOT / "app.py"}"')
    print("=" * 72)

    if args.open_notebook:
        open_jupyter()


if __name__ == "__main__":
    main()
