from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "kaggle_large"
OUT.mkdir(parents=True, exist_ok=True)

csvs = list(OUT.rglob("*.csv"))
if csvs:
    print(f"Large dataset already exists ({len(csvs)} CSV file(s)). Skipping download.")
    for p in csvs[:10]:
        print(" -", p)
    raise SystemExit(0)

try:
    import kagglehub
except Exception:
    print("kagglehub is missing. Install requirements_core.txt first.")
    raise

slug = "naserabdullahalam/phishing-email-dataset"
print("Downloading public Kaggle dataset:", slug)
try:
    cache_path = Path(kagglehub.dataset_download(slug))
except Exception as e:
    print("\nAutomatic Kaggle download failed:", repr(e))
    print("Manual fallback:")
    print("1) Download the dataset from Kaggle in your browser.")
    print(r"2) Extract its CSV files into D:\Ai projects\data\raw\kaggle_large")
    print("3) Re-run this notebook from the data-build cell.")
    raise SystemExit(2)

print("Kaggle cache:", cache_path)
for p in cache_path.rglob("*"):
    if p.is_file():
        dest = OUT / p.relative_to(cache_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)

csvs = list(OUT.rglob("*.csv"))
print(f"Copied {len(csvs)} CSV file(s) to:", OUT)
for p in csvs[:10]:
    print(" -", p)
