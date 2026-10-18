from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results" / "visual_ready"
DEST.mkdir(parents=True, exist_ok=True)

sources = [
    ROOT/"results"/"baseline"/"baseline_metrics.csv",
    ROOT/"results"/"baseline"/"baseline_confusion_matrices.csv",
    ROOT/"results"/"baseline"/"baseline_predictions.csv",
    ROOT/"results"/"baseline"/"cross_generator_phishing_detection.csv",
    ROOT/"results"/"baseline"/"top_weighted_terms.csv",
    ROOT/"results"/"analysis"/"subgroup_metrics.csv",
    ROOT/"results"/"analysis"/"error_examples.csv",
    ROOT/"results"/"analysis"/"error_counts.csv",
    ROOT/"results"/"analysis"/"logreg_calibration.csv",
    ROOT/"data"/"processed"/"master_dataset_stats.json",
]
for src in sources:
    if src.exists():
        shutil.copy2(src, DEST/src.name)
print("Visual-ready tables copied to", DEST)
