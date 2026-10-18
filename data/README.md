# Data

The datasets are **not stored in this repository** (size and third-party licensing).
Run `DOWNLOAD_ALL_DATA_ONLY.bat` (or `python bootstrap_everything.py`) to download them and rebuild `data/processed/master_dataset.csv`.

| Source | Size | License |
|---|---|---|
| [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) (Kaggle) | ~82.5K emails | CC BY-SA 4.0 |
| [Human-LLM generated phishing-legitimate emails](https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails) (Kaggle) | 4,000 emails | CC BY-NC-SA 4.0 |
| Cross-model evaluation of phishing detectors against LLM-generated emails — Gutierrez, Villegas-Ch & Govea (2026), Zenodo, DOI [10.5281/zenodo.20250116](https://doi.org/10.5281/zenodo.20250116) | 9,986 emails | CC BY 4.0 |

See `docs/DATA_DICTIONARY.md` for the unified schema.
