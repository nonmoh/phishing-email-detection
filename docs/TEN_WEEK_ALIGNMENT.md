# CS0617 10-Week Alignment

This enhanced package maps directly to the course roadmap.

## Week 1 — Problem Definition & Scoping
- Precise application domain and AI task
- Final implemented research question
- Hypotheses / expected outcomes
- Constraints and out-of-scope items
- Initial experiment plan

## Weeks 2–3 — Literature & Positioning
- Related-work matrix (existing project literature matrix)
- Comparison of datasets, tasks, metrics and baselines
- Gap framed around robustness/generalization rather than “using model X”

## Week 4 — Data Acquisition, Understanding & Quality
- Automatic Kaggle large-dataset acquisition
- Dataset statistics and EDA charts
- Class balance, origin, generator and source distributions
- Missing/duplicate/conflicting-label checks
- SHA-256 exact-text deduplication
- Fixed preprocessing and data dictionary

## Week 5 — Baselines & Experimental Setup
- DummyClassifier trivial baseline
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Fixed 70/15/15 stratified splits
- Seed = 42
- Validation/test metrics, saved split IDs, model artifacts and configuration

## Weeks 6–7 — Core Experiments
- In-distribution held-out test
- Cross-generator leave-one-generator-out evaluation
- Both Logistic Regression and Linear SVM
- Relative F1 drop vs. ID performance
- Main result tables and figures

## Weeks 8–9 — Error Analysis & Interpretation
- False positives / false negatives
- Subgroup metrics by origin, dataset, generator and email length
- Logistic Regression calibration table/plot
- Top weighted lexical features
- Limitations and threats to validity

## Week 10 — Writing, Reproducibility & Final Delivery
- Notebook-first reproducible workflow
- Saved configuration, split IDs and model artifacts
- Streamlit interface
- Visual-ready CSV outputs and figures
- Final report and presentation assets

## Important scope decision
The current data support cross-generator robustness, but not genuine adversarial writing-register labels.
Therefore, this package does NOT claim completion of the combined generator + register experiment.
That experiment remains future work unless a provenance-controlled register-labelled dataset is added.
