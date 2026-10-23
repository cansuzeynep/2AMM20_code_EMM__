# Mining Subgroups with Exceptional Calibration Residuals
### 2AMM20
**Authors:** Ronan Alberts, Yanislav Herold, Cansu Kars  

---

## Overview

This repository contains the code accompanying the paper *Mining Subgroups with Exceptional Calibration Residuals*. We extend Exceptional Model Residual Mining (EMRM) to binary probabilistic classifiers by scoring subgroups with the mean signed residual $r_i = y_i - \hat{p}_i$, and compare it against the mean squared residual (subgroup Brier score).

The code implements:
- A beam search over conjunctive descriptions with local numeric discretisation
- Two residual quality measures: signed (calibration) and squared (Brier)
- A controlled synthetic experiment with a planted miscalibrated subgroup
- A sensitivity analysis over the number of discretisation bins
- A real-world pipeline over four UCI binary classification datasets

---

## Repository Structure

```
code/
├── beam_search.py               # Beam search over conjunctive descriptions
├── quality_measures.py          # Signed and squared residual quality functions
├── synthetic_data.py            # Synthetic dataset generator (planted subgroup)
├── run_synthetic_experiment.py  # Reproduces the synthetic experiment (Section 5.1)
├── run_bin_sensitivity.py       # Reproduces Table 1: bin sensitivity analysis
├── real_world_data_pipeline.py  # Reproduces case study tables (Section 5.3)
├── bin_sensitivity_results.csv  # Pre-computed per-seed results (seeds 42–51)
├── bin_sensitivity_summary.csv  # Aggregated summary used in Table 1
└── requirements.txt             # Python dependencies
```

---

## Requirements

Python 3.10 or higher is recommended.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scikit-learn`, `ucimlrepo`

> **Note:** `real_world_data_pipeline.py` downloads datasets automatically from the
> UCI Machine Learning Repository via the `ucimlrepo` package and requires an internet
> connection on first run. Subsequent runs use a local cache.

---

## Reproducing the Results

### Synthetic experiment (Section 5.1)
Runs beam search with the signed and squared residual on the seed-42 synthetic dataset
and prints the top-5 subgroups and their Jaccard similarity with the planted subgroup.

```bash
python run_synthetic_experiment.py
```

### Bin sensitivity analysis (Table 1 / Section 5.2)
Repeats the synthetic experiment over seeds 42–51 and bin counts {5, 10, 20}.
Saves `bin_sensitivity_results.csv` (per-seed) and `bin_sensitivity_summary.csv`
(means and standard deviations used in Table 1). Pre-computed CSV files are already
included so you can inspect the results without re-running (~5 minutes on a laptop).

```bash
python run_bin_sensitivity.py
```

### Real-world case studies (Tables 2–5 / Section 5.3)
Loads the four UCI datasets, generates five-fold out-of-fold random-forest predictions,
runs beam search with both residual definitions, and prints the top-3 subgroups per
dataset and residual type. Runtime is approximately 5–10 minutes in total.

```bash
python real_world_data_pipeline.py
```

The four datasets used, with their UCI IDs:

| Dataset | UCI ID |
|---|---|
| Heart Disease (Cleveland) | 45 |
| Breast Cancer Wisconsin (Diagnostic) | 17 |
| Early Stage Diabetes Risk Prediction | 529 |
| Ionosphere | 52 |

---

## Key Design Decisions

| Choice | Value | Where set |
|---|---|---|
| Random forest trees | 200 | `real_world_data_pipeline.py` |
| Random state | 42 | `real_world_data_pipeline.py` |
| Cross-validation folds | 5 | `real_world_data_pipeline.py` |
| Search depth | 2 | all scripts |
| Beam width | 20 | all scripts |
| Discretisation bins | 10 (default) | `beam_search.py` |
| Min subgroup size (real data) | max(10, 5% of dataset) | `real_world_data_pipeline.py` |
| Min subgroup size (synthetic) | 100 | `run_synthetic_experiment.py` |

---

## Notes on Reproducibility

- All random seeds are fixed. Re-running any script produces identical results.
- The real-world pipeline uses `cross_val_predict` with `cv=5` and `random_state=42`,
  so out-of-fold predictions are deterministic.
- UCI dataset contents are retrieved via `ucimlrepo` and may change if the repository
  updates a dataset. The paper results were produced with the dataset versions
  available in October 2026.
