"""
Template for running the same q_calib vs q_brier comparison on real UCI datasets
This produces the per-dataset results table in the same style as
EMRM's own Tables 2-5 (description, quality, coverage, mean statistic)
 
NOTE: this script might need network access to archive.ics.uci.edu via the
ucimlrepo package
 
Install first:  pip install ucimlrepo scikit-learn pandas numpy
"""

from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from quality_measures import setup_quality_fn, calculate_residual
from beam_search import beam_search

# UCI dataset IDs (via ucimlrepo)
DATASETS = {
    "heart_disease": 45,  # Cleveland Heart Disease, around 303 records, binary
    "breast_cancer": 17,  # Wisconsin Diagnostic, 569 records, binary
    # "diabetes_130": 296,    # Large-scale
    #                           beam search at around 100k rows will be much slower.
}


def load_uci_binary(dataset_id: int):
    """
    Loads a UCI dataset via ucimlrepo and returns (X, y) with y binarised
    we will likely need to adjust target binarisation per dataset 
    e.g. Heart Disease's target is 0-4 severity, so we collapse >0 to "disease
    present" (1) below
    """
    from ucimlrepo import fetch_ucirepo

    data = fetch_ucirepo(id=dataset_id)
    X = data.data.features.copy()
    X_type = data.variables.type[data.variables['name'].isin(X.columns)].reset_index(drop=True)
    y_raw = data.data.targets.copy().iloc[:, 0]

    # Drop rows with missing values for simplicity, put this in the
    # Experimental Setup section as a stated preprocessing decision, same
    # as Branderhorst's thesis does for its own missing-data handling
    mask = X.notna().all(axis=1) & y_raw.notna()
    X, y_raw = X[mask].reset_index(drop=True), y_raw[mask].reset_index(drop=True)

    # Encode any categorical columns numerically
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == "category":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # binarise target: adjust this rule per dataset
    if y_raw.nunique() > 2:
        y = (y_raw > 0).astype(int)
    else:
        y = LabelEncoder().fit_transform(y_raw)

    return X.astype(float), np.asarray(y), X_type


def get_out_of_fold_predictions(
        X: pd.DataFrame,
        y: np.ndarray,
        n_folds: int = 5
) -> np.ndarray:
    """
    Random Forest predicted probabilities via cross_val_predict, i.e.
    every prediction is made by a model that did NOT see that record during
    training. This is the methodological upgrade over EMRM's in-sample
    residuals discussed in the Methods section, a "high-residual subgroup"
    here reflects genuine generalisation failure, not memorisation
    """
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    proba = cross_val_predict(clf, X, y, cv=n_folds, method="predict_proba")
    return proba[:, 1]  # P(y=1)


def run_one_dataset(
        name: str,
        dataset_id: int,
        min_subgroup_frac: float = 0.05,
        depth: int = 2,
        top_k: int = 3
):
    """

    Args:
        name (str): Name of the UCI Dataset analyzed
        dataset_id (int): ID of the UCI Dataset analyzed
        min_subgroup_frac (float): What fraction of total dataset size a
            subgroup has to be at minimum.
        depth (int): The maximum number of conditions in a description.
        top_k (int): The number of highest quality subgroups that are at most returned

    """
    print(f"\n{'*' * 70}\n{name:^65}\n{'*' * 70}\n")
    X, y, X_type = load_uci_binary(dataset_id)
    p_hat = get_out_of_fold_predictions(X, y)
    min_size = max(10, int(min_subgroup_frac * len(X)))

    for residual in ["calibration", "squared"]:
        print(f"  --- {residual} ---  (min subgroup size = {min_size})\n")
        quality_fn = setup_quality_fn(y, p_hat, residual)
        results = beam_search(
            X, quality_fn, list(X.columns), X_type,
            depth=depth, beam_width=20, top_k=top_k, min_subgroup_size=min_size,
        )

        r_all = calculate_residual(y, p_hat, residual)
        for rank, (desc, q, n) in enumerate(results, start=1):
            mask = desc.mask(X)
            mean_r = r_all[mask].mean()
            print(f"  #{rank}"
                  f"{'Quality':>10}: {'':>8} q ={q:7.3f}\n"
                  f"{'Subgroup size':>20}: {'':>2} n ={n:5d}\n"
                  f"{'Mean residual':>20}: {'':>2} mean_r ={mean_r:+.3f}\n"
                  f"{'Description':>18}: {'':>4} {desc}\n")
 
 
if __name__ == "__main__":
    for name, dataset_id in DATASETS.items():
        run_one_dataset(name, dataset_id)
