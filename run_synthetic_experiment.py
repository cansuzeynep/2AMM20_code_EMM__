"""
runs beam search twice on the same synthetic data 
once with our proposed q_calib (signed residual), once with q_brier
(squared residual, EMRM's original choice)
and checks which one actually recovers the subgroup we planted

Recovery is measured by Jaccard overlap between the discovered subgroup's
covered records and the true planted subgroup's records

Jaccard = 1.0 means perfect recovery. Jaccard = 0.0 means no overlap at all
"""

import numpy as np
from functools import partial

from synthetic_data import generate_synthetic_dataset
from quality_measures import q_calib, q_brier
from beam_search import beam_search


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union > 0 else 0.0


def main():
    df = generate_synthetic_dataset(n=5000, seed=42)
    y = df["y"].values
    p_hat = df["p_hat"].values
    true_planted_mask = df["is_planted"].values.astype(bool)
    descriptor_attrs = ["x1", "x2"]  # the model is NOT allowed to see true_p or y as descriptors

    print("=" * 70)
    print("SEARCH 1: q_calib (signed residual, our proposed measure)")
    print("=" * 70)
    quality_fn_calib = partial(q_calib, y, p_hat)
    results_calib = beam_search(
        df, quality_fn_calib, descriptor_attrs,
        depth=2, beam_width=20, top_k=5, min_subgroup_size=100,
    )
    for rank, (desc, q, n) in enumerate(results_calib, start=1):
        mask = desc.mask(df)
        jac = jaccard(mask, true_planted_mask)
        print(f"  #{rank}  q={q:7.3f}  n={n:5d} Jaccard-vs-truth={jac:.3f}   {desc}")

    print()
    print("=" * 70)
    print("SEARCH 2: q_brier (squared residual, EMRM's original choice)")
    print("=" * 70)
    quality_fn_brier = partial(q_brier, y, p_hat)
    results_brier = beam_search(
        df, quality_fn_brier, descriptor_attrs,
        depth=2, beam_width=20, top_k=5, min_subgroup_size=100,
    )
    for rank, (desc, q, n) in enumerate(results_brier, start=1):
        mask = desc.mask(df)
        jac = jaccard(mask, true_planted_mask)
        print(f"  #{rank}  q={q:7.3f}  n={n:5d}  Jaccard-vs-truth={jac:.3f}   {desc}")

    print()
    print("*" * 70)
    print("SUMMARY")
    print("*" * 70)
    top1_calib_jaccard = jaccard(results_calib[0][0].mask(df), true_planted_mask)
    top1_brier_jaccard = jaccard(results_brier[0][0].mask(df), true_planted_mask)
    print(f"Top-1 recovery (Jaccard), q_calib: {top1_calib_jaccard:.3f}")
    print(f"Top-1 recovery (Jaccard), q_brier: {top1_brier_jaccard:.3f}")
    print(f"\nTrue planted subgroup was: x2 > 0.7   (n={true_planted_mask.sum()})")


if __name__ == "__main__":
    main()