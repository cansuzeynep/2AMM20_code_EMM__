"""Check how numeric threshold resolution affects synthetic subgroup recovery."""

from pathlib import Path

import numpy as np
import pandas as pd

from beam_search import beam_search
from quality_measures import setup_quality_fn
from synthetic_data import generate_synthetic_dataset


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union > 0 else 0.0


def main() -> None:
    bin_counts = (5, 10, 20)
    seeds = range(42, 52)
    descriptor_attrs = ["x1", "x2"]
    attrs_type = pd.Series(["Continuous", "Continuous"])
    results = []

    for seed in seeds:
        df = generate_synthetic_dataset(n=5000, seed=seed)
        y = df["y"].to_numpy()
        p_hat = df["p_hat"].to_numpy()
        planted_mask = df["is_planted"].to_numpy(dtype=bool)

        for measure in ("calibration", "squared"):
            quality_fn = setup_quality_fn(y, p_hat, measure)
            for n_bins in bin_counts:
                subgroups = beam_search(
                    df,
                    quality_fn,
                    descriptor_attrs,
                    attrs_type,
                    depth=2,
                    beam_width=20,
                    top_k=5,
                    min_subgroup_size=100,
                    n_bins=n_bins,
                )
                if not subgroups:
                    raise RuntimeError(
                        f"No subgroups returned for {seed=}, {measure=}, {n_bins=}"
                    )

                top_description, top_quality, top_size = subgroups[0]
                recovered_jaccards = [
                    jaccard(description.mask(df), planted_mask)
                    for description, _, _ in subgroups
                ]
                results.append(
                    {
                        "seed": seed,
                        "measure": measure,
                        "n_bins": n_bins,
                        "top1_description": str(top_description),
                        "top1_quality": top_quality,
                        "top1_size": top_size,
                        "top1_jaccard": recovered_jaccards[0],
                        "best_top5_jaccard": max(recovered_jaccards),
                    }
                )

    results_df = pd.DataFrame(results)
    output_path = Path(__file__).with_name("bin_sensitivity_results.csv")
    results_df.to_csv(output_path, index=False, float_format="%.6f")

    summary_df = (
        results_df.groupby(["measure", "n_bins"])
        .agg(
            top1_jaccard_mean=("top1_jaccard", "mean"),
            top1_jaccard_sd=("top1_jaccard", "std"),
            best_top5_jaccard_mean=("best_top5_jaccard", "mean"),
            best_top5_jaccard_sd=("best_top5_jaccard", "std"),
        )
        .reset_index()
    )
    summary_path = Path(__file__).with_name("bin_sensitivity_summary.csv")
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")

    display_df = summary_df.copy()
    for column in display_df.columns[2:]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.3f}")
    print(display_df.to_string(index=False))
    print(f"\nSaved per-seed results to {output_path.name}")
    print(f"Saved aggregate results to {summary_path.name}")


if __name__ == "__main__":
    main()