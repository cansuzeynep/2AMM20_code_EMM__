"""
generates a synthetic binary-classification dataset where we control exactly which subgroup is miscalibrated
This gives us the ground truth: 
we know in advance which subgroup our method SHOULD recover, so we can check
whether it actually does

Design:

    - x1, x2 ~ Uniform(0, 1), independent descriptive features
    - true_p(x) = sigmoid(STEEPNESS * (x1 - 0.5))
          This depends only on x1, it is close to 0 for small x1, close to 1
          for large x1, and close to 0.5 (maximally uncertain / noisy) in a
          band around x1 = 0.5.
    - y ~ Bernoulli(true_p(x))
          The actual observed outcome. Even a prefect  model cannot predict
          this exactly when true_p is near 0.5 -- that's irreducible noise
          (Murphy's "Uncertainty" term)
    - p_hat(x), the model's predicted probability we're auditing:
          By default p_hat(x) = true_p(x) EVERYWHERE (perfectly calibrated)
          Except: for the PLANTED subgroup (x2 > MISCAL_THRESHOLD), we add a
          fixed positive bias, so the model is systematically OVERCONFIDENT
          there. This subgroup is defined on x2, which true_p does not depend
          on. So it is a real, clean calibration failure, not just a region
          that happens to be hard to predict.

The dataset also contains a DECOY subgroup for free: "x1 in [0.4, 0.6]" is
a region of high inherent uncertainty (true_p near 0.5) where the model is
still perfectly calibrated (p_hat = true_p there). A squared-residual
(Brier-style) search can be fooled into flagging this decoy, because it has
high mean squared error despite having ZERO miscalibration. A signed-residual
search should NOT be fooled, because positive and negative errors cancel
there on average
"""

import numpy as np
import pandas as pd


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_synthetic_dataset(
    n: int = 5000,
    steepness: float = 10.0,
    miscal_threshold: float = 0.7,
    miscal_bias: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        x1, x2: descriptive attributes (the subgroup search operates on these)
        true_p: the TRUE underlying probability (never given to the search; to generate y and to
                        verify results)
        p_hat: the model's predicted probability (what EMRM-style mining
                        actually gets to see)
        y : the realised binary outcome
        is_planted: 1 if this record belongs to the planted miscalibrated
                        subgroup (x2 > miscal_threshold), else 0. This is our
                        ground truth label for evaluating recovery.
        is_decoy: 1 if this record belongs to the high-uncertainty-but-
                        calibrated decoy subgroup (0.4 <= x1 <= 0.6), else 0
    """
    rng = np.random.default_rng(seed)

    x1 = rng.uniform(0, 1, size=n)
    x2 = rng.uniform(0, 1, size=n)

    true_p = sigmoid(steepness * (x1 - 0.5))

    #start the model's predictions as perfectly calibrated
    p_hat = true_p.copy()

    #then inject a systematic positive bias (overconfidence) ONLY in the
    # planted subgroup
    # np.clip keeps probabilities valid after the shift.
    is_planted = (x2 > miscal_threshold).astype(int)
    p_hat = np.where(is_planted == 1, np.clip(p_hat + miscal_bias, 0, 1), p_hat)

    # Realised outcomes are drawn from the TRUE probability, not from p_hat 
    # p_hat is the (flawed) model's belief, true_p is reality
    y = rng.binomial(1, true_p)

    is_decoy = ((x1 >= 0.4) & (x1 <= 0.6)).astype(int)

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "true_p": true_p,
            "p_hat": p_hat,
            "y": y,
            "is_planted": is_planted,
            "is_decoy": is_decoy,
        }
    )


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    print(df.head())
    print("\nPlanted subgroup size:", df["is_planted"].sum(), "/", len(df))
    print("Decoy subgroup size:   ", df["is_decoy"].sum(), "/", len(df))
    print("\nMean signed residual (y - p_hat):")
    print("  planted subgroup:", (df.loc[df.is_planted == 1, "y"] - df.loc[df.is_planted == 1, "p_hat"]).mean())
    print("  decoy subgroup:  ", (df.loc[df.is_decoy == 1, "y"] - df.loc[df.is_decoy == 1, "p_hat"]).mean())
    print("  whole dataset:   ", (df["y"] - df["p_hat"]).mean())
    print("\nMean squared residual ((y - p_hat)^2), i.e. Brier contribution:")
    print("  planted subgroup:", ((df.loc[df.is_planted == 1, "y"] - df.loc[df.is_planted == 1, "p_hat"]) ** 2).mean())
    print("  decoy subgroup:  ", ((df.loc[df.is_decoy == 1, "y"] - df.loc[df.is_decoy == 1, "p_hat"]) ** 2).mean())
    print("  whole dataset:   ", ((df["y"] - df["p_hat"]) ** 2).mean())