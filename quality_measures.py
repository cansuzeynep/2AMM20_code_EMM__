from typing import Callable
import numpy as np

def setup_quality_fn(y: np.ndarray, p: np.ndarray, residual: str) -> Callable[[np.ndarray], float]:
    """ Closure function: Sets up the static environment for the quality_fn.

    Calculates the residuals of the full dataset true outcomes and predicted
    outcome probabilities. Return the function 'quality_fn'.

    Args:
        y (np.ndarray): Array that contains the true outcomes
        p (np.ndarray): Array that contains the predicted outcome probabilities
        residual (str): String that defines which residual calculation method to use.

    Returns:
        quality_fn (function): Function that calculates the residual exceptionality
            score if a particular subgroup

    Raises:
        ValueError: If 'y' and 'p' do not have the same shape.

    """
    if y.shape != p.shape:
        raise ValueError("'y' and 'p' array size mismatch")

    # Calculate the residual given specified type
    r = calculate_residual(y, p, residual)

    # Calculate global residual mean and variance
    r_g_mean = r.mean()
    r_g_var = np.var(r)
    if r_g_var == 0:
        raise ValueError("Global residual variance is zero; quality is undefined.")

    # Create inner function
    def quality_fn(mask: np.ndarray) -> float:
        """ Calculates and returns the quality score of a subgroup.
        Based on the residual exceptionality score by Mishra et al. (2026)

        Args:
            mask (np.ndarray): Array that contains the mask a subgroup

        Returns:
            q_res (float): Residual exceptionality score of subgroup

        """
        if mask.shape != r.shape:
            raise ValueError("'r' and 'mask' array size mismatch")

        # Calculate subgroup residual mean
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("Cannot calculate quality for an empty subgroup.")
        r_s = r * mask
        r_s_mean = r_s.sum() / mask_sum

        # Calculate quality
        q_res = abs((r_s_mean - r_g_mean)) / np.sqrt(r_g_var / mask_sum)

        return q_res

    return quality_fn

def calculate_residual(y: np.ndarray, p: np.ndarray, residual: str) -> np.ndarray:
    """ Calculates the array of residuals given the true outcomes and predicted
        outcome probabilities.

    Args:
        y (np.ndarray): Array that contains the true outcomes
        p (np.ndarray): Array that contains the predicted outcome probabilities
        residual (str): String that defines which residual calculation method to use.

    Returns:
        np.ndarray: Array that contains the residuals

    Raises:
        ValueError: If an invalid residual type is passed

    """
    residual = residual.lower().strip()

    if residual == "calibration":
        return y - p
    if residual == "squared":
        return (y - p) ** 2

    raise ValueError(f"Unknown residual: {residual}.")
