"""
Helpers for imputation methods.

This module implements a Python equivalent of an R routine that imputes
left-censored missing values per column using a truncated normal model.

Algorithm (per column):
- Estimate mean and sd by regressing empirical quantiles of the observed data
  on theoretical standard-normal quantiles (Q–Q linear fit) over [0.001, 0.99].
- Let pNAs be the fraction of NAs in the column. Set an upper cut at the
  (pNAs + 0.001)-quantile of N(mean, sd).
- Impute NAs with random draws from N(mean, sd * tune_sigma) truncated above
  at that cut (i.e., draw from the left tail).

If the fit is unstable (few non-NA values, or zero/negative slope), a fallback
imputer using the column's finite values (median and MAD-like scale) is used.
"""

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm

@dataclass
class QQFit:
    mean: float
    sd: float
    n_obs: int


def _quantile_seq(start: float, stop: float, step: float) -> np.ndarray:
    """Replicates R's seq(start, stop, by=step) inclusive behavior for floats."""
    if step <= 0:
        return np.array([stop], dtype=float)
    n = max(1, int(math.floor((stop - start) / step + 0.5)) + 1)
    # build then clip to stop due to float accumulations
    seq = start + step * np.arange(n, dtype=float)
    seq[seq > stop] = stop
    # ensure last is exactly stop
    if len(seq) > 0:
        seq[-1] = stop
    return seq


def _qq_fit_normal(y_observed: np.ndarray, upper_q: float = 0.99):
    """Return a closure that fits mean/sd for a given pNAs using QQ regression.

    The R code ties the theoretical probs to pNAs, so we produce a function
    fit_for_pnas(pnas) -> Optional[QQFit]. Returns None if not enough data.
    """
    y = y_observed[~np.isnan(y_observed)]
    if y.size < 5:
        return None

    # Empirical quantiles at probs 0.001, 0.011, ..., upper_q+0.001 (~100 points)
    probs_emp = _quantile_seq(0.001, upper_q + 0.001, 0.01)
    # Use default quantile method for broad NumPy compatibility
    q_emp = np.quantile(y, probs_emp)

    def fit_for_pnas(pnas: float) -> Optional[QQFit]:
        start = float(pnas) + 0.001
        stop = upper_q + 0.001
        step = (upper_q - float(pnas)) / (upper_q * 100.0) if upper_q > 0 else 0.0
        if step <= 0 or start >= stop:
            return None
        probs_theor = _quantile_seq(start, stop, step)
        # Align lengths (robust to rounding differences)
        m = min(probs_theor.shape[0], q_emp.shape[0])
        probs_theor = probs_theor[:m]
        q_emp_fit = q_emp[:m]
        q_theor = norm.ppf(probs_theor, loc=0.0, scale=1.0)

        # OLS fit: q_emp_fit = a + b * q_theor
        X = np.vstack([np.ones_like(q_theor), q_theor]).T
        try:
            coef, *_ = np.linalg.lstsq(X, q_emp_fit, rcond=None)
            a, b = float(coef[0]), float(coef[1])
        except Exception:
            return None

        sd = abs(b)
        if not np.isfinite(a) or not np.isfinite(sd) or sd == 0:
            return None
        return QQFit(mean=a, sd=sd, n_obs=y.size)

    return fit_for_pnas

def _r_qnorm(p: float, mean: float, sd: float) -> float:
    """R's qnorm equivalent (inverse CDF of Normal). Requires SciPy."""
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(norm.ppf(p, loc=mean, scale=sd))

def _sample_left_trunc_normal(n: int, mean: float, sd: float, upper: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n values from N(mean, sd) truncated above at `upper` (left tail) using SciPy."""
    sd = float(abs(sd))
    if sd == 0 or not np.isfinite(sd):
        return np.full(n, fill_value=min(mean, upper))
    a = -10.0  # approx -inf (in sd units)
    b = (upper - mean) / sd
    if b <= a:
        return np.full(n, fill_value=min(mean, upper))
    return truncnorm.rvs(a=a, b=b, loc=mean, scale=sd, size=n, random_state=rng)


def impute_left_censored(
    data: pd.DataFrame,
    tune_sigma: float = 1.0,
    seed: Optional[int] = 123,
    axis: str = "columns",
) -> pd.DataFrame:
    """Impute NAs per column using a left-tail truncated normal, R-style.

    Parameters
    - data: 2D pandas DataFrame (features x samples).
    - tune_sigma: multiplier on the fitted sd for the imputation draws (>=0).
    - seed: int seed for reproducible random sampling (None = non-deterministic).
    - axis: "columns" (default, like R code; per column) or "rows" (per row).

    Returns
    - pandas DataFrame with imputed values.
    """
    rng = np.random.default_rng(seed)
    UPPER_Q = 0.99  # Fixed upper quantile used for QQ fit, matching the R code

    values = data.values.astype(float, copy=True)
    index = data.index
    columns = data.columns
    if values.ndim != 2:
        raise ValueError("data must be 2D (features x samples)")

    # Decide orientation: we always iterate over columns of `work`
    work = values if axis == "columns" else values.T

    n_features, n_samples = work.shape

    for j in range(n_samples):
        col = work[:, j]
        mask_na = np.isnan(col)
        pnas = float(np.mean(mask_na)) if col.size > 0 else 0.0

        qq_fit_closure = _qq_fit_normal(col, upper_q=UPPER_Q)
        mean_j: Optional[float] = None
        sd_j: Optional[float] = None
        if callable(qq_fit_closure):
            fit = qq_fit_closure(pnas)  # type: ignore[misc]
            if isinstance(fit, QQFit):
                mean_j, sd_j = fit.mean, fit.sd

        # Fallback if fit is None
        if mean_j is None or sd_j is None:
            finite = col[~mask_na]
            if finite.size >= 1:
                med = float(np.median(finite))
                mad = float(np.median(np.abs(finite - med)))
                sd_fallback = mad / 0.6745 if mad > 0 else float(np.std(finite))
                sd_fallback = abs(sd_fallback) if np.isfinite(sd_fallback) and sd_fallback > 0 else 1.0
                mean_j = med
                sd_j = sd_fallback
            else:
                # Entire column NA: use global constants
                mean_j = 0.0
                sd_j = 1.0

        # Compute upper truncation point at qnorm(pNAs+0.001, mean, sd)
        upper_cut = _r_qnorm(pnas + 0.001, mean=mean_j, sd=sd_j)

        # Adjust sigma as in the R code: sd * tune_sigma (not squared)
        sd_imp = abs(sd_j) * float(tune_sigma)
        if not np.isfinite(sd_imp) or sd_imp == 0:
            sd_imp = 1e-8

        # Draw n_features values, but we'll only use them for NA positions
        draws = _sample_left_trunc_normal(n_features, mean=mean_j, sd=sd_imp, upper=upper_cut, rng=rng)

        # Impute
        if np.any(mask_na):
            col_imputed = col.copy()
            col_imputed[mask_na] = draws[mask_na]
            work[:, j] = col_imputed

    # Restore original orientation
    out_values = work if axis == "columns" else work.T

    # Rebuild output DataFrame
    result = pd.DataFrame(out_values, index=index, columns=columns)

    return result


