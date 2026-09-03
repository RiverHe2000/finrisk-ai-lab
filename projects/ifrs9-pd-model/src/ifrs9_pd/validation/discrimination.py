"""Discriminatory-power statistics: AUC, Gini, KS, CAP and rank ordering."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import rankdata

Metric = Callable[[NDArray[np.int64], NDArray[np.float64]], float]


def _as_arrays(
    y: NDArray[np.int64] | pd.Series, p: NDArray[np.float64] | pd.Series
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    y_arr = np.asarray(y, dtype=np.int64)
    p_arr = np.asarray(p, dtype=np.float64)
    if y_arr.shape != p_arr.shape:
        msg = "y and p must have the same shape"
        raise ValueError(msg)
    return y_arr, p_arr


def auc(y: NDArray[np.int64] | pd.Series, p: NDArray[np.float64] | pd.Series) -> float:
    """Area under the ROC curve via the Mann-Whitney rank statistic (ties averaged).

    Args:
        y: Binary outcomes.
        p: Scores where higher means riskier.

    Returns:
        AUC in ``[0, 1]``; ``nan`` if only one class is present.
    """
    y_arr, p_arr = _as_arrays(y, p)
    n_pos = int(y_arr.sum())
    n_neg = len(y_arr) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p_arr)
    return float((ranks[y_arr == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def gini(y: NDArray[np.int64] | pd.Series, p: NDArray[np.float64] | pd.Series) -> float:
    """Gini coefficient (accuracy ratio) ``2 AUC - 1``."""
    return 2.0 * auc(y, p) - 1.0


def ks_statistic(y: NDArray[np.int64] | pd.Series, p: NDArray[np.float64] | pd.Series) -> float:
    """Kolmogorov-Smirnov statistic ``max |F_bad(s) - F_good(s)|`` over score cut-offs."""
    y_arr, p_arr = _as_arrays(y, p)
    n_pos = int(y_arr.sum())
    n_neg = len(y_arr) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p_arr, kind="stable")
    cum_bad = np.cumsum(y_arr[order]) / n_pos
    cum_good = np.cumsum(1 - y_arr[order]) / n_neg
    return float(np.max(np.abs(cum_bad - cum_good)))


def bootstrap_ci(
    metric_fn: Metric,
    y: NDArray[np.int64] | pd.Series,
    p: NDArray[np.float64] | pd.Series,
    *,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of a metric.

    Args:
        metric_fn: Function ``(y, p) -> float``.
        y: Binary outcomes.
        p: Scores.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.
        alpha: Two-sided significance level.

    Returns:
        ``(lower, upper)`` percentile bounds.
    """
    y_arr, p_arr = _as_arrays(y, p)
    rng = np.random.default_rng(seed)
    n = len(y_arr)
    values = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        values[i] = metric_fn(y_arr[idx], p_arr[idx])
    values = values[~np.isnan(values)]
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def cap_curve(
    y: NDArray[np.int64] | pd.Series, p: NDArray[np.float64] | pd.Series, n_points: int = 101
) -> tuple[pd.DataFrame, float]:
    """Cumulative accuracy profile and accuracy ratio.

    Args:
        y: Binary outcomes.
        p: Scores where higher means riskier.
        n_points: Number of points on the curve (population share grid).

    Returns:
        A DataFrame with ``population_share`` and ``default_share`` columns,
        and the accuracy ratio (area between model and random over the area
        between perfect and random), which equals the Gini coefficient.
    """
    y_arr, p_arr = _as_arrays(y, p)
    order = np.argsort(-p_arr, kind="stable")
    cum_defaults = np.concatenate([[0.0], np.cumsum(y_arr[order]) / max(int(y_arr.sum()), 1)])
    grid = np.linspace(0.0, 1.0, n_points)
    positions = np.round(grid * len(y_arr)).astype(np.int64)
    curve = pd.DataFrame({"population_share": grid, "default_share": cum_defaults[positions]})
    return curve, gini(y_arr, p_arr)


def rank_ordering(
    y: NDArray[np.int64] | pd.Series,
    grade: NDArray[np.int64] | pd.Series,
    min_defaults: int = 5,
) -> tuple[pd.DataFrame, bool]:
    """Observed default rate per grade and whether it is non-decreasing in grade.

    A decrease between adjacent grades only counts as a violation when both
    grades hold at least ``min_defaults`` defaults; with fewer defaults the
    observed rate is dominated by sampling noise.

    Args:
        y: Binary outcomes.
        grade: Rating grade per row (1 = lowest risk).
        min_defaults: Minimum defaults in both grades for a violation to count.

    Returns:
        A table with ``grade``, ``n``, ``defaults``, ``default_rate`` columns
        and a boolean that is true when default rates are monotonic.
    """
    frame = pd.DataFrame(
        {"grade": np.asarray(grade, dtype=np.int64), "y": np.asarray(y, dtype=np.int64)}
    )
    table = frame.groupby("grade", sort=True)["y"].agg(n="size", defaults="sum").reset_index()
    table["default_rate"] = table["defaults"] / table["n"]
    rates = table["default_rate"].to_numpy(dtype=np.float64)
    defaults = table["defaults"].to_numpy(dtype=np.int64)
    reliable = (defaults[:-1] >= min_defaults) & (defaults[1:] >= min_defaults)
    violation = (np.diff(rates) < 0.0) & reliable
    return table, not bool(violation.any())
