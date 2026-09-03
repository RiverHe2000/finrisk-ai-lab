"""Calibration tests: Hosmer-Lemeshow, binomial and Jeffreys by grade, Brier.

Hosmer-Lemeshow groups observations into ``g`` PD-deciles and compares
observed to expected defaults::

    HL = sum_k (O_k - E_k)^2 / (E_k (1 - E_k / n_k)),   HL ~ chi2(g - 2)

The Jeffreys test (ECB TRIM guide) uses the posterior of the default rate
under a Jeffreys prior, ``Beta(d + 1/2, n - d + 1/2)``; the reported p-value
is ``P(DR <= PD_pred)``, so a small value means the PD underestimates risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import beta, binomtest, chi2

_FLOAT_EPS = 1e-12


def _as_arrays(
    y: NDArray[np.int64] | pd.Series, p: NDArray[np.float64] | pd.Series
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    return np.asarray(y, dtype=np.int64), np.asarray(p, dtype=np.float64)


def brier_score(y: NDArray[np.int64] | pd.Series, p: NDArray[np.float64] | pd.Series) -> float:
    """Mean squared difference between outcome and predicted PD."""
    y_arr, p_arr = _as_arrays(y, p)
    return float(np.mean((p_arr - y_arr) ** 2))


def calibration_table(
    y: NDArray[np.int64] | pd.Series, p: NDArray[np.float64] | pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    """Predicted versus observed default rate per PD quantile bin.

    Args:
        y: Binary outcomes.
        p: Predicted PDs.
        n_bins: Number of quantile bins (fewer if PDs have many ties).

    Returns:
        DataFrame with ``bin``, ``n``, ``defaults``, ``mean_pd``,
        ``observed_rate``, ``expected_defaults`` columns.
    """
    y_arr, p_arr = _as_arrays(y, p)
    edges = np.unique(np.quantile(p_arr, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    bin_idx = np.searchsorted(edges, p_arr, side="right")
    frame = pd.DataFrame({"bin": bin_idx + 1, "y": y_arr, "p": p_arr})
    table = (
        frame.groupby("bin", sort=True)
        .agg(n=("y", "size"), defaults=("y", "sum"), mean_pd=("p", "mean"))
        .reset_index()
    )
    table["observed_rate"] = table["defaults"] / table["n"]
    table["expected_defaults"] = table["mean_pd"] * table["n"]
    return table


def hosmer_lemeshow(
    y: NDArray[np.int64] | pd.Series, p: NDArray[np.float64] | pd.Series, n_groups: int = 10
) -> tuple[float, float, pd.DataFrame]:
    """Hosmer-Lemeshow goodness-of-fit test (formula in the module docstring).

    Args:
        y: Binary outcomes.
        p: Predicted PDs.
        n_groups: Number of PD-quantile groups.

    Returns:
        ``(statistic, p_value, table)`` where ``table`` is the calibration table.
    """
    table = calibration_table(y, p, n_groups)
    n = table["n"].to_numpy(dtype=np.float64)
    expected = table["expected_defaults"].to_numpy(dtype=np.float64)
    observed = table["defaults"].to_numpy(dtype=np.float64)
    denom = np.maximum(expected * (1.0 - expected / n), _FLOAT_EPS)
    statistic = float(np.sum((observed - expected) ** 2 / denom))
    dof = max(len(table) - 2, 1)
    return statistic, float(chi2.sf(statistic, dof)), table


def _grade_frame(
    y: NDArray[np.int64] | pd.Series,
    p: NDArray[np.float64] | pd.Series,
    grade: NDArray[np.int64] | pd.Series,
) -> pd.DataFrame:
    y_arr, p_arr = _as_arrays(y, p)
    frame = pd.DataFrame({"grade": np.asarray(grade, dtype=np.int64), "y": y_arr, "p": p_arr})
    table = (
        frame.groupby("grade", sort=True)
        .agg(n=("y", "size"), defaults=("y", "sum"), predicted_pd=("p", "mean"))
        .reset_index()
    )
    table["observed_rate"] = table["defaults"] / table["n"]
    return table


def binomial_test_by_grade(
    y: NDArray[np.int64] | pd.Series,
    p: NDArray[np.float64] | pd.Series,
    grade: NDArray[np.int64] | pd.Series,
) -> pd.DataFrame:
    """Two-sided exact binomial test of observed defaults against predicted PD per grade.

    Args:
        y: Binary outcomes.
        p: Predicted PDs.
        grade: Grade per row.

    Returns:
        Table with ``grade``, ``n``, ``defaults``, ``predicted_pd``,
        ``observed_rate`` and ``p_value`` columns.
    """
    table = _grade_frame(y, p, grade)
    table["p_value"] = [
        binomtest(int(d), int(n), float(np.clip(pd_, _FLOAT_EPS, 1 - _FLOAT_EPS))).pvalue
        for d, n, pd_ in zip(table["defaults"], table["n"], table["predicted_pd"], strict=True)
    ]
    return table


def jeffreys_test(
    y: NDArray[np.int64] | pd.Series,
    p: NDArray[np.float64] | pd.Series,
    grade: NDArray[np.int64] | pd.Series,
) -> pd.DataFrame:
    """Jeffreys-prior calibration test per grade (see module docstring).

    Args:
        y: Binary outcomes.
        p: Predicted PDs.
        grade: Grade per row.

    Returns:
        Table with ``grade``, ``n``, ``defaults``, ``predicted_pd``,
        ``observed_rate`` and ``p_value`` (``P(DR <= PD)``) columns.
    """
    table = _grade_frame(y, p, grade)
    d = table["defaults"].to_numpy(dtype=np.float64)
    n = table["n"].to_numpy(dtype=np.float64)
    table["p_value"] = beta.cdf(
        table["predicted_pd"].to_numpy(dtype=np.float64), d + 0.5, n - d + 0.5
    )
    return table


def expected_vs_observed_overall(
    y: NDArray[np.int64] | pd.Series, p: NDArray[np.float64] | pd.Series
) -> dict[str, float]:
    """Portfolio-level expected versus observed defaults with a binomial p-value.

    Args:
        y: Binary outcomes.
        p: Predicted PDs.

    Returns:
        Mapping with ``n``, ``expected``, ``observed``, ``mean_pd``,
        ``observed_rate``, ``pd_to_dr_ratio`` and ``p_value``.
    """
    y_arr, p_arr = _as_arrays(y, p)
    n = len(y_arr)
    observed = int(y_arr.sum())
    mean_pd = float(p_arr.mean())
    observed_rate = observed / n
    return {
        "n": float(n),
        "expected": mean_pd * n,
        "observed": float(observed),
        "mean_pd": mean_pd,
        "observed_rate": observed_rate,
        "pd_to_dr_ratio": mean_pd / observed_rate if observed_rate > 0 else float("inf"),
        "p_value": float(
            binomtest(observed, n, float(np.clip(mean_pd, _FLOAT_EPS, 1 - _FLOAT_EPS))).pvalue
        ),
    }
