"""Population and characteristic stability indices.

With expected (development) and actual (recent) shares ``e_k`` and ``a_k``
across ``K`` bins::

    PSI = sum_k (a_k - e_k) ln(a_k / e_k)

Bins are quantile edges of the *expected* distribution; empty shares are
floored at ``eps`` so the logarithm stays finite. The characteristic
stability index (CSI) is the same statistic applied to a feature.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

MISSING = "missing"


def _numeric_shares(
    expected: NDArray[np.float64], actual: NDArray[np.float64], bins: int, eps: float
) -> pd.DataFrame:
    exp_ok, act_ok = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(exp_ok, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    n_bins = len(edges) + 1
    exp_counts = np.bincount(np.searchsorted(edges, exp_ok, side="left"), minlength=n_bins)
    act_counts = np.bincount(np.searchsorted(edges, act_ok, side="left"), minlength=n_bins)
    bounds = [-np.inf, *edges, np.inf]
    labels = [f"({bounds[i]:.4g}, {bounds[i + 1]:.4g}]" for i in range(n_bins)]
    exp_counts = np.append(exp_counts, np.isnan(expected).sum())
    act_counts = np.append(act_counts, np.isnan(actual).sum())
    labels.append(MISSING)
    keep = (exp_counts + act_counts) > 0
    table = pd.DataFrame(
        {
            "bin": np.asarray(labels)[keep],
            "expected_share": np.maximum(exp_counts[keep] / max(len(expected), 1), eps),
            "actual_share": np.maximum(act_counts[keep] / max(len(actual), 1), eps),
        }
    )
    return table


def _categorical_shares(expected: pd.Series, actual: pd.Series, eps: float) -> pd.DataFrame:
    exp_vals = expected.astype("str").where(expected.notna(), MISSING)
    act_vals = actual.astype("str").where(actual.notna(), MISSING)
    exp_share = exp_vals.value_counts(normalize=True)
    act_share = act_vals.value_counts(normalize=True)
    cats = sorted(set(exp_share.index) | set(act_share.index))
    return pd.DataFrame(
        {
            "bin": cats,
            "expected_share": [max(float(exp_share.get(c, 0.0)), eps) for c in cats],
            "actual_share": [max(float(act_share.get(c, 0.0)), eps) for c in cats],
        }
    )


def psi_table(
    expected: NDArray[np.float64] | pd.Series,
    actual: NDArray[np.float64] | pd.Series,
    bins: int = 10,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Per-bin PSI contributions for a numeric variable.

    Args:
        expected: Development-sample values.
        actual: Recent-sample values.
        bins: Number of quantile bins taken from ``expected``.
        eps: Floor applied to empty shares.

    Returns:
        DataFrame with ``bin``, ``expected_share``, ``actual_share`` and
        ``contribution`` columns.
    """
    table = _numeric_shares(
        np.asarray(expected, dtype=np.float64), np.asarray(actual, dtype=np.float64), bins, eps
    )
    table["contribution"] = (table["actual_share"] - table["expected_share"]) * np.log(
        table["actual_share"] / table["expected_share"]
    )
    return table


def psi(
    expected: NDArray[np.float64] | pd.Series,
    actual: NDArray[np.float64] | pd.Series,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population stability index of a numeric variable (see :func:`psi_table`)."""
    return float(psi_table(expected, actual, bins, eps)["contribution"].sum())


def csi_by_feature(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
    *,
    bins: int = 10,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Characteristic stability index per feature.

    Args:
        expected: Development sample.
        actual: Recent sample.
        numeric_features: Numeric columns (quantile bins).
        categorical_features: Categorical columns (one bin per category).
        bins: Quantile bins for numeric features.
        eps: Floor applied to empty shares.

    Returns:
        DataFrame with ``feature``, ``kind`` and ``csi`` columns sorted by CSI.
    """
    rows: list[dict[str, object]] = [
        {"feature": f, "kind": "numeric", "csi": psi(expected[f], actual[f], bins, eps)}
        for f in numeric_features
    ]
    for f in categorical_features:
        table = _categorical_shares(expected[f], actual[f], eps)
        value = float(
            (
                (table["actual_share"] - table["expected_share"])
                * np.log(table["actual_share"] / table["expected_share"])
            ).sum()
        )
        rows.append({"feature": f, "kind": "categorical", "csi": value})
    return pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)
