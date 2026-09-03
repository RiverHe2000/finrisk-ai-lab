"""Out-of-time backtest comparing development and OOT performance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ifrs9_pd.validation.calibration import brier_score
from ifrs9_pd.validation.discrimination import auc, gini, ks_statistic
from ifrs9_pd.validation.stability import psi
from ifrs9_pd.validation.thresholds import traffic_light

__all__ = ["ScoredSample", "out_of_time_backtest", "traffic_light"]


@dataclass(frozen=True)
class ScoredSample:
    """Outcomes, PDs and scorecard points of one sample.

    Attributes:
        y: Binary outcomes.
        pd: Predicted 12-month PD.
        score: Scorecard points.
    """

    y: NDArray[np.int64]
    pd: NDArray[np.float64]
    score: NDArray[np.float64]


def out_of_time_backtest(
    dev: ScoredSample, oot: ScoredSample, psi_bins: int = 10
) -> dict[str, float]:
    """Compare discrimination, calibration and score stability across samples.

    Args:
        dev: Development sample.
        oot: Out-of-time sample.
        psi_bins: Quantile bins for the score PSI.

    Returns:
        Flat mapping with ``dev_``/``oot_`` prefixed AUC, Gini, KS, Brier,
        mean PD and default rate, plus ``gini_deterioration``,
        ``score_psi`` and ``mean_pd_drift`` (change in mean PD minus change in
        default rate between the samples).
    """
    out: dict[str, float] = {}
    for name, sample in (("dev", dev), ("oot", oot)):
        out[f"{name}_auc"] = auc(sample.y, sample.pd)
        out[f"{name}_gini"] = gini(sample.y, sample.pd)
        out[f"{name}_ks"] = ks_statistic(sample.y, sample.pd)
        out[f"{name}_brier"] = brier_score(sample.y, sample.pd)
        out[f"{name}_mean_pd"] = float(np.mean(sample.pd))
        out[f"{name}_default_rate"] = float(np.mean(sample.y))
    out["gini_deterioration"] = out["dev_gini"] - out["oot_gini"]
    out["score_psi"] = psi(dev.score, oot.score, bins=psi_bins)
    out["mean_pd_drift"] = (out["oot_mean_pd"] - out["dev_mean_pd"]) - (
        out["oot_default_rate"] - out["dev_default_rate"]
    )
    return out
