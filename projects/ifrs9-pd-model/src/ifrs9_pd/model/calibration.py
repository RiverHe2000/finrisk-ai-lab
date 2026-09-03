"""Central-tendency calibration and Vasicek PIT/TTC conversion.

Intercept calibration finds the logit shift ``delta`` such that::

    mean_i sigmoid(logit(p_i) + delta) = target

which anchors the model's average PD to a long-run (through-the-cycle) default
rate without changing rank ordering.

The Vasicek one-factor model links a TTC PD to a point-in-time PD given the
systematic factor ``z`` and asset correlation ``rho``::

    PD_pit = Phi( (Phi^-1(PD_ttc) - sqrt(rho) z) / sqrt(1 - rho) )
    PD_ttc = Phi( sqrt(1 - rho) Phi^-1(PD_pit) + sqrt(rho) z )

A negative ``z`` (economic stress) raises the PIT PD. Here ``z`` is proxied by
the negative standardised unemployment gap (or, optionally, the standardised
change in unemployment), multiplied by a factor loading ``lambda`` that is
estimated by maximum likelihood on the development sample because the
amplitude of a standardised macro index need not match that of the latent
Vasicek factor.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import norm

_EPS = 1e-9


def shift_logit(pd_values: NDArray[np.float64], delta: float) -> NDArray[np.float64]:
    """Apply an additive shift on the logit scale.

    Args:
        pd_values: Probabilities in ``(0, 1)``.
        delta: Shift added to ``logit(p)``.

    Returns:
        Shifted probabilities.
    """
    p = np.clip(np.asarray(pd_values, dtype=np.float64), _EPS, 1 - _EPS)
    return np.asarray(expit(logit(p) + delta), dtype=np.float64)


def calibrate_intercept(
    pd_values: NDArray[np.float64],
    target_rate: float,
    overlay: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> float:
    """Solve for the logit shift that makes the mean PD equal ``target_rate``.

    Args:
        pd_values: Uncalibrated PDs.
        target_rate: Long-run average default rate to anchor on.
        overlay: Optional transform applied to the shifted PDs before
            averaging, e.g. a PIT conversion, so that the anchored quantity is
            the PD actually compared with observed defaults.

    Returns:
        The shift ``delta`` to pass to :func:`shift_logit`.
    """
    if not 0.0 < target_rate < 1.0:
        msg = "target_rate must lie strictly between 0 and 1"
        raise ValueError(msg)

    def gap(delta: float) -> float:
        shifted = shift_logit(pd_values, delta)
        return float(np.mean(overlay(shifted) if overlay is not None else shifted)) - target_rate

    return float(brentq(gap, -30.0, 30.0, xtol=1e-12))


def ttc_to_pit(
    pd_ttc: NDArray[np.float64], z: NDArray[np.float64] | float, rho: float = 0.15
) -> NDArray[np.float64]:
    """Convert TTC PD to PIT PD under the Vasicek model (see module docstring).

    Args:
        pd_ttc: Through-the-cycle PDs.
        z: Systematic factor (scalar or per-row).
        rho: Asset correlation.

    Returns:
        Point-in-time PDs.
    """
    p = np.clip(np.asarray(pd_ttc, dtype=np.float64), _EPS, 1 - _EPS)
    x = (norm.ppf(p) - np.sqrt(rho) * np.asarray(z, dtype=np.float64)) / np.sqrt(1.0 - rho)
    return np.asarray(norm.cdf(x), dtype=np.float64)


def pit_to_ttc(
    pd_pit: NDArray[np.float64], z: NDArray[np.float64] | float, rho: float = 0.15
) -> NDArray[np.float64]:
    """Inverse of :func:`ttc_to_pit`.

    Args:
        pd_pit: Point-in-time PDs.
        z: Systematic factor (scalar or per-row).
        rho: Asset correlation.

    Returns:
        Through-the-cycle PDs.
    """
    p = np.clip(np.asarray(pd_pit, dtype=np.float64), _EPS, 1 - _EPS)
    x = np.sqrt(1.0 - rho) * norm.ppf(p) + np.sqrt(rho) * np.asarray(z, dtype=np.float64)
    return np.asarray(norm.cdf(x), dtype=np.float64)


def macro_z_score(unemployment: pd.Series, window: int | None = None) -> pd.Series:
    """Standardised systematic-factor proxy from an unemployment series.

    With ``window=None`` the proxy is the unemployment gap
    ``z_t = -(u_t - mean(u)) / std(u)``; with an integer window it is the
    standardised ``window``-month change. The sign flip makes rising
    unemployment a negative (adverse) factor.

    Args:
        unemployment: Monthly unemployment rate indexed by month.
        window: Change horizon in months, or ``None`` for the level gap.

    Returns:
        Series of ``z`` values aligned to ``unemployment``.
    """
    base = unemployment.diff(window) if window else unemployment
    valid = base.dropna()
    scale = float(valid.std()) if len(valid) > 1 else 1.0
    z = -(base - float(valid.mean())) / (scale if scale > 0 else 1.0)
    return z.fillna(0.0).rename("z")


def fit_factor_loading(
    pd_values: NDArray[np.float64],
    y: NDArray[np.int64],
    z: NDArray[np.float64],
    rho: float = 0.15,
    max_loading: float = 3.0,
) -> tuple[float, float]:
    """Estimate the factor loading ``lambda`` and calibration shift jointly.

    For each candidate ``lambda`` the intercept shift is re-solved so the
    mean PIT PD equals the observed default rate, and the Bernoulli
    log-likelihood of ``PIT(shift(p), lambda z)`` is evaluated; the loading
    maximising the likelihood is returned.

    Args:
        pd_values: Uncalibrated TTC PDs.
        y: Binary outcomes.
        z: Standardised factor proxy per row.
        rho: Asset correlation.
        max_loading: Upper bound of the search interval.

    Returns:
        ``(loading, shift)``.
    """
    target = float(np.mean(y))
    y_arr = np.asarray(y, dtype=np.float64)

    def shift_for(loading: float) -> float:
        return calibrate_intercept(
            pd_values, target, overlay=lambda p: ttc_to_pit(p, loading * z, rho)
        )

    def negative_log_likelihood(loading: float) -> float:
        p = np.clip(
            ttc_to_pit(shift_logit(pd_values, shift_for(loading)), loading * z, rho), _EPS, 1 - _EPS
        )
        return float(-np.sum(y_arr * np.log(p) + (1.0 - y_arr) * np.log1p(-p)))

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(0.0, max_loading),
        method="bounded",
        options={"xatol": 1e-4},
    )
    loading = float(result.x)
    return loading, shift_for(loading)
