"""Lifetime PD term structure with seasoning and scenario overlays.

For scenario ``s`` the systematic factor starts at ``z_0 + shift_s`` and
mean-reverts to zero at speed ``kappa``. The 12-month PD prevailing at future
month ``t`` is the Vasicek-conditional PD at ``z_t`` and is converted to a
constant monthly hazard, scaled by a seasoning multiplier ``m(a + t)`` that
is hump-shaped in loan age ``a + t`` (peaking at ``peak_month``) and
normalised per loan to average one over its next twelve months, so the
scorecard's 12-month PD is preserved::

    z_t        = (z_0 + shift_s) (1 - kappa)^(t - 1)
    PD12_t^s   = Phi( (Phi^-1(PD12_ttc) - sqrt(rho) z_t) / sqrt(1 - rho) )
    h_t^s      = m(a + t) * (1 - (1 - PD12_t^s)^(1/12))

Applying the transform at the annual horizon (where ``rho`` is calibrated)
rather than to monthly hazards keeps the factor sensitivity consistent with
the 12-month PIT/TTC conversion. Survival, marginal and cumulative PD follow
from the hazards::

    S_t = prod_{k<=t} (1 - h_k),   PD^marg_t = S_{t-1} h_t,   PD^cum_t = 1 - S_t

The probability-weighted curve is the weighted average of the scenario curves
(the weighting is linear, so ``survival = 1 - cumulative`` still holds).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ifrs9_pd.config import ScenarioConfig
from ifrs9_pd.model.calibration import ttc_to_pit

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class ScenarioCurves:
    """Marginal and cumulative PD curves of one scenario, shape ``(n, H)``."""

    marginal: FloatMatrix
    cumulative: FloatMatrix


@dataclass(frozen=True)
class TermStructure:
    """Probability-weighted lifetime PD curves plus the per-scenario curves.

    Attributes:
        marginal: Weighted marginal PD, shape ``(n, H)``.
        cumulative: Weighted cumulative PD, shape ``(n, H)``.
        survival: ``1 - cumulative``.
        by_scenario: Curves of each scenario.
    """

    marginal: FloatMatrix
    cumulative: FloatMatrix
    survival: FloatMatrix
    by_scenario: dict[str, ScenarioCurves] = field(default_factory=dict)


def monthly_hazard(pd_12m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Constant monthly hazard equivalent to a 12-month PD."""
    p = np.clip(np.asarray(pd_12m, dtype=np.float64), 0.0, 1.0 - 1e-12)
    return np.asarray(1.0 - (1.0 - p) ** (1.0 / 12.0), dtype=np.float64)


def seasoning_curve(
    horizon: int,
    peak_month: int = 24,
    peak_multiplier: float = 1.4,
    age: NDArray[np.int64] | None = None,
) -> FloatMatrix:
    """Hump-shaped hazard multiplier over the next ``horizon`` months.

    The raw curve in loan age ``u`` is ``1 + (peak_multiplier - 1) (u / p)
    exp(1 - u / p)``. For a loan of current age ``a`` the multiplier for
    projection month ``t`` is the raw curve at ``a + t`` divided by its mean
    over ``t = 1..12``, so the first-year PD is unchanged.

    Args:
        horizon: Number of projection months.
        peak_month: Age ``p`` at which the raw multiplier peaks.
        peak_multiplier: Raw multiplier at the peak.
        age: Current age in months per loan; ``None`` means new loans.

    Returns:
        Array of shape ``(n, horizon)`` (``(1, horizon)`` when ``age`` is ``None``).
    """
    t = np.arange(1, horizon + 1, dtype=np.float64)[None, :]
    age_arr = np.zeros(1) if age is None else np.asarray(age, dtype=np.float64)
    u = age_arr[:, None] + t
    raw = 1.0 + (peak_multiplier - 1.0) * (u / peak_month) * np.exp(1.0 - u / peak_month)
    return np.asarray(raw / raw[:, :12].mean(axis=1, keepdims=True), dtype=np.float64)


def z_path(z0: NDArray[np.float64] | float, horizon: int, mean_reversion: float) -> FloatMatrix:
    """Mean-reverting systematic factor path ``z_t = z0 (1 - kappa)^(t - 1)``.

    Args:
        z0: Initial factor, scalar or shape ``(n,)``.
        horizon: Number of months.
        mean_reversion: ``kappa`` in ``[0, 1]``.

    Returns:
        Array of shape ``(n, horizon)`` (``(1, horizon)`` for a scalar ``z0``).
    """
    decay = (1.0 - mean_reversion) ** np.arange(horizon, dtype=np.float64)
    return np.atleast_1d(np.asarray(z0, dtype=np.float64))[:, None] * decay[None, :]


def _curves_from_hazard(hazard: FloatMatrix) -> ScenarioCurves:
    survival = np.cumprod(1.0 - hazard, axis=1)
    prev = np.concatenate([np.ones((hazard.shape[0], 1)), survival[:, :-1]], axis=1)
    return ScenarioCurves(
        marginal=np.asarray(prev * hazard, dtype=np.float64),
        cumulative=np.asarray(1.0 - survival, dtype=np.float64),
    )


def build_term_structure(
    pd_12m: NDArray[np.float64],
    scenarios: ScenarioConfig,
    *,
    z0: NDArray[np.float64] | float = 0.0,
    rho: float = 0.15,
    horizon: int = 60,
    peak_month: int = 24,
    peak_multiplier: float = 1.4,
    mean_reversion: float = 0.05,
    age: NDArray[np.int64] | None = None,
) -> TermStructure:
    """Build scenario-weighted lifetime PD curves (formulas in the module docstring).

    Args:
        pd_12m: 12-month TTC PD per loan, shape ``(n,)``.
        scenarios: Scenario weights and factor shifts.
        z0: Current systematic factor (scalar or per loan).
        rho: Vasicek asset correlation.
        horizon: Lifetime horizon ``H`` in months.
        peak_month: Seasoning peak age in months.
        peak_multiplier: Seasoning peak multiplier.
        mean_reversion: Monthly mean reversion of the factor.
        age: Current loan age in months per loan (``None`` = new loans).

    Returns:
        A :class:`TermStructure` with arrays of shape ``(n, H)``.
    """
    pd_ttc = np.asarray(pd_12m, dtype=np.float64)[:, None]
    seasoning = seasoning_curve(horizon, peak_month, peak_multiplier, age)
    by_scenario: dict[str, ScenarioCurves] = {}
    for name, shift in scenarios.z_shifts.items():
        z = z_path(np.asarray(z0, dtype=np.float64) + shift, horizon, mean_reversion)
        hazard = np.clip(monthly_hazard(ttc_to_pit(pd_ttc, z, rho)) * seasoning, 0.0, 1.0)
        by_scenario[name] = _curves_from_hazard(hazard)
    marginal = sum(scenarios.weights[n] * c.marginal for n, c in by_scenario.items())
    cumulative = sum(scenarios.weights[n] * c.cumulative for n, c in by_scenario.items())
    marginal_arr = np.asarray(marginal, dtype=np.float64)
    cumulative_arr = np.asarray(cumulative, dtype=np.float64)
    return TermStructure(
        marginal=marginal_arr,
        cumulative=cumulative_arr,
        survival=1.0 - cumulative_arr,
        by_scenario=by_scenario,
    )
