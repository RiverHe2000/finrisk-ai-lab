"""Expected credit loss from a marginal PD term structure.

For loan ``i`` with exposure profile ``EAD_t``, loss-given-default ``LGD``,
marginal PD ``PD^marg_t`` and monthly discount factor ``DF_t = (1 + r)^(-t/12)``::

    ECL_12m      = sum_{t=1}^{12} PD^marg_t * LGD * EAD_t * DF_t
    ECL_lifetime = sum_{t=1}^{H}  PD^marg_t * LGD * EAD_t * DF_t
    ECL_stage3   = EAD_1 * LGD                       (PD = 1)

Stage 1 loans carry ``ECL_12m``, Stage 2 loans ``ECL_lifetime`` and Stage 3
loans the credit-impaired amount.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def discount_factors(rate: float, horizon: int) -> NDArray[np.float64]:
    """Monthly discount factors ``(1 + r)^(-t/12)`` for ``t = 1..horizon``."""
    t = np.arange(1, horizon + 1, dtype=np.float64)
    return np.asarray((1.0 + rate) ** (-t / 12.0), dtype=np.float64)


def ead_profile(
    ead: NDArray[np.float64], horizon: int, remaining_term: NDArray[np.int64] | None = None
) -> NDArray[np.float64]:
    """Exposure at the start of each future month.

    With a remaining term ``T`` the balance amortises linearly:
    ``EAD_t = EAD * max(0, (T - t + 1) / T)``, so exposure is zero past
    maturity. Without a term the exposure is flat.

    Args:
        ead: Current exposure, shape ``(n,)``.
        horizon: Number of months.
        remaining_term: Remaining term in months per loan, or ``None``.

    Returns:
        Array of shape ``(n, horizon)``.
    """
    ead_arr = np.asarray(ead, dtype=np.float64)[:, None]
    if remaining_term is None:
        return np.repeat(ead_arr, horizon, axis=1)
    t = np.arange(1, horizon + 1, dtype=np.float64)[None, :]
    term = np.asarray(remaining_term, dtype=np.float64)[:, None]
    return np.asarray(ead_arr * np.clip((term - t + 1.0) / term, 0.0, 1.0), dtype=np.float64)


def compute_ecl(
    ead: NDArray[np.float64],
    lgd: NDArray[np.float64] | float,
    marginal_pd: NDArray[np.float64],
    discount_rate: float,
    stage: NDArray[np.int64],
    *,
    remaining_term: NDArray[np.int64] | None = None,
) -> pd.DataFrame:
    """Per-loan 12-month, lifetime and stage-applied ECL.

    Args:
        ead: Current exposure per loan, shape ``(n,)``.
        lgd: Loss given default, scalar or shape ``(n,)``.
        marginal_pd: Marginal PD curves, shape ``(n, H)`` with ``H >= 12``.
        discount_rate: Annual effective discount rate.
        stage: Stage per loan in ``{1, 2, 3}``.
        remaining_term: Optional remaining term for amortisation and lifetime cap.

    Returns:
        DataFrame with ``ead``, ``lgd``, ``stage``, ``ecl_12m``, ``ecl_lifetime``
        and ``ecl_applied`` columns.
    """
    pd_curve = np.asarray(marginal_pd, dtype=np.float64)
    if pd_curve.ndim != 2 or pd_curve.shape[1] < 12:
        msg = "marginal_pd must have shape (n, H) with H >= 12"
        raise ValueError(msg)
    n, horizon = pd_curve.shape
    lgd_arr = np.broadcast_to(np.asarray(lgd, dtype=np.float64), (n,))
    exposure = ead_profile(np.asarray(ead, dtype=np.float64), horizon, remaining_term)
    loss = (
        pd_curve * exposure * lgd_arr[:, None] * discount_factors(discount_rate, horizon)[None, :]
    )
    ecl_12m = loss[:, :12].sum(axis=1)
    ecl_lifetime = loss.sum(axis=1)
    stage_arr = np.asarray(stage, dtype=np.int64)
    ecl_stage3 = np.asarray(ead, dtype=np.float64) * lgd_arr
    applied = np.select(
        [stage_arr == 1, stage_arr == 2], [ecl_12m, ecl_lifetime], default=ecl_stage3
    )
    return pd.DataFrame(
        {
            "ead": np.asarray(ead, dtype=np.float64),
            "lgd": lgd_arr,
            "stage": stage_arr,
            "ecl_12m": ecl_12m,
            "ecl_lifetime": ecl_lifetime,
            "ecl_applied": applied,
        }
    )


def portfolio_summary(ecl: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ECL, EAD and coverage by stage with a total row.

    Args:
        ecl: Output of :func:`compute_ecl`.

    Returns:
        DataFrame with ``stage``, ``n_loans``, ``ead``, ``ecl``, ``coverage``
        (``ecl / ead``) and ``share_of_ead`` columns.
    """
    grouped = ecl.groupby("stage", sort=True).agg(
        n_loans=("ead", "size"), ead=("ead", "sum"), ecl=("ecl_applied", "sum")
    )
    summary = grouped.reset_index()
    summary["stage"] = summary["stage"].astype("str")
    total = pd.DataFrame(
        {
            "stage": ["total"],
            "n_loans": [int(summary["n_loans"].sum())],
            "ead": [float(summary["ead"].sum())],
            "ecl": [float(summary["ecl"].sum())],
        }
    )
    out = pd.concat([summary, total], ignore_index=True)
    out["coverage"] = out["ecl"] / out["ead"]
    out["share_of_ead"] = out["ead"] / float(total["ead"].iloc[0])
    return out
