"""Seeded synthetic retail-mortgage portfolio with a known latent PD.

The generator draws borrower and loan characteristics, a smooth macro
unemployment path with a 2020 stress bump, and a *true* 12-month PD. The
borrower-level (through-the-cycle) PD is a logistic function of the
characteristics plus a seasoning hump; the point-in-time PD then follows the
Vasicek one-factor model with asset correlation ``TRUE_RHO`` and a systematic
factor ``z_t = -(u_t - 5) / TRUE_FACTOR_SCALE`` driven by the unemployment
gap. Default labels are Bernoulli draws from the true PIT PD, so the
discriminatory power a model can attain is bounded and known, and the model's
Vasicek overlay has a well-defined target to recover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import expit

from ifrs9_pd.config import DataConfig
from ifrs9_pd.data.schema import Col
from ifrs9_pd.model.calibration import ttc_to_pit

EMPLOYMENT_TYPES: tuple[str, ...] = ("full_time", "part_time", "self_employed", "casual", "retired")
REGIONS: tuple[str, ...] = ("NSW", "VIC", "QLD", "WA", "SA", "other")
DPD_BUCKETS: tuple[int, ...] = (0, 30, 60, 90)
ORIGINAL_TERM_MONTHS = 360
TRUE_RHO = 0.15
TRUE_FACTOR_SCALE = 2.0
NEUTRAL_UNEMPLOYMENT = 5.0

_EMPLOYMENT_PROBS = (0.60, 0.15, 0.12, 0.08, 0.05)
_EMPLOYMENT_EFFECT = {
    "full_time": 0.0,
    "part_time": 0.2,
    "self_employed": 0.4,
    "casual": 0.6,
    "retired": 0.1,
}
_REGION_PROBS = (0.32, 0.26, 0.20, 0.10, 0.07, 0.05)
_REGION_EFFECT = {"NSW": 0.0, "VIC": 0.0, "QLD": 0.1, "WA": 0.25, "SA": 0.05, "other": 0.15}
_DPD_EFFECT = {0: 0.0, 30: 1.0, 60: 1.8, 90: 3.0}


def month_index(start: str, end: str) -> pd.DatetimeIndex:
    """Return month-start timestamps from ``start`` to ``end`` inclusive.

    Args:
        start: First month as ``YYYY-MM``.
        end: Last month as ``YYYY-MM``.

    Returns:
        A monthly ``DatetimeIndex`` of month starts.
    """
    return pd.date_range(start=f"{start}-01", end=f"{end}-01", freq="MS")


def macro_unemployment_series(start: str = "2016-01", end: str = "2025-12") -> pd.Series:
    """Smooth monthly unemployment path (%) with a stress bump peaking in mid-2020.

    The path is ``5.0 + 0.25 sin(2 pi t / 48) - 0.006 t + bump(t)`` where the
    bump is a Gaussian of amplitude 2.6 centred on 2020-06 with a faster rise
    (sigma 3 months) than recovery (sigma 9 months).

    Args:
        start: First month as ``YYYY-MM``.
        end: Last month as ``YYYY-MM``.

    Returns:
        Series indexed by month start with unemployment in percent.
    """
    idx = month_index(start, end)
    t = np.arange(len(idx), dtype=np.float64)
    peak = float(idx.get_indexer(pd.DatetimeIndex(["2020-06-01"]))[0])
    sigma = np.where(t < peak, 3.0, 9.0)
    bump = 2.6 * np.exp(-0.5 * ((t - peak) / sigma) ** 2)
    level = 5.0 + 0.25 * np.sin(2.0 * np.pi * t / 48.0) - 0.006 * t + bump
    return pd.Series(np.round(level, 3), index=idx, name=Col.UNEMPLOYMENT.value)


def seasoning_effect(
    months_on_book: NDArray[np.float64], peak_month: float = 24.0
) -> NDArray[np.float64]:
    """Hump-shaped ageing effect on the log-odds of default.

    ``0.4 * (m / p) * exp(1 - m / p)`` rises from zero at origination to 0.4 at
    ``p`` months and decays thereafter, mimicking the well-known mortgage
    default hump.

    Args:
        months_on_book: Loan age in months.
        peak_month: Age at which the effect peaks.

    Returns:
        Additive log-odds effect, same shape as the input.
    """
    ratio = months_on_book / peak_month
    return np.asarray(0.4 * ratio * np.exp(1.0 - ratio), dtype=np.float64)


def _draw_loan_characteristics(rng: np.random.Generator, n: int) -> dict[str, NDArray[np.float64]]:
    bureau = np.clip(np.round(rng.normal(690.0, 80.0, n)), 300.0, 900.0)
    ltv = 0.3 + 0.68 * rng.beta(4.0, 2.5, n)
    dti = np.clip(rng.lognormal(np.log(4.5), 0.35, n), 0.5, 15.0)
    utilisation = rng.beta(2.0, 3.0, n)
    return {
        Col.BUREAU_SCORE.value: bureau,
        Col.LTV.value: np.round(ltv, 4),
        Col.DTI.value: np.round(dti, 3),
        Col.UTILISATION.value: np.round(utilisation, 4),
    }


def _draw_behaviour(
    rng: np.random.Generator, bureau: NDArray[np.float64], months_on_book: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    lam = np.exp(-2.6 + 0.012 * (680.0 - bureau))
    arrears = np.minimum(rng.poisson(lam), 12).astype(np.int64)
    arrears[months_on_book == 0] = 0
    p_delinquent = 0.015 + 0.07 * np.minimum(arrears, 4)
    delinquent = rng.random(len(bureau)) < p_delinquent
    severity = rng.choice(np.arange(1, 4), size=len(bureau), p=(0.62, 0.25, 0.13))
    dpd_code = np.where(delinquent, severity, 0)
    dpd_code[months_on_book == 0] = 0
    current_dpd = np.asarray(DPD_BUCKETS, dtype=np.int64)[dpd_code]
    return arrears, current_dpd


def _true_logit(df: pd.DataFrame) -> NDArray[np.float64]:
    score = df[Col.BUREAU_SCORE.value].to_numpy(dtype=np.float64)
    dpd = df[Col.CURRENT_DPD.value].map(_DPD_EFFECT).to_numpy(dtype=np.float64)
    employment = df[Col.EMPLOYMENT_TYPE.value].map(_EMPLOYMENT_EFFECT).to_numpy(dtype=np.float64)
    region = df[Col.REGION.value].map(_REGION_EFFECT).to_numpy(dtype=np.float64)
    logit = (
        -0.014 * (score - 680.0)
        + 2.5 * (df[Col.LTV.value].to_numpy(dtype=np.float64) - 0.7)
        + 0.12 * (df[Col.DTI.value].to_numpy(dtype=np.float64) - 4.5)
        + 0.8 * (df[Col.UTILISATION.value].to_numpy(dtype=np.float64) - 0.4)
        + 0.40 * np.minimum(df[Col.ARREARS_12M.value].to_numpy(dtype=np.float64), 6.0)
        + dpd
        + 0.3 * df[Col.INTEREST_ONLY.value].to_numpy(dtype=np.float64)
        + employment
        + region
        + seasoning_effect(df[Col.MONTHS_ON_BOOK.value].to_numpy(dtype=np.float64))
    )
    return np.asarray(logit, dtype=np.float64)


def true_factor(unemployment: NDArray[np.float64]) -> NDArray[np.float64]:
    """Systematic factor of the data-generating process: ``-(u - 5) / 2``."""
    return np.asarray(-(unemployment - NEUTRAL_UNEMPLOYMENT) / TRUE_FACTOR_SCALE, dtype=np.float64)


def _true_pit_pd(
    logit: NDArray[np.float64], z: NDArray[np.float64], target_rate: float
) -> NDArray[np.float64]:
    """PIT PD with the intercept chosen so the mean PD equals ``target_rate``."""

    def gap(a: float) -> float:
        return float(np.mean(ttc_to_pit(expit(logit + a), z, TRUE_RHO))) - target_rate

    intercept = float(brentq(gap, -25.0, 25.0, xtol=1e-10))
    return ttc_to_pit(np.asarray(expit(logit + intercept), dtype=np.float64), z, TRUE_RHO)


def generate_portfolio(cfg: DataConfig | None = None) -> pd.DataFrame:
    """Generate a seeded synthetic loan-snapshot table.

    Each row is one loan observed at one snapshot month; ``default_12m`` is one
    if the loan defaults within the following twelve months. The true PD used
    to draw the label is kept in ``true_pd`` for diagnostics. Origination
    months are drawn with linearly decreasing weights so older vintages (and
    hence the development window) carry more observations.

    Args:
        cfg: Generator configuration; defaults to :class:`DataConfig`.

    Returns:
        Portfolio DataFrame conforming to :mod:`ifrs9_pd.data.schema`.
    """
    cfg = cfg or DataConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_loans
    orig_months = month_index(cfg.origination_start, cfg.origination_end)
    last_snapshot = pd.Timestamp(f"{cfg.last_snapshot}-01")
    macro = macro_unemployment_series()

    calendar = macro.index
    vintage_weights = np.linspace(2.0, 0.6, len(orig_months))
    vintage = rng.choice(len(orig_months), n, p=vintage_weights / vintage_weights.sum())
    orig_pos = calendar.get_indexer(orig_months[vintage])
    last_pos = int(calendar.get_indexer(pd.DatetimeIndex([last_snapshot]))[0])
    max_mob = np.minimum(last_pos - orig_pos, 60)
    months_on_book = (rng.random(n) * (max_mob + 1)).astype(np.int64)
    origination = calendar[orig_pos]
    snapshot = calendar[orig_pos + months_on_book]

    chars = _draw_loan_characteristics(rng, n)
    arrears, current_dpd = _draw_behaviour(rng, chars[Col.BUREAU_SCORE.value], months_on_book)
    interest_only = (rng.random(n) < 0.2).astype(np.int64)
    remaining_term = np.maximum(ORIGINAL_TERM_MONTHS - months_on_book, 12)
    balance = rng.lognormal(np.log(450_000.0), 0.45, n)
    amortised = np.where(interest_only == 1, 1.0, remaining_term / ORIGINAL_TERM_MONTHS)

    df = pd.DataFrame(
        {
            Col.LOAN_ID.value: [f"L{i:07d}" for i in range(n)],
            Col.ORIGINATION_DATE.value: origination,
            Col.SNAPSHOT_DATE.value: snapshot,
            **chars,
            Col.MONTHS_ON_BOOK.value: months_on_book,
            Col.EMPLOYMENT_TYPE.value: rng.choice(EMPLOYMENT_TYPES, n, p=_EMPLOYMENT_PROBS),
            Col.REGION.value: rng.choice(REGIONS, n, p=_REGION_PROBS),
            Col.INTEREST_ONLY.value: interest_only,
            Col.ARREARS_12M.value: arrears,
            Col.CURRENT_DPD.value: current_dpd,
            Col.UNEMPLOYMENT.value: macro.reindex(snapshot).to_numpy(),
            Col.UNEMPLOYMENT_ORIG.value: macro.reindex(origination).to_numpy(),
            Col.EAD.value: np.round(balance * amortised, 2),
            Col.REMAINING_TERM.value: remaining_term,
        }
    )
    z = true_factor(df[Col.UNEMPLOYMENT.value].to_numpy(dtype=np.float64))
    true_pd = _true_pit_pd(_true_logit(df), z, cfg.target_default_rate)
    df[Col.TRUE_PD.value] = true_pd
    df[Col.DEFAULT_12M.value] = (rng.random(n) < true_pd).astype(np.int64)

    for col in (Col.BUREAU_SCORE.value, Col.DTI.value):
        mask = rng.random(n) < cfg.missing_rate
        df[col] = df[col].where(~mask, np.nan)
    return df


def split_dev_oot(df: pd.DataFrame, dev_cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the portfolio into development and out-of-time samples.

    Args:
        df: Portfolio table.
        dev_cutoff: Last development snapshot month as ``YYYY-MM`` (inclusive).

    Returns:
        ``(development, out_of_time)`` DataFrames with fresh integer indexes.
    """
    cutoff = pd.Timestamp(f"{dev_cutoff}-01")
    is_dev = df[Col.SNAPSHOT_DATE.value] <= cutoff
    dev = df.loc[is_dev].reset_index(drop=True)
    oot = df.loc[~is_dev].reset_index(drop=True)
    return dev, oot


def origination_view(df: pd.DataFrame) -> pd.DataFrame:
    """Age-matched origination view of each loan for the SICR comparison.

    Delinquency fields are reset to their origination values (no arrears, no
    days past due) and the macro rate is replaced by the unemployment rate
    observed at origination, while ``months_on_book`` is kept as observed.
    Comparing the current PD with this *age-matched* origination PD means
    that the expected seasoning pattern does not by itself trigger Stage 2;
    only deterioration in behaviour or in the economy does. Static
    characteristics are kept as observed at the snapshot, which is a
    simplification: a production system stores the PD assigned at origination.

    Args:
        df: Portfolio table.

    Returns:
        A copy with delinquency and macro columns reset.
    """
    out = df.copy()
    out[Col.ARREARS_12M.value] = 0
    out[Col.CURRENT_DPD.value] = 0
    out[Col.UNEMPLOYMENT.value] = df[Col.UNEMPLOYMENT_ORIG.value]
    return out
