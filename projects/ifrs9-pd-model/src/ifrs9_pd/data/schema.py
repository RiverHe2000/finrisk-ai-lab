"""Column names and schema validation for the loan-snapshot portfolio table."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd


class Col(StrEnum):
    """Column names of the portfolio DataFrame."""

    LOAN_ID = "loan_id"
    ORIGINATION_DATE = "origination_date"
    SNAPSHOT_DATE = "snapshot_date"
    BUREAU_SCORE = "bureau_score"
    LTV = "loan_to_value"
    DTI = "debt_to_income"
    UTILISATION = "utilisation"
    MONTHS_ON_BOOK = "months_on_book"
    EMPLOYMENT_TYPE = "employment_type"
    REGION = "region"
    INTEREST_ONLY = "interest_only"
    ARREARS_12M = "arrears_history_12m"
    CURRENT_DPD = "current_dpd"
    UNEMPLOYMENT = "macro_unemployment_rate"
    UNEMPLOYMENT_ORIG = "macro_unemployment_at_origination"
    EAD = "exposure_at_default"
    REMAINING_TERM = "remaining_term_months"
    TRUE_PD = "true_pd"
    DEFAULT_12M = "default_12m"


NUMERIC_FEATURES: tuple[Col, ...] = (
    Col.BUREAU_SCORE,
    Col.LTV,
    Col.DTI,
    Col.UTILISATION,
    Col.MONTHS_ON_BOOK,
    Col.ARREARS_12M,
    Col.CURRENT_DPD,
)
"""Numeric candidate features for the scorecard.

The macro rate is deliberately *not* a scorecard input: the scorecard ranks
borrowers through the cycle and the macro effect enters through the Vasicek
PIT overlay.
"""

CATEGORICAL_FEATURES: tuple[Col, ...] = (Col.EMPLOYMENT_TYPE, Col.REGION, Col.INTEREST_ONLY)
"""Categorical candidate features for the scorecard."""

REQUIRED_COLUMNS: tuple[Col, ...] = (
    Col.LOAN_ID,
    Col.ORIGINATION_DATE,
    Col.SNAPSHOT_DATE,
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
    Col.UNEMPLOYMENT,
    Col.UNEMPLOYMENT_ORIG,
    Col.EAD,
    Col.REMAINING_TERM,
    Col.DEFAULT_12M,
)
"""Columns that must be present for the pipeline to run (``true_pd`` is optional)."""

_RANGES: dict[Col, tuple[float, float]] = {
    Col.BUREAU_SCORE: (300.0, 900.0),
    Col.LTV: (0.0, 1.5),
    Col.DTI: (0.0, 20.0),
    Col.UTILISATION: (0.0, 1.0),
    Col.MONTHS_ON_BOOK: (0.0, 600.0),
    Col.ARREARS_12M: (0.0, 12.0),
    Col.CURRENT_DPD: (0.0, 10_000.0),
    Col.UNEMPLOYMENT: (0.0, 100.0),
    Col.UNEMPLOYMENT_ORIG: (0.0, 100.0),
    Col.EAD: (0.0, np.inf),
    Col.REMAINING_TERM: (1.0, 600.0),
}

_ALLOW_MISSING: frozenset[Col] = frozenset({Col.BUREAU_SCORE, Col.DTI})


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c.value for c in REQUIRED_COLUMNS if c.value not in df.columns]
    if missing:
        msg = f"portfolio is missing required columns: {missing}"
        raise ValueError(msg)


def _check_numeric_ranges(df: pd.DataFrame) -> None:
    for col, (lo, hi) in _RANGES.items():
        series = df[col.value]
        if not pd.api.types.is_numeric_dtype(series):
            msg = f"column '{col.value}' must be numeric, got dtype {series.dtype}"
            raise ValueError(msg)
        if col not in _ALLOW_MISSING and series.isna().any():
            msg = f"column '{col.value}' must not contain missing values"
            raise ValueError(msg)
        values = series.dropna().to_numpy(dtype=np.float64)
        if values.size and (values.min() < lo or values.max() > hi):
            msg = (
                f"column '{col.value}' has values outside [{lo}, {hi}]: "
                f"min={values.min():.4g}, max={values.max():.4g}"
            )
            raise ValueError(msg)


def _check_target_and_dates(df: pd.DataFrame) -> None:
    target = df[Col.DEFAULT_12M.value]
    if target.isna().any() or not set(np.unique(target.to_numpy())) <= {0, 1}:
        msg = f"column '{Col.DEFAULT_12M.value}' must be binary 0/1 without missing values"
        raise ValueError(msg)
    for col in (Col.ORIGINATION_DATE, Col.SNAPSHOT_DATE):
        if not pd.api.types.is_datetime64_any_dtype(df[col.value]):
            msg = f"column '{col.value}' must be a datetime64 column"
            raise ValueError(msg)
    if (df[Col.SNAPSHOT_DATE.value] < df[Col.ORIGINATION_DATE.value]).any():
        msg = "snapshot_date must not precede origination_date"
        raise ValueError(msg)
    if df[Col.LOAN_ID.value].duplicated().any():
        msg = "loan_id must be unique"
        raise ValueError(msg)


def validate_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the portfolio table and return it unchanged.

    Checks presence of :data:`REQUIRED_COLUMNS`, numeric dtypes and plausible
    ranges, absence of missing values outside ``bureau_score`` / ``debt_to_income``,
    a binary target, datetime dates and unique loan identifiers.

    Args:
        df: Portfolio table to validate.

    Returns:
        The same DataFrame, so the call can be chained.

    Raises:
        ValueError: If any check fails; the message names the offending column.
    """
    if df.empty:
        msg = "portfolio is empty"
        raise ValueError(msg)
    _check_columns(df)
    _check_numeric_ranges(df)
    _check_target_and_dates(df)
    return df
