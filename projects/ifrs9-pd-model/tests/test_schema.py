import numpy as np
import pytest

from ifrs9_pd.data.schema import Col, validate_portfolio


def test_valid_portfolio_passes(portfolio):
    assert validate_portfolio(portfolio) is portfolio


def test_missing_column(portfolio):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_portfolio(portfolio.drop(columns=[Col.LTV.value]))


def test_out_of_range(portfolio):
    bad = portfolio.copy()
    bad.loc[0, Col.BUREAU_SCORE.value] = 1200.0
    with pytest.raises(ValueError, match="bureau_score"):
        validate_portfolio(bad)


def test_missing_not_allowed(portfolio):
    bad = portfolio.copy()
    bad.loc[0, Col.LTV.value] = np.nan
    with pytest.raises(ValueError, match="must not contain missing"):
        validate_portfolio(bad)


def test_non_numeric(portfolio):
    bad = portfolio.copy()
    bad[Col.EAD.value] = bad[Col.EAD.value].astype("str")
    with pytest.raises(ValueError, match="must be numeric"):
        validate_portfolio(bad)


def test_bad_target(portfolio):
    bad = portfolio.copy()
    bad.loc[0, Col.DEFAULT_12M.value] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_portfolio(bad)


def test_dates(portfolio):
    bad = portfolio.copy()
    bad[Col.SNAPSHOT_DATE.value] = bad[Col.SNAPSHOT_DATE.value].astype("str")
    with pytest.raises(ValueError, match="datetime64"):
        validate_portfolio(bad)
    bad = portfolio.copy()
    bad.loc[0, Col.SNAPSHOT_DATE.value] = bad.loc[0, Col.ORIGINATION_DATE.value] - np.timedelta64(
        31, "D"
    )
    with pytest.raises(ValueError, match="precede"):
        validate_portfolio(bad)


def test_duplicate_ids_and_empty(portfolio):
    bad = portfolio.copy()
    bad.loc[1, Col.LOAN_ID.value] = bad.loc[0, Col.LOAN_ID.value]
    with pytest.raises(ValueError, match="unique"):
        validate_portfolio(bad)
    with pytest.raises(ValueError, match="empty"):
        validate_portfolio(portfolio.iloc[0:0])
