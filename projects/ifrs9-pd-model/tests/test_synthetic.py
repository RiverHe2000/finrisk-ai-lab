import numpy as np
import pandas as pd

from ifrs9_pd.config import DataConfig
from ifrs9_pd.data.schema import Col, validate_portfolio
from ifrs9_pd.data.synthetic import (
    generate_portfolio,
    macro_unemployment_series,
    origination_view,
    seasoning_effect,
    split_dev_oot,
)


def test_portfolio_shape_and_schema(portfolio, cfg):
    assert len(portfolio) == cfg.data.n_loans
    assert validate_portfolio(portfolio) is portfolio
    assert portfolio[Col.LOAN_ID.value].is_unique


def test_default_rate_and_true_pd(portfolio, cfg):
    rate = portfolio[Col.DEFAULT_12M.value].mean()
    assert 0.01 < rate < 0.06
    assert abs(portfolio[Col.TRUE_PD.value].mean() - cfg.data.target_default_rate) < 0.003
    assert portfolio[Col.TRUE_PD.value].between(0, 1).all()


def test_missing_values_only_in_expected_columns(portfolio, cfg):
    missing = portfolio.isna().mean()
    assert 0.0 < missing[Col.BUREAU_SCORE.value] < 3 * cfg.data.missing_rate
    assert 0.0 < missing[Col.DTI.value] < 3 * cfg.data.missing_rate
    assert missing.drop([Col.BUREAU_SCORE.value, Col.DTI.value]).eq(0).all()


def test_ranges(portfolio):
    assert portfolio[Col.BUREAU_SCORE.value].dropna().between(300, 900).all()
    assert portfolio[Col.LTV.value].between(0.3, 0.98).all()
    assert set(portfolio[Col.CURRENT_DPD.value].unique()) <= {0, 30, 60, 90}
    assert (portfolio[Col.SNAPSHOT_DATE.value] >= portfolio[Col.ORIGINATION_DATE.value]).all()
    assert (portfolio[Col.MONTHS_ON_BOOK.value] <= 60).all()


def test_deterministic():
    a = generate_portfolio(DataConfig(n_loans=500, seed=3))
    b = generate_portfolio(DataConfig(n_loans=500, seed=3))
    c = generate_portfolio(DataConfig(n_loans=500, seed=4))
    pd.testing.assert_frame_equal(a, b)
    assert not a[Col.BUREAU_SCORE.value].equals(c[Col.BUREAU_SCORE.value])


def test_macro_series_has_2020_stress():
    macro = macro_unemployment_series()
    assert macro.idxmax().year == 2020
    assert macro.loc["2019-01-01"] < macro.max() - 1.5


def test_split_dev_oot(portfolio, cfg):
    dev, oot = split_dev_oot(portfolio, cfg.data.dev_cutoff)
    assert len(dev) + len(oot) == len(portfolio)
    assert (dev[Col.SNAPSHOT_DATE.value] <= pd.Timestamp("2022-06-01")).all()
    assert (oot[Col.SNAPSHOT_DATE.value] > pd.Timestamp("2022-06-01")).all()


def test_origination_view_resets_delinquency_only(portfolio):
    view = origination_view(portfolio)
    assert (view[Col.ARREARS_12M.value] == 0).all()
    assert (view[Col.CURRENT_DPD.value] == 0).all()
    assert view[Col.MONTHS_ON_BOOK.value].equals(portfolio[Col.MONTHS_ON_BOOK.value])
    assert view[Col.UNEMPLOYMENT.value].equals(portfolio[Col.UNEMPLOYMENT_ORIG.value])
    assert (portfolio[Col.ARREARS_12M.value] > 0).any()


def test_seasoning_effect_peaks_at_peak_month():
    months = np.arange(0, 61, dtype=np.float64)
    effect = seasoning_effect(months, peak_month=24.0)
    assert effect[0] == 0.0
    assert int(np.argmax(effect)) == 24
