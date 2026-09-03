import numpy as np
import pandas as pd
import pytest

from ifrs9_pd.data.synthetic import macro_unemployment_series
from ifrs9_pd.model.calibration import (
    calibrate_intercept,
    fit_factor_loading,
    macro_z_score,
    pit_to_ttc,
    shift_logit,
    ttc_to_pit,
)


def test_calibrate_intercept_hits_target():
    rng = np.random.default_rng(0)
    p = rng.beta(1, 30, 5000)
    delta = calibrate_intercept(p, 0.05)
    assert abs(shift_logit(p, delta).mean() - 0.05) < 1e-4
    assert delta > 0


def test_calibrate_intercept_with_overlay():
    rng = np.random.default_rng(1)
    p = rng.beta(1, 30, 5000)
    z = rng.normal(size=5000)
    delta = calibrate_intercept(p, 0.03, overlay=lambda q: ttc_to_pit(q, z, 0.15))
    assert abs(ttc_to_pit(shift_logit(p, delta), z, 0.15).mean() - 0.03) < 1e-4


def test_calibrate_intercept_rejects_bad_target():
    with pytest.raises(ValueError, match="strictly between"):
        calibrate_intercept(np.array([0.1, 0.2]), 1.5)


def test_vasicek_round_trip():
    x = np.array([0.001, 0.01, 0.05, 0.2, 0.6])
    for z in (-2.0, 0.0, 1.5):
        np.testing.assert_allclose(pit_to_ttc(ttc_to_pit(x, z), z), x, rtol=1e-8)
        np.testing.assert_allclose(ttc_to_pit(pit_to_ttc(x, z), z), x, rtol=1e-8)


def test_vasicek_direction():
    p = np.array([0.02])
    assert ttc_to_pit(p, -1.0)[0] > ttc_to_pit(p, 0.0)[0] > ttc_to_pit(p, 1.0)[0]
    assert ttc_to_pit(p, np.array([-1.0]), rho=0.3)[0] > ttc_to_pit(p, np.array([-1.0]), rho=0.1)[0]


def test_macro_z_score_sign_and_scale():
    macro = macro_unemployment_series()
    z = macro_z_score(macro)
    assert z.loc["2020-06-01"] < -1.5
    assert abs(z.mean()) < 1e-9
    assert abs(z.std() - 1.0) < 1e-9
    z_change = macro_z_score(macro, window=12)
    assert z_change.iloc[:12].eq(0.0).all()
    assert z_change.loc["2020-06-01"] < 0


def test_macro_z_score_constant_series():
    z = macro_z_score(pd.Series([5.0, 5.0, 5.0]))
    assert z.eq(0.0).all()


def test_fit_factor_loading_recovers_truth():
    rng = np.random.default_rng(5)
    n = 40_000
    p_ttc = rng.beta(2, 60, n)
    z = rng.normal(size=n)
    true_loading = 0.6
    y = (rng.random(n) < ttc_to_pit(p_ttc, true_loading * z, 0.15)).astype(np.int64)
    loading, shift = fit_factor_loading(p_ttc, y, z, rho=0.15)
    assert abs(loading - true_loading) < 0.15
    fitted = ttc_to_pit(shift_logit(p_ttc, shift), loading * z, 0.15)
    assert abs(fitted.mean() - y.mean()) < 1e-4
