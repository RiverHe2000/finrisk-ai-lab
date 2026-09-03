import numpy as np
import pytest

from ifrs9_pd.config import ScenarioConfig
from ifrs9_pd.model.term_structure import (
    build_term_structure,
    monthly_hazard,
    seasoning_curve,
    z_path,
)


@pytest.fixture(scope="module")
def ts():
    pd12 = np.array([0.005, 0.02, 0.1, 0.4])
    return build_term_structure(
        pd12, ScenarioConfig(), z0=0.0, horizon=60, age=np.array([0, 12, 30, 59])
    )


def test_shapes(ts):
    assert ts.marginal.shape == ts.cumulative.shape == ts.survival.shape == (4, 60)
    assert set(ts.by_scenario) == {"base", "upside", "downside"}


def test_cumulative_monotone_and_bounded(ts):
    assert np.all(np.diff(ts.cumulative, axis=1) >= -1e-15)
    assert np.all((ts.cumulative >= 0) & (ts.cumulative <= 1))
    for curves in ts.by_scenario.values():
        assert np.all(np.diff(curves.cumulative, axis=1) >= -1e-15)


def test_survival_plus_cumulative_is_one(ts):
    np.testing.assert_allclose(ts.survival + ts.cumulative, 1.0)
    np.testing.assert_allclose(np.cumsum(ts.marginal, axis=1), ts.cumulative, atol=1e-12)


def test_weighting_equals_weighted_mean(ts):
    cfg = ScenarioConfig()
    expected = sum(cfg.weights[k] * ts.by_scenario[k].cumulative for k in cfg.weights)
    np.testing.assert_allclose(ts.cumulative, expected)


def test_scenario_ordering(ts):
    assert np.all(ts.by_scenario["downside"].cumulative > ts.by_scenario["base"].cumulative)
    assert np.all(ts.by_scenario["base"].cumulative > ts.by_scenario["upside"].cumulative)


def test_monthly_hazard_consistent_with_pd12():
    pd12 = np.array([0.01, 0.2])
    h = monthly_hazard(pd12)
    np.testing.assert_allclose(1 - (1 - h) ** 12, pd12)


def test_seasoning_curve_normalised():
    curve = seasoning_curve(60, peak_month=24, peak_multiplier=1.4)
    assert curve.shape == (1, 60)
    assert curve[0, :12].mean() == pytest.approx(1.0)
    aged = seasoning_curve(60, age=np.array([0, 40]))
    assert aged.shape == (2, 60)
    np.testing.assert_allclose(aged[:, :12].mean(axis=1), 1.0)
    assert int(np.argmax(aged[0])) == 23  # peak at age 24 for a new loan


def test_base_scenario_preserves_pd12_without_factor():
    pd12 = np.array([0.03])
    scen = ScenarioConfig(weights={"base": 1.0}, z_shifts={"base": 0.0})
    ts = build_term_structure(pd12, scen, z0=0.0, mean_reversion=0.0)
    # with z = 0 the Vasicek-conditional PD sits below the unconditional PD
    assert ts.cumulative[0, 11] < pd12[0]
    assert ts.cumulative[0, 11] > 0.5 * pd12[0]


def test_z_path_decays():
    path = z_path(np.array([1.0, -2.0]), 5, 0.5)
    np.testing.assert_allclose(path[0], [1.0, 0.5, 0.25, 0.125, 0.0625])
    assert path.shape == (2, 5)


def test_scenario_config_validation():
    with pytest.raises(ValueError, match="sum to 1"):
        ScenarioConfig(weights={"a": 0.6, "b": 0.6}, z_shifts={"a": 0.0, "b": 1.0})
    with pytest.raises(ValueError, match="same scenarios"):
        ScenarioConfig(weights={"a": 1.0}, z_shifts={"b": 0.0})
