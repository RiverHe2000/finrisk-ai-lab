import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ifrs9_pd.config import MetricThreshold, ValidationThresholds
from ifrs9_pd.validation import (
    ScoredSample,
    auc,
    binomial_test_by_grade,
    bootstrap_ci,
    brier_score,
    calibration_table,
    cap_curve,
    csi_by_feature,
    expected_vs_observed_overall,
    gini,
    hosmer_lemeshow,
    jeffreys_test,
    ks_statistic,
    out_of_time_backtest,
    psi,
    psi_table,
    rank_ordering,
    traffic_light,
    worst_rating,
)


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(0)
    p = rng.beta(1, 20, 5000)
    y = (rng.random(5000) < p).astype(np.int64)
    return y, p


def test_auc_matches_sklearn(sample):
    y, p = sample
    assert auc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert gini(y, p) == pytest.approx(2 * roc_auc_score(y, p) - 1)
    _, ar = cap_curve(y, p)
    assert ar == pytest.approx(gini(y, p))


def test_auc_with_ties():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.5, 0.5, 0.9, 0.5, 0.1])
    assert auc(y, p) == pytest.approx(roc_auc_score(y, p))


def test_single_class_is_nan():
    assert np.isnan(auc(np.zeros(5, dtype=int), np.linspace(0, 1, 5)))
    assert np.isnan(ks_statistic(np.ones(5, dtype=int), np.linspace(0, 1, 5)))


def test_ks_on_separable_data():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)])
    assert ks_statistic(y, p) == pytest.approx(1.0)
    assert gini(y, p) == pytest.approx(1.0)
    assert 0 < ks_statistic(y, np.random.default_rng(1).random(100)) < 0.5


def test_bootstrap_ci_contains_point_estimate(sample):
    y, p = sample
    lo, hi = bootstrap_ci(gini, y, p, n_boot=50, seed=1)
    assert lo < gini(y, p) < hi
    assert bootstrap_ci(gini, y, p, n_boot=50, seed=1) == (lo, hi)


def test_cap_curve_shape(sample):
    curve, _ = cap_curve(*sample, n_points=11)
    assert len(curve) == 11
    assert curve["default_share"].iloc[0] == 0.0
    assert curve["default_share"].iloc[-1] == pytest.approx(1.0)
    assert curve["default_share"].is_monotonic_increasing


def test_rank_ordering():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    grade = np.array([1, 1, 2, 2, 3, 3, 3, 3])
    table, monotonic = rank_ordering(y, grade, min_defaults=0)
    assert monotonic
    assert table["default_rate"].tolist() == [0.0, 0.5, 0.75]
    _, monotonic = rank_ordering(y, np.array([3, 3, 2, 2, 1, 1, 1, 1]), min_defaults=0)
    assert not monotonic
    _, tolerant = rank_ordering(y, np.array([3, 3, 2, 2, 1, 1, 1, 1]), min_defaults=5)
    assert tolerant


def test_psi_zero_for_identical_and_positive_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, a) == pytest.approx(0.0, abs=1e-12)
    shifted = psi(a, a + 1.0)
    assert shifted > 0.25
    assert psi(a, rng.normal(size=5000)) < 0.05
    table = psi_table(a, a + 1.0)
    assert table["contribution"].sum() == pytest.approx(shifted)
    assert table["expected_share"].sum() == pytest.approx(1.0, abs=1e-3)


def test_psi_handles_missing_and_empty_bins():
    a = np.array([0.0] * 100 + [np.nan] * 10)
    b = np.array([1.0] * 100)
    value = psi(a, b)
    assert np.isfinite(value)
    assert value > 0


def test_csi_by_feature(dev_oot):
    dev, oot = dev_oot
    table = csi_by_feature(dev, oot, ["bureau_score", "loan_to_value"], ["region"], bins=5)
    assert set(table["feature"]) == {"bureau_score", "loan_to_value", "region"}
    assert (table["csi"] >= 0).all()
    assert table["csi"].is_monotonic_decreasing


def test_hosmer_lemeshow_calibrated_has_high_p(sample):
    y, p = sample
    _, p_value, table = hosmer_lemeshow(y, p)
    assert p_value > 0.05
    assert len(table) == 10
    _, p_bad, _ = hosmer_lemeshow(y, np.clip(p * 3, 0, 1))
    assert p_bad < 0.01


def test_calibration_table_columns(sample):
    table = calibration_table(*sample, n_bins=5)
    assert list(table.columns) == [
        "bin",
        "n",
        "defaults",
        "mean_pd",
        "observed_rate",
        "expected_defaults",
    ]
    assert table["n"].sum() == len(sample[0])


def test_binomial_and_jeffreys_shapes(sample):
    y, p = sample
    grade = np.digitize(p, [0.02, 0.05, 0.1]) + 1
    binom = binomial_test_by_grade(y, p, grade)
    jeff = jeffreys_test(y, p, grade)
    assert list(binom.columns) == [
        "grade",
        "n",
        "defaults",
        "predicted_pd",
        "observed_rate",
        "p_value",
    ]
    assert list(jeff.columns) == list(binom.columns)
    assert len(binom) == len(np.unique(grade))
    assert binom["p_value"].between(0, 1).all()
    assert jeff["p_value"].between(0, 1).all()
    # calibrated sample: no grade should be strongly rejected
    assert (binom["p_value"] > 0.01).all()


def test_jeffreys_flags_underestimation():
    y = np.array([1] * 30 + [0] * 70)
    p = np.full(100, 0.05)
    jeff = jeffreys_test(y, p, np.ones(100, dtype=int))
    assert jeff["p_value"].iloc[0] < 0.001


def test_brier_and_overall(sample):
    y, p = sample
    assert brier_score(y, p) == pytest.approx(np.mean((p - y) ** 2))
    overall = expected_vs_observed_overall(y, p)
    assert overall["n"] == len(y)
    assert overall["p_value"] > 0.05
    assert overall["pd_to_dr_ratio"] == pytest.approx(p.mean() / y.mean())


def test_backtest_keys(sample):
    y, p = sample
    s = ScoredSample(y, p, 600 - 20 * np.log(p / (1 - p)))
    out = out_of_time_backtest(s, s)
    assert out["gini_deterioration"] == pytest.approx(0.0)
    assert out["score_psi"] == pytest.approx(0.0, abs=1e-12)
    assert out["dev_gini"] == pytest.approx(gini(y, p))


def test_traffic_light():
    th = ValidationThresholds()
    assert traffic_light(0.6, th.gini) == "green"
    assert traffic_light(0.45, th.gini) == "amber"
    assert traffic_light(0.2, th.gini) == "red"
    assert traffic_light(0.05, th.psi) == "green"
    assert traffic_light(0.2, th.psi) == "amber"
    assert traffic_light(0.5, th.psi) == "red"
    assert traffic_light(float("nan"), th.gini) == "red"
    assert worst_rating(["green", "amber", "green"]) == "amber"
    assert worst_rating([]) == "green"
    with pytest.raises(ValueError, match="stricter"):
        MetricThreshold(green=0.1, amber=0.5)
