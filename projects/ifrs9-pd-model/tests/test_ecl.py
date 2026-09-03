import numpy as np
import pytest

from ifrs9_pd.model.ecl import compute_ecl, discount_factors, ead_profile, portfolio_summary


def _marginal(n=4, horizon=60, pd12=0.024):
    h = 1 - (1 - pd12) ** (1 / 12)
    survival = (1 - h) ** np.arange(horizon)
    return np.tile(survival * h, (n, 1))


def test_stage3_is_ead_times_lgd():
    ead = np.array([100.0, 200.0, 300.0, 400.0])
    out = compute_ecl(ead, 0.35, _marginal(), 0.05, np.array([3, 3, 1, 2]))
    np.testing.assert_allclose(out["ecl_applied"].to_numpy()[:2], ead[:2] * 0.35)
    assert out["ecl_applied"].iloc[2] == out["ecl_12m"].iloc[2]
    assert out["ecl_applied"].iloc[3] == out["ecl_lifetime"].iloc[3]


def test_stage1_at_most_lifetime():
    ead = np.full(4, 1000.0)
    out = compute_ecl(ead, 0.4, _marginal(), 0.05, np.ones(4, dtype=np.int64))
    assert (out["ecl_12m"] <= out["ecl_lifetime"]).all()
    assert (out["ecl_12m"] > 0).all()


def test_discounting_reduces_ecl():
    ead = np.full(4, 1000.0)
    undiscounted = compute_ecl(ead, 0.4, _marginal(), 0.0, np.full(4, 2))
    discounted = compute_ecl(ead, 0.4, _marginal(), 0.08, np.full(4, 2))
    assert (discounted["ecl_lifetime"] < undiscounted["ecl_lifetime"]).all()
    expected = 1000.0 * 0.4 * _marginal()[0].sum()
    assert undiscounted["ecl_lifetime"].iloc[0] == pytest.approx(expected)


def test_discount_factors():
    df = discount_factors(0.05, 24)
    assert df[11] == pytest.approx(1.05**-1)
    assert df[23] == pytest.approx(1.05**-2)
    assert np.all(np.diff(df) < 0)


def test_ead_profile_amortises_and_caps_at_term():
    ead = np.array([1200.0, 600.0])
    flat = ead_profile(ead, 24)
    assert flat.shape == (2, 24)
    assert np.all(flat[0] == 1200.0)
    prof = ead_profile(ead, 24, np.array([12, 48]))
    assert prof[0, 0] == pytest.approx(1200.0)
    assert prof[0, 11] == pytest.approx(100.0)
    assert np.all(prof[0, 12:] == 0.0)
    assert prof[1, 23] > 0


def test_remaining_term_caps_lifetime():
    ead = np.full(4, 1000.0)
    full = compute_ecl(ead, 0.4, _marginal(), 0.05, np.full(4, 2))
    capped = compute_ecl(ead, 0.4, _marginal(), 0.05, np.full(4, 2), remaining_term=np.full(4, 12))
    assert (capped["ecl_lifetime"] < full["ecl_lifetime"]).all()


def test_lgd_vector_and_summary():
    ead = np.array([100.0, 200.0, 300.0, 400.0])
    lgd = np.array([0.2, 0.3, 0.4, 0.5])
    out = compute_ecl(ead, lgd, _marginal(), 0.05, np.array([1, 2, 3, 1]))
    summary = portfolio_summary(out)
    assert summary["stage"].tolist() == ["1", "2", "3", "total"]
    assert summary["ead"].iloc[-1] == pytest.approx(1000.0)
    assert summary["ecl"].iloc[-1] == pytest.approx(out["ecl_applied"].sum())
    assert summary["coverage"].iloc[2] == pytest.approx(0.4)
    assert summary["share_of_ead"].iloc[-1] == pytest.approx(1.0)


def test_invalid_shape():
    with pytest.raises(ValueError, match="shape"):
        compute_ecl(np.array([1.0]), 0.4, np.ones((1, 6)), 0.05, np.array([1]))
