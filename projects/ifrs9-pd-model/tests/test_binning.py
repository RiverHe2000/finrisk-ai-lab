import numpy as np
import pandas as pd
import pytest

from ifrs9_pd.data.schema import Col
from ifrs9_pd.features.binning import WoEBinner, select_features

TARGET = Col.DEFAULT_12M.value


@pytest.fixture(scope="module")
def binner(dev_oot):
    dev, _ = dev_oot
    return WoEBinner(
        [Col.BUREAU_SCORE.value, Col.LTV.value, Col.ARREARS_12M.value],
        [Col.EMPLOYMENT_TYPE.value, Col.REGION.value],
        n_bins=10,
        min_bin_share=0.02,
    ).fit(dev, TARGET)


def test_numeric_bins_monotonic(binner):
    table = binner.binning_table(Col.BUREAU_SCORE.value)
    regular = table[table["label"] != "missing"]
    rates = regular["event_rate"].to_numpy()
    assert len(regular) >= 3
    assert np.all(np.diff(rates) <= 1e-12)  # higher score -> lower default rate
    assert np.all(np.diff(regular["woe"].to_numpy()) >= -1e-12)


def test_missing_bin_present(binner):
    table = binner.tables_[Col.BUREAU_SCORE.value]
    assert table.missing is not None
    assert table.missing.count > 0
    assert binner.binning_table(Col.BUREAU_SCORE.value)["label"].iloc[-1] == "missing"


def test_iv_non_negative_and_ranked(binner):
    iv = binner.iv_table()
    assert (iv["iv"] >= 0).all()
    assert iv["iv"].is_monotonic_decreasing
    assert iv["feature"].iloc[0] == Col.BUREAU_SCORE.value
    assert binner.information_value(Col.BUREAU_SCORE.value) == iv["iv"].iloc[0]


def test_bin_min_share(binner, dev_oot):
    n = len(dev_oot[0])
    for feature in binner.numeric_features:
        counts = [b.count for b in binner.tables_[feature].bins]
        assert min(counts) >= 0.02 * n * 0.999


def test_transform_matches_table(binner, dev_oot):
    dev, _ = dev_oot
    woe = binner.transform(dev)
    assert list(woe.columns) == binner.features
    table = binner.tables_[Col.BUREAU_SCORE.value]
    valid = np.array(sorted({b.woe for b in table.bins} | {table.missing_woe}))
    encoded = woe[Col.BUREAU_SCORE.value].to_numpy()
    assert np.all(np.isin(np.round(encoded, 10), np.round(valid, 10)))
    missing = dev[Col.BUREAU_SCORE.value].isna().to_numpy()
    assert np.allclose(encoded[missing], table.missing_woe)


def test_transform_unseen_category_maps_to_other(binner):
    table = binner.tables_[Col.REGION.value]
    series = pd.Series(["NSW", "Mars", None])
    out = binner.transform_feature(series, Col.REGION.value)
    assert out[0] == table.bins[table.category_map["NSW"]].woe
    assert out[1] == table.other_woe
    assert out[2] == table.other_woe or "missing" in table.category_map


def test_rare_categories_grouped_to_other():
    rng = np.random.default_rng(0)
    n = 2000
    cat = rng.choice(["a", "b", "rare1", "rare2"], n, p=[0.6, 0.38, 0.01, 0.01])
    y = (rng.random(n) < np.where(cat == "a", 0.02, 0.08)).astype(int)
    df = pd.DataFrame({"cat": cat, "y": y})
    binner = WoEBinner([], ["cat"], min_bin_share=0.05).fit(df, "y")
    labels = [b.label for b in binner.tables_["cat"].bins]
    assert "other" in labels
    assert "rare1" not in labels
    assert binner.tables_["cat"].other_woe != 0.0


def test_no_missing_in_training_gives_zero_missing_woe():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    y = (rng.random(1000) < 1 / (1 + np.exp(-x))).astype(int)
    df = pd.DataFrame({"x": x, "y": y})
    binner = WoEBinner(["x"]).fit(df, "y")
    assert binner.tables_["x"].missing is None
    out = binner.transform_feature(pd.Series([np.nan, 0.0]), "x")
    assert out[0] == 0.0


def test_round_trip_serialisation(binner, dev_oot):
    _, oot = dev_oot
    clone = WoEBinner.from_dict(binner.to_dict())
    pd.testing.assert_frame_equal(clone.transform(oot), binner.transform(oot))


def test_unfitted_raises():
    with pytest.raises(RuntimeError, match="not been fitted"):
        WoEBinner(["x"]).transform(pd.DataFrame({"x": [1.0]}))


def test_select_features():
    iv = pd.DataFrame({"feature": ["a", "b", "c"], "iv": [2.0, 0.5, 0.01]})
    assert select_features(iv, min_iv=0.02) == ["a", "b"]
    assert select_features(iv, min_iv=0.02, max_iv=1.5) == ["b"]


def test_fit_transform(dev_oot):
    dev, _ = dev_oot
    out = WoEBinner([Col.LTV.value]).fit_transform(dev, TARGET)
    assert out.shape == (len(dev), 1)
