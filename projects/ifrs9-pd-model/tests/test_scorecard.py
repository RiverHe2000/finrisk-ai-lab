import json

import numpy as np
import pytest

from ifrs9_pd.config import ModelConfig
from ifrs9_pd.data.schema import Col
from ifrs9_pd.model.scorecard import PDScorecard, build_master_scale, load_model, save_model
from ifrs9_pd.validation.discrimination import gini


def test_fit_and_discrimination(fitted, dev_oot):
    dev, oot = dev_oot
    for df in (dev, oot):
        p = fitted.scorecard.predict_proba(fitted.binner.transform(df))
        assert p.shape == (len(df),)
        assert gini(df[Col.DEFAULT_12M.value].to_numpy(), p) > 0.4


def test_coefficients_negative_on_woe(fitted):
    assert (fitted.scorecard.coef_ < 0).all()


def test_points_monotone_in_pd(fitted):
    p = np.linspace(0.001, 0.5, 50)
    points = fitted.scorecard.to_points(p)
    assert np.all(np.diff(points) < 0)
    cfg = fitted.scorecard.cfg
    at_base = fitted.scorecard.to_points(np.array([1.0 / (1.0 + cfg.base_odds)]))
    assert at_base[0] == pytest.approx(cfg.base_score)
    twice_odds = fitted.scorecard.to_points(np.array([1.0 / (1.0 + 2 * cfg.base_odds)]))
    assert twice_odds[0] - at_base[0] == pytest.approx(cfg.pdo)


def test_scorecard_table_points_sum_to_score(fitted, dev_oot):
    dev, _ = dev_oot
    table = fitted.scorecard.scorecard_table(fitted.binner)
    assert set(table.columns) >= {"feature", "bin", "woe", "coefficient", "points"}
    assert set(table["feature"]) == set(fitted.scorecard.features)
    woe = fitted.binner.transform(dev, fitted.scorecard.features)
    total = np.zeros(len(dev))
    for feature, beta in zip(fitted.scorecard.features, fitted.scorecard.coef_, strict=True):
        n = len(fitted.scorecard.features)
        base = (
            fitted.scorecard.offset
            - fitted.scorecard.factor
            * (fitted.scorecard.intercept_ + fitted.scorecard.calibration_shift_)
        ) / n
        total += -fitted.scorecard.factor * beta * woe[feature].to_numpy() + base
    assert np.allclose(total, fitted.scorecard.score(woe), atol=1e-6)


def test_master_scale_grades_ordered():
    scale = build_master_scale(10, 0.0003, 0.30)
    assert scale.n_grades == 10
    assert np.all(np.diff(scale.boundaries) > 0)
    grades = scale.assign(np.array([0.0001, 0.0003, 0.01, 0.3, 0.9]))
    assert grades.tolist() == [1, 2, 6, 10, 10]
    table = scale.table()
    assert table["pd_mid"].is_monotonic_increasing
    assert table["pd_lower"].iloc[0] == 0.0
    assert table["pd_upper"].iloc[-1] == 1.0


def test_assign_grade_monotone(fitted):
    p = np.geomspace(1e-4, 0.5, 30)
    grades = fitted.scorecard.assign_grade(p)
    assert np.all(np.diff(grades) >= 0)


def test_save_load_round_trip(fitted, dev_oot, tmp_path):
    _, oot = dev_oot
    path = save_model(tmp_path / "model" / "scorecard.json", fitted.scorecard, fitted.binner)
    payload = json.loads(path.read_text())
    assert {"scorecard", "binner"} <= set(payload)
    card, binner = load_model(path)
    assert card.features == fitted.scorecard.features
    assert card.factor_loading_ == fitted.scorecard.factor_loading_
    np.testing.assert_allclose(
        card.predict_proba(binner.transform(oot)),
        fitted.scorecard.predict_proba(fitted.binner.transform(oot)),
    )


def test_coefficient_table(fitted):
    table = fitted.scorecard.coefficient_table()
    assert table["term"].tolist()[-3:] == ["intercept", "calibration_shift", "factor_loading"]


def test_unfitted_scorecard_predicts_half():
    card = PDScorecard(ModelConfig(), ["x"])
    import pandas as pd

    assert card.predict_proba(pd.DataFrame({"x": [0.0]}))[0] == pytest.approx(0.5)
