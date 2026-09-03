import numpy as np
import pandas as pd
import pytest

from ifrs9_pd.config import StagingConfig
from ifrs9_pd.data.schema import Col
from ifrs9_pd.model.staging import assign_stage


def _frame(dpd):
    return pd.DataFrame({Col.CURRENT_DPD.value: dpd})


def test_hand_crafted_cases():
    cfg = StagingConfig(
        relative_pd_threshold=2.0, absolute_pd_threshold_bps=50, dpd_backstop=30, default_dpd=90
    )
    df = _frame([0, 0, 0, 0, 30, 60, 90, 120])
    pd_now = np.array([0.01, 0.03, 0.0012, 0.011, 0.01, 0.01, 0.01, 0.01])
    pd_orig = np.array([0.01, 0.01, 0.0005, 0.0055, 0.01, 0.01, 0.01, 0.01])
    stage = assign_stage(df, pd_now, pd_orig, cfg)
    # 1: unchanged -> 1; 2: 3x and +200bps -> 2; 3: 2.4x but only +7bps -> 1;
    # 4: exactly 2x and +55bps -> 2; 5/6: backstop -> 2; 7/8: default -> 3
    assert stage.tolist() == [1, 2, 1, 2, 2, 2, 3, 3]
    assert stage.dtype == np.int64


def test_relative_only_not_enough():
    cfg = StagingConfig(absolute_pd_threshold_bps=100)
    stage = assign_stage(_frame([0]), np.array([0.005]), np.array([0.001]), cfg)
    assert stage.tolist() == [1]


def test_absolute_only_not_enough():
    cfg = StagingConfig(relative_pd_threshold=3.0, absolute_pd_threshold_bps=10)
    stage = assign_stage(_frame([0]), np.array([0.05]), np.array([0.03]), cfg)
    assert stage.tolist() == [1]


def test_zero_origination_pd_does_not_divide_by_zero():
    stage = assign_stage(_frame([0]), np.array([0.02]), np.array([0.0]), StagingConfig())
    assert stage.tolist() == [2]


def test_config_validation():
    with pytest.raises(ValueError, match="default_dpd"):
        StagingConfig(dpd_backstop=90, default_dpd=60)
