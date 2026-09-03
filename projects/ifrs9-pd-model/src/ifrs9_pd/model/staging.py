"""IFRS 9 stage allocation (significant increase in credit risk).

Rules, applied in order of precedence:

1. Stage 3 (credit-impaired) if ``current_dpd >= default_dpd``.
2. Stage 2 if ``current_dpd >= dpd_backstop`` (IFRS 9 B5.5.19 rebuttable
   presumption at 30 days past due), **or** if the lifetime-risk proxy has
   increased significantly: ``PD_now / PD_orig >= relative_pd_threshold`` and
   ``PD_now - PD_orig >= absolute_pd_threshold_bps / 10 000``.
3. Stage 1 otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ifrs9_pd.config import StagingConfig
from ifrs9_pd.data.schema import Col


def assign_stage(
    df: pd.DataFrame,
    pd_now: NDArray[np.float64],
    pd_at_origination: NDArray[np.float64],
    cfg: StagingConfig,
) -> NDArray[np.int64]:
    """Vectorised stage allocation.

    Args:
        df: Portfolio rows; only ``current_dpd`` is read.
        pd_now: Current 12-month PIT PD per row.
        pd_at_origination: 12-month PIT PD per row at origination.
        cfg: Staging thresholds.

    Returns:
        Integer array of stages in ``{1, 2, 3}``.
    """
    dpd = df[Col.CURRENT_DPD.value].to_numpy(dtype=np.float64)
    now = np.asarray(pd_now, dtype=np.float64)
    orig = np.clip(np.asarray(pd_at_origination, dtype=np.float64), 1e-12, None)
    sicr = (now / orig >= cfg.relative_pd_threshold) & (
        now - orig >= cfg.absolute_pd_threshold_bps / 10_000.0
    )
    stage = np.ones(len(df), dtype=np.int64)
    stage[sicr | (dpd >= cfg.dpd_backstop)] = 2
    stage[dpd >= cfg.default_dpd] = 3
    return stage
