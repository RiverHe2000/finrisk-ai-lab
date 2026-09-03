"""Serialisable container for every metric and table produced by a run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

Table = list[dict[str, Any]]
"""Rows of a table as plain dictionaries (JSON-friendly)."""


def frame_to_table(df: pd.DataFrame) -> Table:
    """Convert a DataFrame to a list of row dictionaries with native Python scalars."""
    rows: Table = []
    for record in df.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key, value in record.items():
            if hasattr(value, "item"):
                value = value.item()  # noqa: PLW2901 - unwrap numpy scalar
            row[str(key)] = value
        rows.append(row)
    return rows


def round_floats(obj: Any, ndigits: int = 6) -> Any:
    """Recursively round floats (``nan``/``inf`` become ``None``) for stable JSON."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [round_floats(v, ndigits) for v in obj]
    return obj


class SampleMetrics(BaseModel):
    """Discrimination and calibration statistics of one sample."""

    name: str
    n: int
    defaults: int
    default_rate: float
    mean_pd: float
    auc: float
    gini: float
    gini_ci_low: float
    gini_ci_high: float
    ks: float
    ks_ci_low: float
    ks_ci_high: float
    brier: float
    hl_statistic: float
    hl_p_value: float
    pd_to_dr_ratio: float
    binomial_p_value_overall: float
    rank_ordering_monotonic: bool
    rank_ordering_table: Table
    calibration_table: Table
    binomial_table: Table
    jeffreys_table: Table
    cap_accuracy_ratio: float


class DataQuality(BaseModel):
    """Row counts, default rates and missing-value rates."""

    n_total: int
    n_dev: int
    n_oot: int
    n_dev_performing: int
    n_oot_performing: int
    default_rate_total: float
    default_rate_dev: float
    default_rate_oot: float
    missing_rates: dict[str, float]
    first_snapshot: str
    last_snapshot: str
    dev_cutoff: str
    default_rate_by_year: Table


class ModelDesign(BaseModel):
    """Binning, feature selection, coefficients and master scale."""

    candidate_features: list[str]
    selected_features: list[str]
    iv_table: Table
    binning_tables: dict[str, Table]
    coefficients: Table
    scorecard_table: Table
    master_scale: Table
    calibration_shift: float
    calibration_target_rate: float
    asset_correlation: float
    factor_loading: float


class StabilityResults(BaseModel):
    """Score PSI and per-feature CSI."""

    score_psi: float
    score_psi_table: Table
    csi_table: Table


class IFRS9Results(BaseModel):
    """PIT/TTC, term structure, staging and ECL results on the reporting book."""

    n_loans: int
    mean_z_now: float
    mean_pd_ttc: float
    mean_pd_pit: float
    mean_pd_at_origination: float
    weighted_pd_12m: float
    weighted_pd_lifetime: float
    term_structure: Table
    stage_distribution: Table
    ecl_by_stage: Table
    scenario_sensitivity: Table
    total_ead: float
    total_ecl: float
    coverage_ratio: float


class Finding(BaseModel):
    """One auto-generated validation finding."""

    metric: str
    value: float
    rating: str
    severity: str
    message: str


class ValidationResults(BaseModel):
    """Everything the report needs, serialisable to JSON."""

    package_version: str
    seed: int
    config: dict[str, Any]
    data_quality: DataQuality
    model_design: ModelDesign
    samples: dict[str, SampleMetrics]
    backtest: dict[str, float]
    stability: StabilityResults
    ifrs9: IFRS9Results
    traffic_lights: dict[str, str]
    findings: list[Finding] = Field(default_factory=list)
    overall_rating: str
    figures: dict[str, str] = Field(default_factory=dict)

    def to_json(self, path: Path, ndigits: int = 6) -> Path:
        """Write the results as deterministic JSON (sorted keys, rounded floats)."""
        payload = round_floats(self.model_dump(mode="json"), ndigits)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: Path) -> ValidationResults:
        """Load results written by :meth:`to_json`."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
