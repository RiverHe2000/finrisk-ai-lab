"""Logistic-regression scorecard on WoE features with points scaling.

Scaling follows the standard scorecard convention. With ``factor = PDO /
ln 2`` and ``offset = base_score - factor * ln(base_odds)`` the score of a
borrower with good:bad odds ``o`` is ``offset + factor * ln(o)``. Since the
logistic model gives ``ln(o) = -(alpha + sum_j beta_j WoE_j)`` the score
decomposes into per-bin points::

    points_j = -factor * beta_j * WoE_j + (offset - factor * alpha) / n_features

so that the sum of points over features equals the score.

The master scale maps PD to ``n_grades`` rating grades whose interior
boundaries are geometrically spaced between ``grade_min_pd`` and
``grade_max_pd``: grade 1 is the safest (PD below the lowest boundary) and
grade ``n_grades`` the riskiest. With the defaults (10 grades, 0.03%-30%)
each grade roughly multiplies PD by 2.4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from ifrs9_pd.config import ModelConfig
from ifrs9_pd.features.binning import WoEBinner


class MasterScale(BaseModel):
    """Rating grades defined by PD boundaries.

    Attributes:
        boundaries: Interior PD boundaries, increasing; grade ``g`` (1-based)
            covers ``boundaries[g-2] <= PD < boundaries[g-1]``.
    """

    boundaries: list[float]

    @property
    def n_grades(self) -> int:
        """Number of grades."""
        return len(self.boundaries) + 1

    def assign(self, pd_values: NDArray[np.float64]) -> NDArray[np.int64]:
        """Map PDs to 1-based grades (1 = lowest risk)."""
        idx = np.searchsorted(np.asarray(self.boundaries), pd_values, side="right")
        return np.asarray(idx + 1, dtype=np.int64)

    def table(self) -> pd.DataFrame:
        """Grade table with lower/upper PD bounds and geometric midpoint."""
        lower = [0.0, *self.boundaries]
        upper = [*self.boundaries, 1.0]
        mid = [
            float(np.sqrt(lo * hi)) if lo > 0 else hi / 2.0
            for lo, hi in zip(lower, upper, strict=True)
        ]
        return pd.DataFrame(
            {
                "grade": range(1, self.n_grades + 1),
                "pd_lower": lower,
                "pd_upper": upper,
                "pd_mid": mid,
            }
        )


def build_master_scale(n_grades: int, min_pd: float, max_pd: float) -> MasterScale:
    """Build a geometric master scale.

    Args:
        n_grades: Number of grades.
        min_pd: Lowest interior boundary.
        max_pd: Highest interior boundary.

    Returns:
        A :class:`MasterScale` with ``n_grades - 1`` interior boundaries.
    """
    bounds = np.geomspace(min_pd, max_pd, n_grades - 1)
    return MasterScale(boundaries=[float(b) for b in bounds])


class PDScorecard:
    """Logistic scorecard on WoE-encoded features.

    Args:
        cfg: Model configuration (regularisation, scaling, master scale).
        features: WoE feature columns, in the order used for fitting.

    Attributes:
        coef_: Fitted coefficients per feature.
        intercept_: Fitted intercept.
        calibration_shift_: Additive logit shift from central-tendency calibration.
        factor_loading_: Multiplier applied to the standardised macro factor
            before the Vasicek PIT conversion.
    """

    def __init__(self, cfg: ModelConfig, features: list[str]) -> None:
        self.cfg = cfg
        self.features = list(features)
        self.coef_: NDArray[np.float64] = np.zeros(len(features), dtype=np.float64)
        self.intercept_: float = 0.0
        self.calibration_shift_: float = 0.0
        self.factor_loading_: float = 1.0
        self.master_scale = build_master_scale(cfg.n_grades, cfg.grade_min_pd, cfg.grade_max_pd)

    @property
    def factor(self) -> float:
        """Points per unit of log-odds: ``PDO / ln 2``."""
        return float(self.cfg.pdo / np.log(2.0))

    @property
    def offset(self) -> float:
        """Score at odds of 1:1: ``base_score - factor * ln(base_odds)``."""
        return float(self.cfg.base_score - self.factor * np.log(self.cfg.base_odds))

    def _matrix(self, x: pd.DataFrame) -> NDArray[np.float64]:
        return x[self.features].to_numpy(dtype=np.float64)

    def fit(self, x: pd.DataFrame, y: NDArray[np.int64] | pd.Series) -> Self:
        """Fit an L2-regularised (sklearn default) logistic regression.

        Args:
            x: WoE-encoded features (must contain ``self.features``).
            y: Binary target.

        Returns:
            The fitted scorecard.
        """
        model = LogisticRegression(C=self.cfg.regularisation_c, solver="lbfgs")
        model.fit(self._matrix(x), np.asarray(y, dtype=np.int64))
        self.coef_ = np.asarray(model.coef_[0], dtype=np.float64)
        self.intercept_ = float(model.intercept_[0])
        self.calibration_shift_ = 0.0
        self.factor_loading_ = 1.0
        return self

    def decision_function(self, x: pd.DataFrame) -> NDArray[np.float64]:
        """Log-odds of default including any calibration shift."""
        raw = self._matrix(x) @ self.coef_ + self.intercept_ + self.calibration_shift_
        return np.asarray(raw, dtype=np.float64)

    def predict_proba(self, x: pd.DataFrame) -> NDArray[np.float64]:
        """Probability of default per row."""
        return np.asarray(expit(self.decision_function(x)), dtype=np.float64)

    def to_points(self, proba: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert PDs to scorecard points: ``offset + factor * ln((1 - p) / p)``."""
        p = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, 1 - 1e-12)
        return np.asarray(self.offset - self.factor * logit(p), dtype=np.float64)

    def score(self, x: pd.DataFrame) -> NDArray[np.float64]:
        """Scorecard points per row."""
        return self.to_points(self.predict_proba(x))

    def assign_grade(self, proba: NDArray[np.float64]) -> NDArray[np.int64]:
        """Master-scale grade per PD (1 = best)."""
        return self.master_scale.assign(np.asarray(proba, dtype=np.float64))

    def scorecard_table(self, binner: WoEBinner) -> pd.DataFrame:
        """Feature/bin/WoE/coefficient/points table.

        Args:
            binner: The fitted binner whose bins define the rows.

        Returns:
            DataFrame with one row per feature bin (including missing bins).
        """
        base = (self.offset - self.factor * (self.intercept_ + self.calibration_shift_)) / len(
            self.features
        )
        records: list[dict[str, object]] = []
        for name, beta in zip(self.features, self.coef_, strict=True):
            table = binner.tables_[name]
            rows = [*table.bins] + ([table.missing] if table.missing is not None else [])
            records.extend(
                {
                    "feature": name,
                    "bin": row.label,
                    "count": row.count,
                    "event_rate": row.event_rate,
                    "woe": row.woe,
                    "coefficient": float(beta),
                    "points": float(np.round(-self.factor * beta * row.woe + base, 1)),
                }
                for row in rows
            )
        return pd.DataFrame(records)

    def coefficient_table(self) -> pd.DataFrame:
        """Coefficient per feature plus intercept and calibration shift rows."""
        rows = [
            {"term": f, "coefficient": float(b)}
            for f, b in zip(self.features, self.coef_, strict=True)
        ]
        rows.append({"term": "intercept", "coefficient": self.intercept_})
        rows.append({"term": "calibration_shift", "coefficient": self.calibration_shift_})
        rows.append({"term": "factor_loading", "coefficient": self.factor_loading_})
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "config": self.cfg.model_dump(),
            "features": self.features,
            "coef": [float(c) for c in self.coef_],
            "intercept": self.intercept_,
            "calibration_shift": self.calibration_shift_,
            "factor_loading": self.factor_loading_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """Rebuild a fitted scorecard from :meth:`to_dict` output."""
        card = cls(
            ModelConfig.model_validate(payload["config"]), [str(f) for f in payload["features"]]
        )
        card.coef_ = np.asarray(payload["coef"], dtype=np.float64)
        card.intercept_ = float(payload["intercept"])
        card.calibration_shift_ = float(payload["calibration_shift"])
        card.factor_loading_ = float(payload["factor_loading"])
        return card


def save_model(path: Path, scorecard: PDScorecard, binner: WoEBinner) -> Path:
    """Write the scorecard and its binner to one JSON file.

    Args:
        path: Destination ``.json`` path.
        scorecard: Fitted scorecard.
        binner: Fitted binner.

    Returns:
        The path written.
    """
    payload = {"scorecard": scorecard.to_dict(), "binner": binner.to_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_model(path: Path) -> tuple[PDScorecard, WoEBinner]:
    """Load a scorecard and binner written by :func:`save_model`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PDScorecard.from_dict(payload["scorecard"]), WoEBinner.from_dict(payload["binner"])
