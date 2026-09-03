"""Weight-of-evidence (WoE) binning with monotonic merging.

For a bin *i* with ``good_i`` non-defaults and ``bad_i`` defaults out of
``G`` and ``B`` in total, the (Laplace-smoothed) weight of evidence is::

    WoE_i = ln( (good_i + a) / (G + k a) ) - ln( (bad_i + a) / (B + k a) )

with ``a = 0.5`` and ``k`` the number of bins. Positive WoE therefore means
*lower* risk than the portfolio average, the classic scorecard convention.
The information value is ``IV = sum_i (dist_good_i - dist_bad_i) * WoE_i``.

Numeric features start from quantile bins, which are merged until every bin
holds at least ``min_bin_share`` of observations and event rates are
monotonic in the bin order. Missing values always form their own bin.
Categorical features get one bin per category, with categories below
``min_bin_share`` pooled into ``"other"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field

SMOOTHING = 0.5
MISSING_LABEL = "missing"
OTHER_LABEL = "other"


class BinRow(BaseModel):
    """One bin of a binned feature.

    Attributes:
        label: Human-readable label (interval, category or ``missing``).
        count: Number of observations.
        events: Number of defaults.
        event_rate: ``events / count``.
        woe: Weight of evidence.
        iv: Contribution to the feature's information value.
    """

    label: str
    count: int
    events: int
    event_rate: float
    woe: float
    iv: float


class FeatureBinning(BaseModel):
    """Fitted binning of a single feature.

    Attributes:
        feature: Feature name.
        kind: ``"numeric"`` or ``"categorical"``.
        edges: Interior cut points (numeric only); bin ``i`` covers
            ``edges[i-1] < x <= edges[i]``.
        bins: Regular bins in order (excluding the missing bin).
        missing: The missing-value bin, if any missing values were seen.
        category_map: Category to bin index (categorical only).
        other_woe: WoE applied to unseen or pooled categories.
        iv: Total information value including the missing bin.
    """

    feature: str
    kind: Literal["numeric", "categorical"]
    edges: list[float] = Field(default_factory=list)
    bins: list[BinRow]
    missing: BinRow | None = None
    category_map: dict[str, int] = Field(default_factory=dict)
    other_woe: float = 0.0
    iv: float

    @property
    def missing_woe(self) -> float:
        """WoE applied to missing values (zero when none were seen in training)."""
        return self.missing.woe if self.missing is not None else 0.0

    def table(self) -> pd.DataFrame:
        """Return the binning table (regular bins followed by the missing bin)."""
        rows = [*self.bins] + ([self.missing] if self.missing is not None else [])
        return pd.DataFrame([r.model_dump() for r in rows])


def _woe_and_iv(
    counts: NDArray[np.int64], events: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    goods = counts - events
    k = len(counts)
    dist_good = (goods + SMOOTHING) / (goods.sum() + k * SMOOTHING)
    dist_bad = (events + SMOOTHING) / (events.sum() + k * SMOOTHING)
    woe = np.log(dist_good) - np.log(dist_bad)
    return woe, (dist_good - dist_bad) * woe


def _merge(
    counts: NDArray[np.int64], events: NDArray[np.int64], i: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Merge bin ``i`` into bin ``i + 1``."""
    counts = counts.copy()
    events = events.copy()
    counts[i + 1] += counts[i]
    events[i + 1] += events[i]
    return np.delete(counts, i), np.delete(events, i)


def _enforce_min_share(
    counts: NDArray[np.int64], events: NDArray[np.int64], edges: list[float], min_share: float
) -> tuple[NDArray[np.int64], NDArray[np.int64], list[float]]:
    total = counts.sum()
    while len(counts) > 1 and counts.min() < min_share * total:
        i = int(np.argmin(counts))
        j = i if i < len(counts) - 1 else i - 1  # merge last bin leftwards
        counts, events = _merge(counts, events, j)
        edges = edges[:j] + edges[j + 1 :]
    return counts, events, edges


def _enforce_monotonic(
    counts: NDArray[np.int64], events: NDArray[np.int64], edges: list[float]
) -> tuple[NDArray[np.int64], NDArray[np.int64], list[float]]:
    while len(counts) > 1:
        rate = events / counts
        order = np.arange(len(rate), dtype=np.float64)
        sign = np.sign(np.cov(order, rate, aweights=counts)[0, 1])
        diffs = np.diff(rate) * (sign if sign != 0 else 1.0)
        violations = np.flatnonzero(diffs < 0)
        if violations.size == 0:
            break
        j = int(violations[0])
        counts, events = _merge(counts, events, j)
        edges = edges[:j] + edges[j + 1 :]
    return counts, events, edges


def _interval_labels(edges: Sequence[float]) -> list[str]:
    bounds = [-np.inf, *edges, np.inf]
    return [f"({bounds[i]:.4g}, {bounds[i + 1]:.4g}]" for i in range(len(bounds) - 1)]


def _rows(
    labels: Sequence[str], counts: NDArray[np.int64], events: NDArray[np.int64]
) -> tuple[list[BinRow], NDArray[np.float64]]:
    woe, iv = _woe_and_iv(counts, events)
    rows = [
        BinRow(
            label=label,
            count=int(c),
            events=int(e),
            event_rate=float(e / c) if c else 0.0,
            woe=float(w),
            iv=float(v),
        )
        for label, c, e, w, v in zip(labels, counts, events, woe, iv, strict=True)
    ]
    return rows, woe


def _fit_numeric(
    name: str, x: NDArray[np.float64], y: NDArray[np.int64], n_bins: int, min_share: float
) -> FeatureBinning:
    present = ~np.isnan(x)
    xv, yv = x[present], y[present]
    quantiles = np.quantile(xv, np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = [float(e) for e in np.unique(quantiles) if e < xv.max()]
    idx = np.searchsorted(np.asarray(edges), xv, side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(np.int64)
    events = np.bincount(idx, weights=yv, minlength=len(edges) + 1).astype(np.int64)
    counts, events, edges = _enforce_min_share(counts, events, edges, min_share)
    counts, events, edges = _enforce_monotonic(counts, events, edges)

    n_missing = int((~present).sum())
    all_counts = np.append(counts, n_missing) if n_missing else counts
    all_events = np.append(events, int(y[~present].sum())) if n_missing else events
    labels = _interval_labels(edges) + ([MISSING_LABEL] if n_missing else [])
    rows, _ = _rows(labels, all_counts, all_events)
    regular, missing = (rows[:-1], rows[-1]) if n_missing else (rows, None)
    return FeatureBinning(
        feature=name,
        kind="numeric",
        edges=edges,
        bins=regular,
        missing=missing,
        iv=float(sum(r.iv for r in rows)),
    )


def _fit_categorical(
    name: str, x: pd.Series, y: NDArray[np.int64], min_share: float
) -> FeatureBinning:
    values = x.astype("str").where(x.notna(), MISSING_LABEL)
    shares = values.value_counts(normalize=True)
    keep = [str(c) for c in shares.index if shares[c] >= min_share]
    grouped = values.where(values.isin(keep), OTHER_LABEL)
    stats = (
        pd.DataFrame({"cat": grouped.to_numpy(), "y": y})
        .groupby("cat", sort=True)["y"]
        .agg(["count", "sum"])
    )
    labels = [str(c) for c in stats.index]
    counts = stats["count"].to_numpy(dtype=np.int64)
    events = stats["sum"].to_numpy(dtype=np.int64)
    order = np.argsort(-(events / counts), kind="stable")
    rows, woe = _rows([labels[i] for i in order], counts[order], events[order])
    ordered_labels = [labels[i] for i in order]
    category_map = {
        str(c): ordered_labels.index(str(c) if str(c) in ordered_labels else OTHER_LABEL)
        for c in shares.index
    }
    other_woe = (
        float(woe[ordered_labels.index(OTHER_LABEL)]) if OTHER_LABEL in ordered_labels else 0.0
    )
    return FeatureBinning(
        feature=name,
        kind="categorical",
        bins=rows,
        category_map=category_map,
        other_woe=other_woe,
        iv=float(sum(r.iv for r in rows)),
    )


class WoEBinner:
    """Fit WoE bins on a development sample and encode new data.

    Args:
        numeric_features: Numeric columns to bin.
        categorical_features: Categorical columns to bin.
        n_bins: Initial number of quantile bins for numeric features.
        min_bin_share: Minimum observation share per bin.

    Attributes:
        tables_: Fitted :class:`FeatureBinning` per feature (after ``fit``).
    """

    def __init__(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str] = (),
        n_bins: int = 10,
        min_bin_share: float = 0.05,
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.n_bins = n_bins
        self.min_bin_share = min_bin_share
        self.tables_: dict[str, FeatureBinning] = {}

    @property
    def features(self) -> list[str]:
        """All features in fitting order."""
        return self.numeric_features + self.categorical_features

    def fit(self, df: pd.DataFrame, target: str) -> Self:
        """Fit binning tables for every configured feature.

        Args:
            df: Development sample.
            target: Name of the binary target column.

        Returns:
            The fitted binner.
        """
        y = df[target].to_numpy(dtype=np.int64)
        self.tables_ = {}
        for name in self.numeric_features:
            x = df[name].to_numpy(dtype=np.float64)
            self.tables_[name] = _fit_numeric(name, x, y, self.n_bins, self.min_bin_share)
        for name in self.categorical_features:
            self.tables_[name] = _fit_categorical(name, df[name], y, self.min_bin_share)
        return self

    def _check_fitted(self) -> None:
        if not self.tables_:
            msg = "WoEBinner has not been fitted"
            raise RuntimeError(msg)

    def transform_feature(self, series: pd.Series, name: str) -> NDArray[np.float64]:
        """WoE-encode one feature.

        Args:
            series: Raw feature values.
            name: Feature name (must have been fitted).

        Returns:
            WoE values as a float array.
        """
        self._check_fitted()
        table = self.tables_[name]
        woe = np.array([b.woe for b in table.bins], dtype=np.float64)
        if table.kind == "numeric":
            x = series.to_numpy(dtype=np.float64)
            idx = np.searchsorted(np.asarray(table.edges, dtype=np.float64), x, side="left")
            out = woe[np.where(np.isnan(x), 0, idx)]
            return np.where(np.isnan(x), table.missing_woe, out)
        values = series.astype("str").where(series.notna(), MISSING_LABEL)
        idx_series = values.map(table.category_map)
        known = idx_series.notna().to_numpy()
        out = np.full(len(series), table.other_woe, dtype=np.float64)
        out[known] = woe[idx_series[known].to_numpy(dtype=np.int64)]
        return out

    def transform(self, df: pd.DataFrame, features: Sequence[str] | None = None) -> pd.DataFrame:
        """WoE-encode a DataFrame.

        Args:
            df: Data with the raw feature columns.
            features: Subset of fitted features to encode (default: all).

        Returns:
            DataFrame of WoE columns named after the raw features.
        """
        self._check_fitted()
        cols = list(features) if features is not None else self.features
        return pd.DataFrame(
            {name: self.transform_feature(df[name], name) for name in cols}, index=df.index
        )

    def fit_transform(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Fit and encode in one step."""
        return self.fit(df, target).transform(df)

    def information_value(self, feature: str) -> float:
        """Information value of a fitted feature."""
        self._check_fitted()
        return self.tables_[feature].iv

    def binning_table(self, feature: str) -> pd.DataFrame:
        """Binning table of a fitted feature (see :meth:`FeatureBinning.table`)."""
        self._check_fitted()
        return self.tables_[feature].table()

    def iv_table(self) -> pd.DataFrame:
        """Information value per feature, sorted descending."""
        self._check_fitted()
        rows = [
            {"feature": t.feature, "kind": t.kind, "n_bins": len(t.bins), "iv": t.iv}
            for t in self.tables_.values()
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation of the fitted binner."""
        self._check_fitted()
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "n_bins": self.n_bins,
            "min_bin_share": self.min_bin_share,
            "tables": {name: t.model_dump() for name, t in self.tables_.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        """Rebuild a fitted binner from :meth:`to_dict` output."""
        binner = cls(
            numeric_features=list(payload["numeric_features"]),  # type: ignore[call-overload]
            categorical_features=list(payload["categorical_features"]),  # type: ignore[call-overload]
            n_bins=int(payload["n_bins"]),  # type: ignore[call-overload]
            min_bin_share=float(payload["min_bin_share"]),  # type: ignore[arg-type]
        )
        tables = payload["tables"]
        if not isinstance(tables, dict):
            msg = "payload['tables'] must be a mapping"
            raise TypeError(msg)
        binner.tables_ = {name: FeatureBinning.model_validate(t) for name, t in tables.items()}
        return binner


def select_features(
    iv_table: pd.DataFrame, min_iv: float = 0.02, max_iv: float | None = None
) -> list[str]:
    """Select features whose information value lies within ``[min_iv, max_iv]``.

    Args:
        iv_table: Output of :meth:`WoEBinner.iv_table`.
        min_iv: Features below this IV are considered non-predictive.
        max_iv: Features above this IV are suspiciously strong (possible
            leakage) and are excluded when provided.

    Returns:
        Selected feature names in descending IV order.
    """
    mask = iv_table["iv"] >= min_iv
    if max_iv is not None:
        mask &= iv_table["iv"] <= max_iv
    return [str(f) for f in iv_table.loc[mask, "feature"]]
