"""Matplotlib figures for the validation report (Agg backend, PNG output)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

FIGSIZE = (6.0, 4.0)
DPI = 110


def _save(fig: matplotlib.figure.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return path


def _roc_points(
    y: NDArray[np.int64], p: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    order = np.argsort(-p, kind="stable")
    tp = np.concatenate([[0.0], np.cumsum(y[order])]) / max(int(y.sum()), 1)
    fp = np.concatenate([[0.0], np.cumsum(1 - y[order])]) / max(int((1 - y).sum()), 1)
    return fp, tp


def roc_curve_plot(
    samples: Mapping[str, tuple[NDArray[np.int64], NDArray[np.float64]]], path: Path
) -> Path:
    """Plot ROC curves for several samples.

    Args:
        samples: Mapping of label to ``(y, pd)`` arrays.
        path: Output PNG path.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, (y, p) in samples.items():
        fp, tp = _roc_points(np.asarray(y, dtype=np.int64), np.asarray(p, dtype=np.float64))
        ax.plot(fp, tp, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    return _save(fig, path)


def calibration_plot(
    predicted: Sequence[float], observed: Sequence[float], counts: Sequence[int], path: Path
) -> Path:
    """Predicted versus observed default rate per bin with the 45-degree line.

    Args:
        predicted: Mean predicted PD per bin.
        observed: Observed default rate per bin.
        counts: Observations per bin (marker size).
        path: Output PNG path.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    size = 20.0 + 80.0 * np.asarray(counts, dtype=np.float64) / max(*counts, 1)
    ax.scatter(predicted, observed, s=size, alpha=0.8, label="PD bins")
    lim = max(*predicted, *observed) * 1.1
    ax.plot([0, lim], [0, lim], linestyle="--", color="grey", linewidth=1, label="perfect")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration (out-of-time)")
    ax.legend()
    return _save(fig, path)


def score_distribution_plot(
    scores: Mapping[str, NDArray[np.float64]], path: Path, bins: int = 30
) -> Path:
    """Overlayed score histograms (density) for several samples."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    all_scores = np.concatenate([np.asarray(s, dtype=np.float64) for s in scores.values()])
    edges = np.linspace(all_scores.min(), all_scores.max(), bins + 1)
    for label, values in scores.items():
        ax.hist(values, bins=edges.tolist(), density=True, alpha=0.5, label=label)
    ax.set_xlabel("Scorecard points")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution")
    ax.legend()
    return _save(fig, path)


def psi_bar_plot(labels: Sequence[str], values: Sequence[float], path: Path, title: str) -> Path:
    """Horizontal bar chart of PSI/CSI values with the 0.10 / 0.25 guide lines."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="steelblue")
    ax.set_yticks(y_pos, labels=list(labels))
    ax.invert_yaxis()
    for level, colour in ((0.10, "orange"), (0.25, "red")):
        ax.axvline(level, linestyle="--", color=colour, linewidth=1)
    ax.set_xlabel("Stability index")
    ax.set_title(title)
    return _save(fig, path)


def ecl_by_stage_plot(
    stages: Sequence[str], ecl: Sequence[float], coverage: Sequence[float], path: Path
) -> Path:
    """ECL amount per stage with coverage ratio annotated."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(stages))
    bars = ax.bar(x, np.asarray(ecl, dtype=np.float64) / 1e6, color="steelblue")
    ax.set_xticks(x, labels=[f"Stage {s}" for s in stages])
    ax.set_ylabel("ECL ($m)")
    ax.set_title("ECL by stage")
    for bar, cov in zip(bars, coverage, strict=True):
        ax.annotate(
            f"{cov:.2%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )
    return _save(fig, path)


def pd_term_structure_plot(curves: Mapping[str, NDArray[np.float64]], path: Path) -> Path:
    """Average cumulative PD by month for each scenario and the weighted curve."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, values in curves.items():
        months = np.arange(1, len(values) + 1)
        style = "-" if label == "weighted" else "--"
        ax.plot(months, values, style, label=label, linewidth=2 if label == "weighted" else 1)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative PD")
    ax.set_title("Lifetime PD term structure")
    ax.legend()
    return _save(fig, path)
