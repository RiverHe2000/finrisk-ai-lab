"""Traffic-light rating of validation metrics.

Bands (see :class:`ifrs9_pd.config.ValidationThresholds` for the rationale):

* Gini: green ``>= 0.50``, amber ``0.40-0.50``, red ``< 0.40``.
* KS: green ``>= 0.30``, amber ``0.20-0.30``, red ``< 0.20``.
* PSI / CSI: green ``< 0.10``, amber ``0.10-0.25``, red ``> 0.25``.
* Hosmer-Lemeshow / binomial p-values: green ``>= 0.05``, amber ``0.01-0.05``,
  red ``< 0.01``.
* Gini deterioration dev to OOT: green ``< 0.05``, amber ``0.05-0.10``,
  red ``> 0.10``.
* Mean PD to observed default rate: green ``>= 0.85``, amber ``0.70-0.85``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from ifrs9_pd.config import MetricThreshold, ValidationThresholds

Rating = Literal["green", "amber", "red"]
DEFAULT_THRESHOLDS = ValidationThresholds()
_SEVERITY: dict[str, int] = {"green": 0, "amber": 1, "red": 2}


def traffic_light(value: float, threshold: MetricThreshold) -> Rating:
    """Rate a metric value against green/amber boundaries.

    Args:
        value: Observed metric value (``nan`` rates red).
        threshold: Boundaries and direction.

    Returns:
        ``"green"``, ``"amber"`` or ``"red"``.
    """
    if value != value:  # noqa: PLR0124 - nan check without numpy
        return "red"
    if threshold.higher_is_better:
        if value >= threshold.green:
            return "green"
        return "amber" if value >= threshold.amber else "red"
    if value <= threshold.green:
        return "green"
    return "amber" if value <= threshold.amber else "red"


def worst_rating(ratings: Iterable[Rating]) -> Rating:
    """Return the most severe rating in ``ratings`` (green if empty)."""
    worst: Rating = "green"
    for rating in ratings:
        if _SEVERITY[rating] > _SEVERITY[worst]:
            worst = rating
    return worst
