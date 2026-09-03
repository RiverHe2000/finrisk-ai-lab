"""Number and period parsing helpers shared by the rule-based extractor and the validator."""

from __future__ import annotations

import re

from report_rag.schemas import Unit

NUMBER_RE = re.compile(r"(?<![\w.])(-?\(?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\)?")
PERIOD_RE = re.compile(r"\b(FY\s?20\d{2}|(?:1H|2H)\s?20\d{2}|20\d{2})\b", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z(\"'])")

_PERCENT_HINT = re.compile(r"%|per\s?cent|percent|percentage points?", re.IGNORECASE)
_MILLION_HINT = re.compile(r"\$?\s?m\b|million|\bmn\b", re.IGNORECASE)
_BILLION_HINT = re.compile(r"\bbn\b|billion|\bb\b", re.IGNORECASE)
_BPS_HINT = re.compile(r"\bbps\b|basis points?", re.IGNORECASE)


def parse_number(token: str) -> float | None:
    """Parse ``'1,234.5'``, ``'$12.3'`` or ``'(45)'`` (accounting negative) into a float."""
    cleaned = token.strip()
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()").replace("$", "").replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries while keeping every sentence a verbatim substring."""
    return [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def unit_after(text: str, end: int) -> Unit | None:
    """Infer the unit from the characters immediately following a number."""
    tail = text[end : end + 18]
    if _PERCENT_HINT.match(tail.lstrip()) or _PERCENT_HINT.search(tail[:4]):
        return Unit.PERCENT
    if _BPS_HINT.search(tail[:14]):
        return Unit.BASIS_POINTS
    if _BILLION_HINT.search(tail[:10]):
        return Unit.AUD_BILLION
    if _MILLION_HINT.search(tail[:10]):
        return Unit.AUD_MILLION
    return None


def find_period(text: str) -> str | None:
    """Return the first period-like token (``FY2025``, ``2H2025``, ``2025``) in ``text``."""
    match = PERIOD_RE.search(text)
    return match.group(1).upper().replace(" ", "") if match else None


def number_variants(value: float) -> set[str]:
    """Lossless renderings of a number as it may appear in prose.

    Only renderings that parse back to the same value are returned, so
    ``12.34`` never matches a quote that says ``12.3`` (that would be a
    hallucinated digit), while ``12.3`` does match ``12.30`` or ``1,480``
    matches ``1480``.
    """
    variants: set[str] = set()
    magnitude = abs(value)
    for fmt in ("{:g}", "{:.1f}", "{:.2f}", "{:,.0f}", "{:,.1f}", "{:,.2f}", "{:.0f}"):
        rendered = fmt.format(magnitude)
        if float(rendered.replace(",", "")) == magnitude:
            variants.add(rendered)
    variants |= {f"({v})" for v in list(variants)}  # accounting negatives
    return variants


def to_spec_unit(value: float, unit: Unit | None, target: Unit) -> tuple[float, Unit]:
    """Convert between AUD millions/billions and percent/bps where unambiguous."""
    if unit is None or unit == target:
        return value, target
    if unit == Unit.AUD_BILLION and target == Unit.AUD_MILLION:
        return value * 1_000.0, target
    if unit == Unit.AUD_MILLION and target == Unit.AUD_BILLION:
        return value / 1_000.0, target
    if unit == Unit.BASIS_POINTS and target == Unit.PERCENT:
        return value / 100.0, target
    if unit == Unit.PERCENT and target == Unit.BASIS_POINTS:
        return value * 100.0, target
    return value, unit
