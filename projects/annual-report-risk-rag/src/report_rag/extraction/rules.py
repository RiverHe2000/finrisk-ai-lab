"""A transparent regex baseline extractor.

It exists for three reasons: it is the offline fallback, it is the control
arm in evaluation (an LLM must beat it to justify its cost), and it stands in
for the model in tests.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from report_rag.extraction.numbers import (
    NUMBER_RE,
    find_period,
    parse_number,
    split_sentences,
    unit_after,
)
from report_rag.schemas import Chunk, ExtractedMetric, MetricSpec, Unit

_COMPATIBLE: dict[Unit, frozenset[Unit]] = {
    Unit.PERCENT: frozenset({Unit.PERCENT, Unit.BASIS_POINTS}),
    Unit.BASIS_POINTS: frozenset({Unit.PERCENT, Unit.BASIS_POINTS}),
    Unit.AUD_MILLION: frozenset({Unit.AUD_MILLION, Unit.AUD_BILLION}),
    Unit.AUD_BILLION: frozenset({Unit.AUD_MILLION, Unit.AUD_BILLION}),
    Unit.RATIO: frozenset({Unit.RATIO, Unit.TIMES}),
    Unit.TIMES: frozenset({Unit.RATIO, Unit.TIMES}),
}


class RuleBasedExtractor:
    """Find the first sentence mentioning a metric alias and read the number after it."""

    name = "rules"

    def extract(self, spec: MetricSpec, chunks: Sequence[Chunk]) -> ExtractedMetric:
        """Scan chunks best-first; return the first grounded candidate."""
        alias_re = _alias_pattern(spec)
        for chunk in chunks:
            for sentence in split_sentences(chunk.text):
                alias_match = alias_re.search(sentence)
                if not alias_match:
                    continue
                candidate = _first_number_after(sentence, alias_match.end(), spec.unit)
                if candidate is None:
                    continue
                value, unit = candidate
                return ExtractedMetric(
                    metric_id=spec.metric_id,
                    found=True,
                    value=value,
                    unit=unit,
                    period=find_period(sentence) or find_period(chunk.section),
                    evidence_quote=sentence.strip(),
                    chunk_id=chunk.chunk_id,
                    confidence=0.5,
                    notes="rule-based extraction",
                )
        return ExtractedMetric(metric_id=spec.metric_id, found=False, confidence=0.0)


def _alias_pattern(spec: MetricSpec) -> re.Pattern[str]:
    names = sorted({spec.name, *spec.aliases}, key=len, reverse=True)
    escaped = [re.escape(n).replace(r"\ ", r"\s+").replace(r"\-", r"[\-\s]?") for n in names]
    return re.compile("|".join(escaped), re.IGNORECASE)


def _first_number_after(sentence: str, start: int, target: Unit) -> tuple[float, Unit] | None:
    allowed = _COMPATIBLE.get(target, frozenset({target}))
    for match in NUMBER_RE.finditer(sentence, start):
        value = parse_number(match.group(0))
        if value is None:
            continue
        unit = unit_after(sentence, match.end())
        if unit is None and _looks_like_year(match.group(0)):
            continue
        if unit is None:
            # Bare number: accept only when the sentence already declares the target unit.
            if target == Unit.PERCENT and "%" not in sentence:
                continue
            if target in {Unit.AUD_MILLION, Unit.AUD_BILLION} and not re.search(
                r"million|\$m\b", sentence, re.IGNORECASE
            ):
                continue
            unit = target
        if unit in allowed:
            return value, unit
    return None


def _looks_like_year(token: str) -> bool:
    digits = token.strip("()$")
    return digits.isdigit() and len(digits) == 4 and digits.startswith(("19", "20"))
