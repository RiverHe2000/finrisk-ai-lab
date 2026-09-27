"""Post-hoc evidence checks: every accepted number must be traceable to the source text."""

from __future__ import annotations

import re
from collections.abc import Mapping

from report_rag.catalogue import RISK_METRICS
from report_rag.extraction.numbers import NUMBER_RE, PERIOD_RE, parse_number, to_spec_unit
from report_rag.schemas import Chunk, ExtractedMetric, GroundingStatus, MetricSpec, ValidatedMetric

_WS_RE = re.compile(r"\s+")

_UNIT_PATTERNS = (
    (r"\s*(?:%|per\s?cent\b|percent\b)", "percent"),
    (r"\s*(?:bps\b|basis points?\b)", "bps"),
    (r"\s*(?:million\b|mn\b|m\b)", "aud_m"),
    (r"\s*(?:billion\b|bn\b)", "aud_b"),
    (r"\s*times\b", "times"),
)


def _normalise(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def _mentions(quote: str, spec: MetricSpec) -> list[tuple[int, int, str]]:
    """Resolve overlapping aliases longest-first, so CET1 cannot become Tier 1."""
    candidates = []
    for metric in (*RISK_METRICS, spec):
        for alias in {metric.name, *metric.aliases}:
            pattern = re.escape(alias).replace(r"\ ", r"\s+")
            for match in re.finditer(rf"(?<!\w){pattern}(?!\w)", quote, re.IGNORECASE):
                candidates.append((match.start(), match.end(), metric.metric_id))
    mentions: list[tuple[int, int, str]] = []
    for start, end, metric_id in sorted(candidates, key=lambda m: (-(m[1] - m[0]), m[0])):
        if not any(
            start < other_end and end > other_start for other_start, other_end, _ in mentions
        ):
            mentions.append((start, end, metric_id))
    return sorted(mentions)


def _period_verified(period: str | None, quote: str) -> bool:
    """Require an unambiguous explicit period; never assign a column in a multi-year table."""
    if period is None:
        return True
    normalised = period.upper().replace(" ", "")
    if PERIOD_RE.fullmatch(normalised) is None:
        return False
    periods = {m.group().upper().replace(" ", "") for m in PERIOD_RE.finditer(quote)}
    # FY and a bare calendar year are treated as the same annual label, not as a
    # verified fiscal year-end convention. Half-year labels remain distinct.
    canonical = {p.removeprefix("FY") for p in periods}
    return canonical == {normalised.removeprefix("FY")}


class GroundingValidator:
    """Reject extractions whose quote or value cannot be verified against the chunks.

    Checks, in order:

    1. ``found`` is false -> ``NOT_FOUND`` (not an error, just nothing to accept).
    2. The cited chunk was actually retrieved and contains the quote verbatim
       (whitespace-insensitive) -> otherwise ``QUOTE_NOT_IN_CHUNK``.
    3. A complete signed numeric token, explicit unit and preceding metric alias agree.
    4. A supplied period is explicit and unambiguous in the quote.
    5. The value, converted to the metric's canonical unit, sits inside the
       spec's ``typical_range`` -> otherwise ``OUT_OF_RANGE`` (kept, not accepted).
    """

    def validate(
        self,
        spec: MetricSpec,
        extracted: ExtractedMetric,
        chunks: Mapping[str, Chunk],
    ) -> ValidatedMetric:
        """Return the extraction annotated with a grounding status."""
        retrieved_ids = list(chunks)
        if not extracted.found or extracted.value is None:
            return ValidatedMetric(
                metric=extracted.model_copy(update={"found": False}),
                grounding=GroundingStatus.NOT_FOUND,
                retrieved_chunk_ids=retrieved_ids,
                accepted=False,
            )

        status = self._check_quote(extracted, chunks)
        if status is None:
            status = self._check_value(spec, extracted)
        if status is None and not _period_verified(
            extracted.period, extracted.evidence_quote or ""
        ):
            status = GroundingStatus.PERIOD_NOT_VERIFIED
        if status is None:
            value, unit = to_spec_unit(extracted.value, extracted.unit, spec.unit)
            if unit != spec.unit:
                status = GroundingStatus.UNIT_NOT_VERIFIED
            else:
                extracted = extracted.model_copy(update={"value": value, "unit": unit})
                status = self._check_range(spec, value)

        return ValidatedMetric(
            metric=extracted,
            grounding=status,
            retrieved_chunk_ids=retrieved_ids,
            accepted=status is GroundingStatus.GROUNDED,
        )

    @staticmethod
    def _check_quote(
        extracted: ExtractedMetric, chunks: Mapping[str, Chunk]
    ) -> GroundingStatus | None:
        if not extracted.chunk_id or not extracted.evidence_quote:
            return GroundingStatus.QUOTE_NOT_IN_CHUNK
        chunk = chunks.get(extracted.chunk_id)
        if chunk is None or _normalise(extracted.evidence_quote) not in _normalise(chunk.text):
            return GroundingStatus.QUOTE_NOT_IN_CHUNK
        return None

    @staticmethod
    def _check_value(spec: MetricSpec, extracted: ExtractedMetric) -> GroundingStatus | None:
        assert extracted.value is not None
        assert extracted.evidence_quote is not None
        quote = extracted.evidence_quote
        matches = [
            m for m in NUMBER_RE.finditer(quote) if parse_number(m.group()) == extracted.value
        ]
        if not matches:
            return GroundingStatus.VALUE_NOT_IN_QUOTE
        if extracted.metric_id != spec.metric_id or "respectively" in quote.lower():
            return GroundingStatus.METRIC_NOT_VERIFIED
        mentions = _mentions(quote, spec)
        associated = []
        for match in matches:
            preceding = [m for m in mentions if m[1] <= match.start()]
            if not preceding or preceding[-1][2] != spec.metric_id:
                continue
            context = quote[preceding[-1][1] : match.start()]
            if re.search(r"[.!?;]\s|\b(?:while|whereas|but|and)\b", context, re.IGNORECASE):
                continue
            # Do not borrow a later number from an unrecognised metric or comparison.
            earlier_quantity = any(
                any(
                    re.match(pattern, quote[token.end() :], re.IGNORECASE)
                    and extracted.unit is not None
                    and unit == extracted.unit.value
                    for pattern, unit in _UNIT_PATTERNS
                )
                for token in NUMBER_RE.finditer(quote, preceding[-1][1], match.start())
            )
            if earlier_quantity:
                continue
            # Unsupported sign spellings must not turn a negative into a positive token.
            if re.search(r"[-\u2212\u2013\u2014]\s*$", quote[: match.start()]):
                continue
            associated.append(match)
        if not associated:
            return GroundingStatus.METRIC_NOT_VERIFIED
        for match in associated:
            tail = quote[match.end() :]
            if any(
                extracted.unit is not None
                and extracted.unit.value == unit
                and re.match(pattern, tail, re.IGNORECASE)
                for pattern, unit in _UNIT_PATTERNS
            ):
                return None
        return GroundingStatus.UNIT_NOT_VERIFIED

    @staticmethod
    def _check_range(spec: MetricSpec, value: float) -> GroundingStatus:
        if spec.typical_range is not None:
            low, high = spec.typical_range
            if not low <= value <= high:
                return GroundingStatus.OUT_OF_RANGE
        return GroundingStatus.GROUNDED
