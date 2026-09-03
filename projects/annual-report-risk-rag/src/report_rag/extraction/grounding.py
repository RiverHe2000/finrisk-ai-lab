"""Post-hoc evidence checks: every accepted number must be traceable to the source text."""

from __future__ import annotations

import re
from collections.abc import Mapping

from report_rag.extraction.numbers import number_variants, to_spec_unit
from report_rag.schemas import Chunk, ExtractedMetric, GroundingStatus, MetricSpec, ValidatedMetric

_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


class GroundingValidator:
    """Reject extractions whose quote or value cannot be verified against the chunks.

    Checks, in order:

    1. ``found`` is false -> ``NOT_FOUND`` (not an error, just nothing to accept).
    2. The cited chunk was actually retrieved and contains the quote verbatim
       (whitespace-insensitive) -> otherwise ``QUOTE_NOT_IN_CHUNK``.
    3. The numeric value appears inside the quote -> otherwise ``VALUE_NOT_IN_QUOTE``.
    4. The value, converted to the metric's canonical unit, sits inside the
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
            status = self._check_value(extracted)
        if status is None:
            value, unit = to_spec_unit(extracted.value, extracted.unit, spec.unit)
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
    def _check_value(extracted: ExtractedMetric) -> GroundingStatus | None:
        assert extracted.value is not None
        assert extracted.evidence_quote is not None
        quote = extracted.evidence_quote
        if any(variant in quote for variant in number_variants(extracted.value)):
            return None
        return GroundingStatus.VALUE_NOT_IN_QUOTE

    @staticmethod
    def _check_range(spec: MetricSpec, value: float) -> GroundingStatus:
        if spec.typical_range is not None:
            low, high = spec.typical_range
            if not low <= value <= high:
                return GroundingStatus.OUT_OF_RANGE
        return GroundingStatus.GROUNDED
