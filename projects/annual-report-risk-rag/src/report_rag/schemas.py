"""Pydantic models shared across ingestion, extraction and evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Unit(StrEnum):
    """Units a prudential metric can be reported in."""

    PERCENT = "percent"
    BASIS_POINTS = "bps"
    AUD_MILLION = "aud_m"
    AUD_BILLION = "aud_b"
    RATIO = "ratio"
    TIMES = "times"


class MetricSpec(BaseModel):
    """Definition of one metric the pipeline should extract.

    ``query`` is what the retriever searches for; ``aliases`` help the
    rule-based extractor and the LLM recognise alternative names.
    """

    model_config = ConfigDict(frozen=True)

    metric_id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    description: str
    unit: Unit
    query: str
    aliases: tuple[str, ...] = ()
    typical_range: tuple[float, float] | None = Field(
        default=None, description="Sanity bounds used by the validator (inclusive)."
    )


class Chunk(BaseModel):
    """A retrievable slice of a document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    section: str
    text: str
    position: int = Field(ge=0, description="0-based order within the document.")


class Document(BaseModel):
    """A source document (one annual report)."""

    doc_id: str
    title: str
    text: str
    source_path: str | None = None


class ExtractedMetric(BaseModel):
    """One value extracted from a report, with the evidence that supports it.

    This is also the JSON schema handed to Claude as the structured output
    format, so field descriptions double as instructions to the model.
    """

    metric_id: str = Field(description="Identifier of the metric being reported.")
    found: bool = Field(description="False if the metric is not disclosed in the context.")
    value: float | None = Field(
        default=None,
        description="Numeric value exactly as disclosed (e.g. 12.3 for '12.3%'). "
        "Null if not found.",
    )
    unit: Unit | None = Field(default=None, description="Unit the value is expressed in.")
    period: str | None = Field(
        default=None, description="Reporting period or date the value refers to, e.g. 'FY2025'."
    )
    evidence_quote: str | None = Field(
        default=None,
        description="Verbatim sentence from the context containing the value. Must be copied "
        "exactly, not paraphrased.",
    )
    chunk_id: str | None = Field(default=None, description="chunk_id the quote was taken from.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str | None = Field(default=None, description="Caveats, e.g. pro-forma or restated.")

    @field_validator("evidence_quote")
    @classmethod
    def _strip_quote(cls, value: str | None) -> str | None:
        return value.strip() if value else value


class GroundingStatus(StrEnum):
    """Outcome of the post-extraction evidence check."""

    GROUNDED = "grounded"
    QUOTE_NOT_IN_CHUNK = "quote_not_in_chunk"
    VALUE_NOT_IN_QUOTE = "value_not_in_quote"
    OUT_OF_RANGE = "out_of_range"
    NOT_FOUND = "not_found"


class ValidatedMetric(BaseModel):
    """An extracted metric after grounding checks."""

    metric: ExtractedMetric
    grounding: GroundingStatus
    retrieved_chunk_ids: list[str]
    accepted: bool


class ExtractionReport(BaseModel):
    """Full output of a pipeline run over a single document."""

    doc_id: str
    extractor: Literal["rules", "llm", "agent"]
    results: list[ValidatedMetric]

    def accepted(self) -> dict[str, ExtractedMetric]:
        """Return accepted metrics keyed by metric id."""
        return {r.metric.metric_id: r.metric for r in self.results if r.accepted}

    def as_table_rows(self) -> list[dict[str, str]]:
        """Flatten for CLI / markdown rendering."""
        rows: list[dict[str, str]] = []
        for r in self.results:
            m = r.metric
            rows.append(
                {
                    "metric": m.metric_id,
                    "value": "" if m.value is None else f"{m.value:g}",
                    "unit": m.unit.value if m.unit else "",
                    "period": m.period or "",
                    "grounding": r.grounding.value,
                    "accepted": "yes" if r.accepted else "no",
                }
            )
        return rows
