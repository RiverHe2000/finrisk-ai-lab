"""Claude-backed extractor using structured outputs."""

from __future__ import annotations

from collections.abc import Sequence

from report_rag.llm.base import StructuredLLM
from report_rag.schemas import Chunk, ExtractedMetric, MetricSpec

SYSTEM_PROMPT = """\
You are a prudential-reporting analyst extracting risk metrics from a bank's annual report.

Rules:
1. Use ONLY the context excerpts provided. Never rely on prior knowledge of the bank.
2. If the metric is not disclosed in the context, set found=false and leave value null.
3. `evidence_quote` must be a verbatim sentence copied from exactly one excerpt and must
   contain the number.
4. `chunk_id` must be the id of the excerpt the quote came from.
5. Report the value as printed (12.3 for "12.3%", 1450 for "$1,450 million") and set `unit`
   accordingly.
6. If several periods are disclosed, prefer the most recent full financial year and record it
   in `period`.
7. Prefer Level 2 / consolidated group figures over subsidiary or pro-forma figures; note caveats.
"""


def build_user_prompt(spec: MetricSpec, chunks: Sequence[Chunk]) -> str:
    """Render the metric definition and numbered context excerpts."""
    aliases = ", ".join(spec.aliases) if spec.aliases else "none"
    lines = [
        f"Metric id: {spec.metric_id}",
        f"Metric: {spec.name}",
        f"Definition: {spec.description}",
        f"Expected unit: {spec.unit.value}",
        f"Also known as: {aliases}",
        "",
        "Context excerpts:",
    ]
    for chunk in chunks:
        lines.append(f'<excerpt chunk_id="{chunk.chunk_id}" section="{chunk.section}">')
        lines.append(chunk.text)
        lines.append("</excerpt>")
    lines.append("")
    lines.append("Extract the metric.")
    return "\n".join(lines)


class LLMExtractor:
    """Wraps any :class:`StructuredLLM` with the extraction prompt."""

    name = "llm"

    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    def extract(self, spec: MetricSpec, chunks: Sequence[Chunk]) -> ExtractedMetric:
        """Run one structured-output call for ``spec``."""
        result = self._llm.complete_structured(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(spec, chunks),
            output_model=ExtractedMetric,
        )
        # The model can mistype the id; the spec is authoritative.
        return result.model_copy(update={"metric_id": spec.metric_id})
