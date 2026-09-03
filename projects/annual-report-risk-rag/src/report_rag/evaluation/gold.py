"""Gold-label schema and loader."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from report_rag.schemas import Unit


class GoldLabel(BaseModel):
    """Ground truth for one metric in one document."""

    metric_id: str
    value: float | None = Field(description="Null means the report does not disclose it.")
    unit: Unit | None = None
    period: str | None = None
    evidence_section: str | None = Field(
        default=None, description="Section heading where the value is disclosed (for recall@k)."
    )


class GoldSet(BaseModel):
    """All gold labels for a document."""

    doc_id: str
    labels: list[GoldLabel]

    def by_id(self) -> dict[str, GoldLabel]:
        """Index labels by metric id."""
        return {label.metric_id: label for label in self.labels}


def load_gold(path: Path) -> GoldSet:
    """Read a gold JSON file."""
    return GoldSet.model_validate(json.loads(path.read_text(encoding="utf-8")))
