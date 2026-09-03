"""Extractor protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from report_rag.schemas import Chunk, ExtractedMetric, MetricSpec


class Extractor(Protocol):
    """Given a metric definition and context chunks, produce an extraction."""

    name: str

    def extract(self, spec: MetricSpec, chunks: Sequence[Chunk]) -> ExtractedMetric:
        """Extract ``spec`` from ``chunks`` (ranked best first)."""
        ...
