from __future__ import annotations

from pathlib import Path

import pytest

from report_rag.catalogue import get_metric
from report_rag.config import Settings
from report_rag.ingest import chunk_document, load_document
from report_rag.schemas import Chunk, Document, MetricSpec

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(data_dir=DATA_DIR)


@pytest.fixture(scope="session")
def southern_cross(settings: Settings) -> Document:
    return load_document(settings.reports_dir / "southern_cross_bank_fy2025.md")


@pytest.fixture(scope="session")
def harbour(settings: Settings) -> Document:
    return load_document(settings.reports_dir / "harbour_mutual_fy2025.md")


@pytest.fixture(scope="session")
def sc_chunks(southern_cross: Document, settings: Settings) -> list[Chunk]:
    return chunk_document(southern_cross, settings.chunk_size, settings.chunk_overlap)


@pytest.fixture
def cet1() -> MetricSpec:
    return get_metric("cet1_ratio")


@pytest.fixture
def tiny_doc() -> Document:
    text = (
        "# Tiny Bank FY2025\n\n"
        "## Capital\n"
        "The CET1 ratio was 11.2% at 30 June 2025. Total capital ratio was 16.0%.\n\n"
        "## Liquidity\n"
        "The Liquidity Coverage Ratio (LCR) was 140% for the quarter.\n"
    )
    return Document(doc_id="tiny", title="Tiny Bank FY2025", text=text)
