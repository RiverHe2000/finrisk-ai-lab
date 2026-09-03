from __future__ import annotations

import pytest

from report_rag.catalogue import RISK_METRICS, get_metric
from report_rag.config import Settings
from report_rag.extraction.llm_extractor import LLMExtractor, build_user_prompt
from report_rag.extraction.rules import RuleBasedExtractor
from report_rag.llm.stub import StubStructuredLLM
from report_rag.pipeline import RiskExtractionPipeline
from report_rag.schemas import Document, GroundingStatus, Unit


def test_rules_pipeline_extracts_capital_ratios(
    southern_cross: Document, settings: Settings
) -> None:
    pipeline = RiskExtractionPipeline(RuleBasedExtractor(), settings)
    report = pipeline.run(southern_cross)
    accepted = report.accepted()
    assert report.extractor == "rules"
    assert accepted["cet1_ratio"].value == 12.4
    assert accepted["tier1_ratio"].value == 14.2
    assert accepted["lcr"].value == 134
    assert accepted["rwa_total"].value == 215640
    assert accepted["stage3_ecl"].value == 1480
    # The baseline grabs "90" from "90 or more days past due"; the range check rejects it.
    npl = next(r for r in report.results if r.metric.metric_id == "npl_ratio")
    assert npl.grounding is GroundingStatus.OUT_OF_RANGE
    assert not npl.accepted


def test_pipeline_metric_subset_and_rows(southern_cross: Document, settings: Settings) -> None:
    pipeline = RiskExtractionPipeline(
        RuleBasedExtractor(), settings, metrics=[get_metric("cet1_ratio")], embedder=None
    )
    report = pipeline.run(southern_cross)
    assert [r.metric.metric_id for r in report.results] == ["cet1_ratio"]
    rows = report.as_table_rows()
    assert rows[0]["value"] == "12.4"
    assert rows[0]["accepted"] == "yes"


def test_llm_pipeline_grounds_stub_output(southern_cross: Document, settings: Settings) -> None:
    spec = get_metric("cet1_ratio")
    _, retriever = RiskExtractionPipeline(RuleBasedExtractor(), settings).build_retriever(
        southern_cross
    )
    hit = retriever.retrieve(spec.query, top_k=1)[0].chunk
    good_quote = next(s for s in hit.text.split(". ") if "12.4%" in s)
    stub = StubStructuredLLM(
        [
            {
                "metric_id": "wrong_id",  # pipeline must overwrite with the spec id
                "found": True,
                "value": 12.4,
                "unit": "percent",
                "period": "FY2025",
                "evidence_quote": good_quote,
                "chunk_id": hit.chunk_id,
                "confidence": 0.95,
            },
            {
                "metric_id": "cet1_ratio",
                "found": True,
                "value": 99.9,
                "unit": "percent",
                "evidence_quote": "fabricated sentence",
                "chunk_id": hit.chunk_id,
                "confidence": 0.95,
            },
        ]
    )
    pipeline = RiskExtractionPipeline(LLMExtractor(stub), settings, metrics=[spec, spec])
    report = pipeline.run(southern_cross)
    first, second = report.results
    assert report.extractor == "llm"
    assert first.accepted
    assert first.metric.metric_id == "cet1_ratio"
    assert first.metric.unit is Unit.PERCENT
    assert second.grounding is GroundingStatus.QUOTE_NOT_IN_CHUNK
    assert "Metric id: cet1_ratio" in stub.calls[0]["user"]
    assert hit.chunk_id in stub.calls[0]["user"]


def test_user_prompt_lists_excerpts_with_ids(sc_chunks: list) -> None:
    prompt = build_user_prompt(get_metric("lcr"), sc_chunks[:2])
    assert prompt.count("<excerpt") == 2
    assert 'chunk_id="southern_cross_bank_fy2025:0000"' in prompt
    assert "Also known as: LCR, Liquidity Coverage Ratio" in prompt


def test_empty_document_raises(settings: Settings) -> None:
    pipeline = RiskExtractionPipeline(RuleBasedExtractor(), settings)
    with pytest.raises(ValueError, match="no chunks"):
        pipeline.run(Document(doc_id="empty", title="Empty", text="# Only a heading\n"))


def test_catalogue_ids_are_unique_and_lookup_errors_are_helpful() -> None:
    ids = [m.metric_id for m in RISK_METRICS]
    assert len(ids) == len(set(ids))
    with pytest.raises(KeyError, match="Known metrics"):
        get_metric("does_not_exist")
