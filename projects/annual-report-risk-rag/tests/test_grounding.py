from __future__ import annotations

import pytest

from report_rag.catalogue import get_metric
from report_rag.extraction.grounding import GroundingValidator
from report_rag.schemas import Chunk, ExtractedMetric, GroundingStatus, Unit

CHUNK = Chunk(
    chunk_id="doc:0001",
    doc_id="doc",
    section="Capital management",
    text="The Group's CET1 ratio was 12.4% at 30 June 2025. Total RWA were $215,640 million.",
    position=1,
)
CHUNKS = {CHUNK.chunk_id: CHUNK}


def _metric(**overrides: object) -> ExtractedMetric:
    base = {
        "metric_id": "cet1_ratio",
        "found": True,
        "value": 12.4,
        "unit": Unit.PERCENT,
        "evidence_quote": "The Group's CET1 ratio was 12.4% at 30 June 2025.",
        "chunk_id": "doc:0001",
        "confidence": 0.9,
    }
    base.update(overrides)
    return ExtractedMetric.model_validate(base)


def test_grounded_when_quote_and_value_verify() -> None:
    out = GroundingValidator().validate(get_metric("cet1_ratio"), _metric(), CHUNKS)
    assert out.grounding is GroundingStatus.GROUNDED
    assert out.accepted
    assert out.retrieved_chunk_ids == ["doc:0001"]


def test_not_found_is_not_accepted() -> None:
    out = GroundingValidator().validate(
        get_metric("cet1_ratio"), _metric(found=False, value=None), CHUNKS
    )
    assert out.grounding is GroundingStatus.NOT_FOUND
    assert not out.accepted


def test_quote_must_be_in_cited_chunk() -> None:
    v = GroundingValidator()
    spec = get_metric("cet1_ratio")
    assert (
        v.validate(spec, _metric(evidence_quote="CET1 ratio was 12.4% yesterday"), CHUNKS).grounding
        is GroundingStatus.QUOTE_NOT_IN_CHUNK
    )
    assert (
        v.validate(spec, _metric(chunk_id="doc:9999"), CHUNKS).grounding
        is GroundingStatus.QUOTE_NOT_IN_CHUNK
    )
    assert (
        v.validate(spec, _metric(evidence_quote=None), CHUNKS).grounding
        is GroundingStatus.QUOTE_NOT_IN_CHUNK
    )


def test_quote_match_is_whitespace_insensitive() -> None:
    quote = "The  Group's CET1 ratio\nwas 12.4% at 30 June 2025."
    out = GroundingValidator().validate(
        get_metric("cet1_ratio"), _metric(evidence_quote=quote), CHUNKS
    )
    assert out.grounding is GroundingStatus.GROUNDED


def test_value_must_appear_in_quote() -> None:
    out = GroundingValidator().validate(get_metric("cet1_ratio"), _metric(value=12.5), CHUNKS)
    assert out.grounding is GroundingStatus.VALUE_NOT_IN_QUOTE
    assert not out.accepted


@pytest.mark.parametrize("value", [12.0, 2.4, -12.4])
def test_partial_or_wrong_sign_number_is_not_evidence(value: float) -> None:
    result = GroundingValidator().validate(get_metric("cet1_ratio"), _metric(value=value), CHUNKS)
    assert result.grounding is GroundingStatus.VALUE_NOT_IN_QUOTE
    assert not result.accepted


@pytest.mark.parametrize(
    ("quote", "changes", "status"),
    [
        (
            "CET1 ratio was 12.4% in 2025.",
            {"period": "FY2024"},
            GroundingStatus.PERIOD_NOT_VERIFIED,
        ),
        (
            "CET1 ratio was 12.4% in 2025 (2024: 12.1%).",
            {"period": "FY2025"},
            GroundingStatus.PERIOD_NOT_VERIFIED,
        ),
        ("CET1 ratio was 12.4%.", {"period": "FY2025"}, GroundingStatus.PERIOD_NOT_VERIFIED),
        (
            "CET1 ratio was 12.4% in 1H2025.",
            {"period": "2H2025"},
            GroundingStatus.PERIOD_NOT_VERIFIED,
        ),
        (
            "CET1 ratio was 12.4% in 2025.",
            {"period": "June 2025"},
            GroundingStatus.PERIOD_NOT_VERIFIED,
        ),
        ("CET1 ratio was 12.4 bps.", {}, GroundingStatus.UNIT_NOT_VERIFIED),
        ("CET1 ratio was 12.4%.", {"unit": None}, GroundingStatus.UNIT_NOT_VERIFIED),
        ("Tier 1 capital ratio was 12.4%.", {}, GroundingStatus.METRIC_NOT_VERIFIED),
        (
            "CET1 ratio was 10.2% and Tier 1 capital ratio was 12.4%.",
            {},
            GroundingStatus.METRIC_NOT_VERIFIED,
        ),
        (
            "CET1 ratio and Tier 1 capital ratio were 12.4% and 14% respectively.",
            {},
            GroundingStatus.METRIC_NOT_VERIFIED,
        ),
        ("CET1 ratio was 12.4%.", {"metric_id": "lcr"}, GroundingStatus.METRIC_NOT_VERIFIED),
    ],
)
def test_rejects_unverified_context(quote: str, changes: dict, status: GroundingStatus) -> None:
    chunk = CHUNK.model_copy(update={"text": quote})
    metric = _metric(evidence_quote=quote, **changes)
    result = GroundingValidator().validate(
        get_metric("cet1_ratio"), metric, {chunk.chunk_id: chunk}
    )
    assert result.grounding is status
    assert not result.accepted


@pytest.mark.parametrize("period", [None, "FY2025", "2025"])
def test_single_explicit_annual_period_is_supported(period: str | None) -> None:
    result = GroundingValidator().validate(get_metric("cet1_ratio"), _metric(period=period), CHUNKS)
    assert result.accepted


@pytest.mark.parametrize(
    "quote",
    [
        "CET1 ratio was 10.2%, while return on equity was 12.4%.",
        "CET1 ratio was 10.2%. Return on equity was 12.4%.",
        "CET1 ratio was undisclosed. Return on equity was 12.4%.",
        "CET1 ratio was 10.2%, compared with 12.4% previously.",
        "CET1 ratio was \u221212.4%.",
        "CET1 ratio was - 12.4%.",
    ],
)
def test_unknown_metric_and_unsupported_sign_cannot_supply_evidence(quote: str) -> None:
    chunk = CHUNK.model_copy(update={"text": quote})
    result = GroundingValidator().validate(
        get_metric("cet1_ratio"), _metric(evidence_quote=quote), {chunk.chunk_id: chunk}
    )
    assert not result.accepted


@pytest.mark.parametrize(
    ("quote", "value", "unit", "accepted"),
    [
        ("Loan impairment expense was ($125.4) million.", -125.4, Unit.AUD_MILLION, True),
        ("Loan impairment expense was ($125.4) million.", 125.4, Unit.AUD_MILLION, False),
        ("Loan impairment expense was $125.4 billion.", 125.4, Unit.AUD_MILLION, False),
        ("Loan impairment expense was $125.4 million.", 125.4, Unit.AUD_BILLION, False),
    ],
)
def test_money_sign_and_scale(quote: str, value: float, unit: Unit, accepted: bool) -> None:
    chunk = CHUNK.model_copy(update={"text": quote})
    metric = _metric(
        metric_id="loan_impairment_expense", value=value, unit=unit, evidence_quote=quote
    )
    result = GroundingValidator().validate(
        get_metric("loan_impairment_expense"), metric, {chunk.chunk_id: chunk}
    )
    assert result.accepted is accepted


def test_unit_is_normalised_to_spec_before_range_check() -> None:
    m = _metric(
        metric_id="rwa_total",
        value=215640,
        unit=Unit.AUD_MILLION,
        evidence_quote="Total RWA were $215,640 million.",
    )
    out = GroundingValidator().validate(get_metric("rwa_total"), m, CHUNKS)
    assert out.accepted
    assert out.metric.unit is Unit.AUD_MILLION
    assert out.metric.value == 215640


@pytest.mark.parametrize("value", [90.0, 0.0])
def test_out_of_range_is_flagged_but_kept(value: float) -> None:
    chunk = Chunk(
        chunk_id="c:0000",
        doc_id="c",
        section="s",
        text=f"The non-performing loans ratio was {value:g}% and 90 days past due.",
        position=0,
    )
    m = _metric(
        metric_id="npl_ratio",
        value=value,
        evidence_quote=chunk.text,
        chunk_id="c:0000",
    )
    out = GroundingValidator().validate(get_metric("npl_ratio"), m, {"c:0000": chunk})
    expected = GroundingStatus.OUT_OF_RANGE if value > 15 else GroundingStatus.GROUNDED
    assert out.grounding is expected
    assert out.metric.value == value
