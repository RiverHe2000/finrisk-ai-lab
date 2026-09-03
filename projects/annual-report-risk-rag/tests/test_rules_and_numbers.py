from __future__ import annotations

import pytest

from report_rag.catalogue import get_metric
from report_rag.extraction.numbers import (
    NUMBER_RE,
    find_period,
    number_variants,
    parse_number,
    split_sentences,
    to_spec_unit,
    unit_after,
)
from report_rag.extraction.rules import RuleBasedExtractor
from report_rag.schemas import Chunk, Unit


@pytest.mark.parametrize(
    ("token", "expected"),
    [("1,234.5", 1234.5), ("$12.3", 12.3), ("(45)", -45.0), ("-3", -3.0), ("abc", None)],
)
def test_parse_number(token: str, expected: float | None) -> None:
    assert parse_number(token) == expected


def test_split_sentences_are_verbatim_substrings() -> None:
    text = "CET1 was 12.4%. Tier 1 was 14.2%; Total was 18.9%. (Note 5) applies."
    parts = split_sentences(text)
    assert len(parts) == 4
    assert all(p in text for p in parts)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12.4% of RWA", Unit.PERCENT),
        ("15.8 per cent of", Unit.PERCENT),
        ("612 million for", Unit.AUD_MILLION),
        ("14.2 billion of", Unit.AUD_BILLION),
        ("9 basis points", Unit.BASIS_POINTS),
        ("47 models", None),
    ],
)
def test_unit_after(text: str, expected: Unit | None) -> None:
    match = NUMBER_RE.search(text)
    assert match is not None
    assert unit_after(text, match.end()) == expected


def test_find_period_variants() -> None:
    assert find_period("for the year ended 30 June 2025") == "2025"
    assert find_period("in FY 2024 the") == "FY2024"
    assert find_period("2H2025 results") == "2H2025"
    assert find_period("no period here") is None


def test_number_variants_cover_thousands_and_rounding() -> None:
    v = number_variants(215640.0)
    assert {"215640", "215,640"} <= v
    assert "1,480" in number_variants(1480)
    assert "12" not in number_variants(12.4)  # rounded renderings are not accepted
    assert "12.30" in number_variants(12.3)
    assert "12.4" in number_variants(12.4)


@pytest.mark.parametrize(
    ("value", "unit", "target", "expected"),
    [
        (14.2, Unit.AUD_BILLION, Unit.AUD_MILLION, (14200.0, Unit.AUD_MILLION)),
        (1480, Unit.AUD_MILLION, Unit.AUD_BILLION, (1.48, Unit.AUD_BILLION)),
        (90, Unit.BASIS_POINTS, Unit.PERCENT, (0.9, Unit.PERCENT)),
        (0.9, Unit.PERCENT, Unit.BASIS_POINTS, (90.0, Unit.BASIS_POINTS)),
        (5, None, Unit.PERCENT, (5, Unit.PERCENT)),
        (5, Unit.TIMES, Unit.PERCENT, (5, Unit.TIMES)),
    ],
)
def test_to_spec_unit(value: float, unit: Unit | None, target: Unit, expected: tuple) -> None:
    assert to_spec_unit(value, unit, target) == expected


def _chunk(text: str, section: str = "Capital") -> Chunk:
    return Chunk(chunk_id="x:0000", doc_id="x", section=section, text=text, position=0)


def test_rules_extract_percent_metric_with_quote() -> None:
    chunk = _chunk("Our Level 2 CET1 ratio was 12.4% at 30 June 2025, up from 12.1%.")
    result = RuleBasedExtractor().extract(get_metric("cet1_ratio"), [chunk])
    assert result.found
    assert result.value == 12.4
    assert result.unit is Unit.PERCENT
    assert result.period == "2025"
    assert result.chunk_id == "x:0000"
    assert result.evidence_quote is not None
    assert result.evidence_quote in chunk.text


def test_rules_skip_years_and_require_unit_context() -> None:
    chunk = _chunk("The Tier 1 capital ratio at 2025 year end was 14.2 per cent.")
    result = RuleBasedExtractor().extract(get_metric("tier1_ratio"), [chunk])
    assert result.value == 14.2


def test_rules_handle_billions_for_million_metric() -> None:
    chunk = _chunk("Total risk-weighted assets were $215.6 billion at year end.")
    result = RuleBasedExtractor().extract(get_metric("rwa_total"), [chunk])
    assert result.found
    assert result.value == 215.6
    assert result.unit is Unit.AUD_BILLION  # conversion happens in the validator


def test_rules_return_not_found_when_alias_absent() -> None:
    chunk = _chunk("Nothing relevant here about dividends.")
    result = RuleBasedExtractor().extract(get_metric("lcr"), [chunk])
    assert not result.found
    assert result.value is None
