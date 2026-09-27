"""Temporal and accounting invariants for the synthetic remediation protocol."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from ifrs9_pd import remediation
from ifrs9_pd.cli import app
from ifrs9_pd.data.schema import Col
from ifrs9_pd.data.synthetic import macro_unemployment_series
from ifrs9_pd.model.scorecard import load_model
from ifrs9_pd.pipeline import FittedModel

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "reports"
PROTOCOL = PROJECT / "docs" / "REMEDIATION_PROTOCOL.md"


@pytest.fixture(scope="module")
def receipt(tmp_path_factory):
    path = tmp_path_factory.mktemp("remediation") / "frozen.json"
    frozen = remediation.prepare_remediation(SOURCE, PROTOCOL, path)
    return path, frozen


def test_prepare_uses_past_only_macro_and_does_not_touch_original(receipt):
    path, frozen = receipt
    historical = macro_unemployment_series().loc[:"2023-12-01"]
    assert frozen.macro_mean == pytest.approx(historical.mean())
    assert frozen.macro_std == pytest.approx(historical.std())
    assert frozen.calibration["latest_label_maturity"] == "2024-12"
    assert frozen.calibration["candidate_mean_pit_pd"] == pytest.approx(
        frozen.calibration["default_rate"], abs=1e-10
    )
    original, _ = load_model(SOURCE / "scorecard.json")
    for key in ("features", "coef", "intercept"):
        assert frozen.reference_scorecard[key] == original.to_dict()[key]
        assert frozen.candidate_scorecard[key] == original.to_dict()[key]
    assert (
        frozen.reference_scorecard["factor_loading"] == frozen.candidate_scorecard["factor_loading"]
    )
    assert remediation.load_frozen(path, SOURCE, PROTOCOL) == frozen
    with pytest.raises(FileExistsError):
        remediation.prepare_remediation(SOURCE, PROTOCOL, path)


def test_prepare_never_generates_assessment_pool(monkeypatch, tmp_path):
    generated = []
    real_generate = remediation.generate_portfolio

    def observe(cfg):
        generated.append(cfg.seed)
        return real_generate(cfg)

    monkeypatch.setattr(remediation, "generate_portfolio", observe)
    result = CliRunner().invoke(
        app,
        [
            "remediation-prepare",
            "--source-dir",
            str(SOURCE),
            "--protocol",
            str(PROTOCOL),
            "--output",
            str(tmp_path / "candidate.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert generated == [42]


def test_frozen_receipt_rejects_tampering(receipt, tmp_path):
    path, _ = receipt
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["payload"]["limits"]["pd_dr_max"] = 99
    altered = tmp_path / "tampered.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        remediation.load_frozen(altered, SOURCE, PROTOCOL)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("altered protocol", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol mismatch"):
        remediation.load_frozen(path, SOURCE, protocol)


def test_maturity_check_precedes_generation(receipt, monkeypatch):
    _, frozen = receipt
    overlapping = frozen.model_copy(update={"assessment_start": "2024-12"})
    monkeypatch.setattr(
        remediation, "generate_portfolio", lambda _: pytest.fail("must not generate")
    )
    with pytest.raises(ValueError, match="label maturity"):
        remediation.generate_assessment(overlapping)


def test_assessment_selection_uses_calendar_not_outcomes(receipt, monkeypatch, portfolio):
    _, frozen = receipt
    small = portfolio.iloc[:3].copy()
    small[Col.SNAPSHOT_DATE.value] = pd.to_datetime(["2024-12-01", "2025-01-01", "2025-12-01"])
    monkeypatch.setattr(remediation, "generate_portfolio", lambda _: small)
    selected = remediation.generate_assessment(frozen)
    small[Col.DEFAULT_12M.value] = 1 - small[Col.DEFAULT_12M.value]
    changed = remediation.generate_assessment(frozen)
    assert selected[Col.LOAN_ID.value].tolist() == changed[Col.LOAN_ID.value].tolist()
    assert len(selected) == 2
    assert selected[Col.LOAN_ID.value].str.startswith("assessment-20250927-").all()
    assert remediation.frame_digest(selected) != remediation.frame_digest(changed)


def test_ecl_attribution_reconciles_and_does_not_use_labels(fitted, portfolio, cfg):
    book = portfolio.iloc[:100].copy()
    candidate_card = type(fitted.scorecard).from_dict(fitted.scorecard.to_dict())
    candidate_card.calibration_shift_ += 0.2
    candidate = FittedModel(fitted.binner, candidate_card, fitted.macro_z)
    before = candidate_card.to_dict()
    first = remediation.ecl_impact(fitted, candidate, book, cfg)
    book[Col.DEFAULT_12M.value] = 1 - book[Col.DEFAULT_12M.value]
    second = remediation.ecl_impact(fitted, candidate, book, cfg)
    assert first == second
    assert first["total_change"] == pytest.approx(
        first["pd_curve_effect"] + first["additional_stage_effect"]
    )
    assert sum(row["n_loans"] for row in first["stage_migration"]) == len(book)
    assert first["baseline"]["total_ead"] == first["candidate"]["total_ead"]
    assert candidate_card.to_dict() == before


def test_closure_fails_for_overprediction_and_sparse_sample(results):
    baseline = results.samples["oot"]
    good = baseline.model_copy(
        update={
            "gini": 0.8,
            "ks": 0.5,
            "hl_p_value": 0.5,
            "binomial_p_value_overall": 0.5,
            "pd_to_dr_ratio": 1.0,
            "defaults": 100,
        }
    )
    assert all(remediation.closure_checks(baseline, good, 0.01, remediation.LIMITS).values())
    bad = good.model_copy(update={"pd_to_dr_ratio": 1.21, "defaults": 49, "gini": np.nan})
    checks = remediation.closure_checks(baseline, bad, 0.01, remediation.LIMITS)
    assert not checks["finite_statistics"]
    assert not checks["pd_dr"]
    assert not checks["enough_defaults"]


def test_assessment_rejects_earlier_book(receipt, portfolio):
    _, frozen = receipt
    with pytest.raises(ValueError, match="outside"):
        remediation.assess_remediation(frozen, SOURCE, portfolio)


def test_assessment_labels_never_change_candidate_or_ecl(receipt, portfolio, monkeypatch):
    _, frozen = receipt
    payload_before = frozen.model_dump(mode="json")
    book = portfolio.iloc[:1000].copy()
    book[Col.SNAPSHOT_DATE.value] = pd.Timestamp("2025-01-01")
    book[Col.LOAN_ID.value] = "assessment-20250927-" + book[Col.LOAN_ID.value]
    monkeypatch.setattr(
        remediation, "calibrate_intercept", lambda *a, **k: pytest.fail("must not fit")
    )
    monkeypatch.setattr(
        remediation, "fit_factor_loading", lambda *a, **k: pytest.fail("must not fit")
    )
    first = remediation.assess_remediation(frozen, SOURCE, book)
    book[Col.DEFAULT_12M.value] = 1 - book[Col.DEFAULT_12M.value]
    second = remediation.assess_remediation(frozen, SOURCE, book)
    assert frozen.model_dump(mode="json") == payload_before
    assert first["ecl_impact"] == second["ecl_impact"]
    assert first["frozen_payload_sha256"] == second["frozen_payload_sha256"]
    assert first["metrics"]["candidate"]["mean_pd"] == second["metrics"]["candidate"]["mean_pd"]
    assert first["metrics"]["candidate"]["defaults"] != second["metrics"]["candidate"]["defaults"]


def test_changed_source_and_duplicate_ids_are_rejected(receipt, tmp_path, portfolio):
    path, frozen = receipt
    for filename in frozen.source_hashes:
        (tmp_path / filename).write_text(
            (SOURCE / filename).read_text(encoding="utf-8"), encoding="utf-8"
        )
    (tmp_path / "scorecard.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Original artifact changed"):
        remediation.load_frozen(path, tmp_path, PROTOCOL)
    book = portfolio.iloc[:2].copy()
    book[Col.SNAPSHOT_DATE.value] = pd.Timestamp("2025-01-01")
    book[Col.LOAN_ID.value] = "assessment-20250927-duplicate"
    with pytest.raises(ValueError, match="IDs must be unique"):
        remediation.assess_remediation(frozen, SOURCE, book)


def test_evaluate_cli_writes_report_without_altering_frozen(
    receipt, tmp_path, portfolio, monkeypatch
):
    path, frozen = receipt
    original_bytes = path.read_bytes()
    book = portfolio.iloc[:1000].copy()
    book[Col.SNAPSHOT_DATE.value] = pd.Timestamp("2025-01-01")
    book[Col.LOAN_ID.value] = "assessment-20250927-" + book[Col.LOAN_ID.value]
    monkeypatch.setattr(remediation, "generate_assessment", lambda _: book)
    result = CliRunner().invoke(
        app,
        [
            "remediation-evaluate",
            "--source-dir",
            str(SOURCE),
            "--protocol",
            str(PROTOCOL),
            "--frozen",
            str(path),
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / "assessment.json").read_text(encoding="utf-8"))
    assert saved["assessment"]["n_rows"] == 1000
    assert "Original report: RED, unchanged" in (tmp_path / "assessment.md").read_text(
        encoding="utf-8"
    )
    assert path.read_bytes() == original_bytes
    assert saved["frozen_payload_sha256"] == remediation.canonical_digest(
        frozen.model_dump(mode="json")
    )
