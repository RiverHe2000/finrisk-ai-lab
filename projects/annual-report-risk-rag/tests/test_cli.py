from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from report_rag.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _point_at_bundled_data(monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    monkeypatch.setenv("REPORT_RAG_DATA_DIR", str(data_dir))


def test_metrics_command_lists_catalogue() -> None:
    result = runner.invoke(app, ["metrics"])
    assert result.exit_code == 0, result.output
    assert "cet1_ratio" in result.output


def test_extract_rules_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    result = runner.invoke(
        app,
        [
            "extract",
            "--report",
            "southern_cross_bank_fy2025",
            "--metric",
            "cet1_ratio",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload[0]["results"][0]["metric"]["value"] == 12.4


def test_evaluate_rules_writes_markdown(tmp_path: Path) -> None:
    out = tmp_path / "eval.md"
    result = runner.invoke(app, ["evaluate", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert "micro-F1" in result.output
    assert out.read_text().startswith("# Extraction evaluation")


def test_evaluate_agent_is_rejected() -> None:
    result = runner.invoke(app, ["evaluate", "--extractor", "agent"])
    assert result.exit_code != 0
