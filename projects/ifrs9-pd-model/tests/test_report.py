import json

from ifrs9_pd.reporting.report import build_validation_report, md_table
from ifrs9_pd.results import ValidationResults, round_floats

SECTIONS = [
    "## 1. Executive summary",
    "## 2. Model overview and scope",
    "## 3. Data quality",
    "## 4. Model design",
    "## 5. Discrimination",
    "## 6. Calibration",
    "## 7. Stability",
    "## 8. IFRS 9 components",
    "## 9. Findings and recommendations",
    "## 10. Limitations and model risk",
    "## 11. Appendix: run metadata",
]


def test_report_sections_and_figures(results, run_dir):
    text = (run_dir / "validation_report.md").read_text(encoding="utf-8")
    for section in SECTIONS:
        assert section in text
    for figure in results.figures.values():
        assert (run_dir / figure).exists()
        assert f"({figure})" in text
    assert "Overall rating" in text
    assert "seed 1" in text


def test_results_json_loads(results, run_dir):
    path = run_dir / "validation_results.json"
    payload = json.loads(path.read_text())
    assert payload["seed"] == 1
    assert set(payload["samples"]) == {"dev", "oot"}
    loaded = ValidationResults.from_json(path)
    assert loaded.overall_rating == results.overall_rating
    assert loaded.samples["oot"].gini == round(results.samples["oot"].gini, 6)


def test_rebuild_report_elsewhere(results, tmp_path):
    path = build_validation_report(results, tmp_path / "again")
    assert path.exists()
    assert (
        path.read_text(encoding="utf-8")
        == (path.parent.parent / "again" / "validation_report.md").read_text()
    )


def test_md_table_formatting():
    rows = [{"a": 1, "b": 0.12345, "c": "x", "d": True, "e": float("nan"), "f": None}]
    text = md_table(rows, formats={"b": "{:.2f}"})
    assert text.splitlines()[0] == "| a | b | c | d | e | f |"
    assert text.splitlines()[2] == "| 1 | 0.12 | x | yes | - |  |"
    assert md_table([]) == "_no rows_"


def test_round_floats():
    out = round_floats({"a": 1.23456789, "b": [float("inf"), True, 2], "c": {"d": float("nan")}})
    assert out == {"a": 1.234568, "b": [None, True, 2], "c": {"d": None}}
