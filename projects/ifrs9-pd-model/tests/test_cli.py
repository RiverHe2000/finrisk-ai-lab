import json

from typer.testing import CliRunner

from ifrs9_pd.cli import app

runner = CliRunner()


def test_run_all(tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["run-all", "--output-dir", str(out), "--n-loans", "2000", "--seed", "3"]
    )
    assert result.exit_code == 0, result.output
    assert "Overall rating" in result.output
    for name in ("validation_report.md", "validation_results.json", "scorecard.json"):
        assert (out / name).exists()
    figures = list((out / "figures").glob("*.png"))
    assert len(figures) == 6
    assert all(f.stat().st_size < 150_000 for f in figures)
    payload = json.loads((out / "validation_results.json").read_text())
    assert payload["config"]["data"]["n_loans"] == 2000


def test_generate_train_validate_show(tmp_path):
    data = tmp_path / "portfolio.csv"
    result = runner.invoke(
        app, ["generate-data", "--output", str(data), "--n-loans", "2500", "--seed", "2"]
    )
    assert result.exit_code == 0, result.output
    assert data.exists()
    model_dir = tmp_path / "model"
    result = runner.invoke(app, ["train", "--data", str(data), "--output-dir", str(model_dir)])
    assert result.exit_code == 0, result.output
    assert (model_dir / "scorecard.json").exists()
    out = tmp_path / "reports"
    result = runner.invoke(
        app,
        ["validate", "--data", str(data), "--model-dir", str(model_dir), "--output-dir", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "validation_report.md").exists()
    result = runner.invoke(app, ["show-scorecard", "--model-dir", str(model_dir)])
    assert result.exit_code == 0, result.output
    assert "bureau_score" in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "run-all" in result.output
