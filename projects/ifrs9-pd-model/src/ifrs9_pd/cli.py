"""Command-line interface: ``ifrs9-pd``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ifrs9_pd.config import DataConfig, PipelineConfig
from ifrs9_pd.data.schema import Col, validate_portfolio
from ifrs9_pd.data.synthetic import generate_portfolio, macro_unemployment_series, split_dev_oot
from ifrs9_pd.model.calibration import macro_z_score
from ifrs9_pd.model.scorecard import load_model, save_model
from ifrs9_pd.pipeline import FittedModel, fit_model, performing, run_pipeline
from ifrs9_pd.results import ValidationResults

app = typer.Typer(help="IFRS 9 PD model: build, validate and report.", no_args_is_help=True)
console = Console()

_RAG_STYLE = {"green": "bold green", "amber": "bold yellow", "red": "bold red"}
_DATE_COLS = [Col.ORIGINATION_DATE.value, Col.SNAPSHOT_DATE.value]


def _read_portfolio(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=_DATE_COLS)
    return validate_portfolio(df)


def _print_summary(results: ValidationResults) -> None:
    table = Table(title="IFRS 9 PD validation summary")
    table.add_column("Metric")
    table.add_column("Development", justify="right")
    table.add_column("Out-of-time", justify="right")
    table.add_column("Rating")
    dev, oot = results.samples["dev"], results.samples["oot"]
    rows = [
        ("Gini", f"{dev.gini:.3f}", f"{oot.gini:.3f}", results.traffic_lights["oot_gini"]),
        ("KS", f"{dev.ks:.3f}", f"{oot.ks:.3f}", results.traffic_lights["oot_ks"]),
        (
            "HL p-value",
            f"{dev.hl_p_value:.3f}",
            f"{oot.hl_p_value:.3f}",
            results.traffic_lights["oot_hl_p_value"],
        ),
        (
            "PD / DR",
            f"{dev.pd_to_dr_ratio:.2f}",
            f"{oot.pd_to_dr_ratio:.2f}",
            results.traffic_lights["oot_pd_to_dr_ratio"],
        ),
        (
            "Score PSI",
            "-",
            f"{results.stability.score_psi:.3f}",
            results.traffic_lights["score_psi"],
        ),
    ]
    for name, d, o, rating in rows:
        table.add_row(name, d, o, f"[{_RAG_STYLE[rating]}]{rating.upper()}[/]")
    console.print(table)
    ifrs = results.ifrs9
    stages = ", ".join(f"S{r['stage']}: {r['share_of_loans']:.1%}" for r in ifrs.stage_distribution)
    console.print(
        f"Stages ({stages}); ECL {ifrs.total_ecl:,.0f} on EAD {ifrs.total_ead:,.0f} "
        f"(coverage {ifrs.coverage_ratio:.2%})"
    )
    style = _RAG_STYLE[results.overall_rating]
    console.print(f"Overall rating: [{style}]{results.overall_rating.upper()}[/]")


@app.command("generate-data")
def generate_data(
    output: Annotated[Path, typer.Option(help="Destination CSV path.")] = Path("portfolio.csv"),
    n_loans: Annotated[int, typer.Option(help="Number of loan snapshots.")] = 20_000,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    """Generate the synthetic portfolio and write it as CSV."""
    df = generate_portfolio(DataConfig(n_loans=n_loans, seed=seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    console.print(f"Wrote {len(df):,} rows to {output}")


@app.command()
def train(
    data: Annotated[Path, typer.Option(help="Portfolio CSV (see generate-data).")],
    output_dir: Annotated[Path, typer.Option(help="Directory for scorecard.json.")] = Path("model"),
) -> None:
    """Fit the scorecard on the development sample and save it."""
    cfg = PipelineConfig()
    df = _read_portfolio(data)
    dev, _ = split_dev_oot(df, cfg.data.dev_cutoff)
    fitted = fit_model(performing(dev, cfg.staging.default_dpd), cfg.model)
    path = save_model(output_dir / "scorecard.json", fitted.scorecard, fitted.binner)
    console.print(f"Saved scorecard with features {fitted.scorecard.features} to {path}")


@app.command()
def validate(
    data: Annotated[Path, typer.Option(help="Portfolio CSV.")],
    model_dir: Annotated[Path, typer.Option(help="Directory holding scorecard.json.")],
    output_dir: Annotated[Path, typer.Option(help="Directory for the report.")] = Path("reports"),
) -> None:
    """Validate a saved scorecard on a portfolio and write the report."""
    cfg = PipelineConfig()
    scorecard, binner = load_model(model_dir / "scorecard.json")
    fitted = FittedModel(binner, scorecard, macro_z_score(macro_unemployment_series()))
    results = run_pipeline(cfg, output_dir, portfolio=_read_portfolio(data), model=fitted)
    _print_summary(results)
    console.print(f"Report written to {output_dir / 'validation_report.md'}")


@app.command("run-all")
def run_all(
    output_dir: Annotated[Path, typer.Option(help="Directory for all artefacts.")] = Path(
        "reports"
    ),
    n_loans: Annotated[int, typer.Option(help="Number of loan snapshots.")] = 20_000,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    """Generate data, build the model, validate it and write every artefact."""
    cfg = PipelineConfig(data=DataConfig(n_loans=n_loans, seed=seed))
    results = run_pipeline(cfg, output_dir)
    _print_summary(results)
    console.print(f"Artefacts written to {output_dir}")


@app.command("show-scorecard")
def show_scorecard(
    model_dir: Annotated[Path, typer.Option(help="Directory holding scorecard.json.")] = Path(
        "reports"
    ),
) -> None:
    """Print the scorecard points table."""
    scorecard, binner = load_model(model_dir / "scorecard.json")
    table = Table(title="Scorecard")
    for col in ("feature", "bin", "count", "event_rate", "woe", "points"):
        table.add_column(col, justify="right" if col != "feature" else "left")
    for row in scorecard.scorecard_table(binner).itertuples(index=False):
        table.add_row(
            str(row.feature),
            str(row.bin),
            str(row.count),
            f"{row.event_rate:.2%}",
            f"{row.woe:.3f}",
            f"{row.points:.1f}",
        )
    console.print(table)
    console.print(
        f"Intercept {scorecard.intercept_:.4f}, "
        f"calibration shift {scorecard.calibration_shift_:.4f}, "
        f"factor loading {scorecard.factor_loading_:.4f}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
