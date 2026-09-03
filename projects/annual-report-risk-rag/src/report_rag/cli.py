"""``report-rag`` command line interface."""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from report_rag.agent import ExtractionAgent
from report_rag.catalogue import RISK_METRICS, get_metric
from report_rag.config import Settings
from report_rag.evaluation.runner import evaluate_corpus, render_markdown
from report_rag.extraction.base import Extractor
from report_rag.extraction.llm_extractor import LLMExtractor
from report_rag.extraction.rules import RuleBasedExtractor
from report_rag.ingest import load_directory, load_document
from report_rag.pipeline import RiskExtractionPipeline
from report_rag.schemas import Document, ExtractionReport

app = typer.Typer(
    help="Extract prudential risk metrics from annual reports with grounded RAG.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()


class ExtractorChoice(StrEnum):
    """Extractor back-ends selectable from the CLI."""

    RULES = "rules"
    LLM = "llm"
    AGENT = "agent"


def _make_extractor(choice: ExtractorChoice, settings: Settings) -> Extractor:
    if choice is ExtractorChoice.RULES:
        return RuleBasedExtractor()
    from report_rag.llm.anthropic_client import AnthropicLLM  # lazy: needs credentials

    return LLMExtractor(AnthropicLLM(settings))


def _load_docs(settings: Settings, report: str) -> list[Document]:
    if report == "all":
        return load_directory(settings.reports_dir)
    path = Path(report)
    if not path.exists():
        path = settings.reports_dir / f"{report}.md"
    return [load_document(path)]


def _print_report(report: ExtractionReport) -> None:
    columns = ("metric", "value", "unit", "period", "grounding", "accepted")
    table = Table(title=f"{report.doc_id} ({report.extractor})", show_lines=False)
    for col in columns:
        table.add_column(col, justify="right" if col == "value" else "left")
    for row in report.as_table_rows():
        style = "green" if row["accepted"] == "yes" else "yellow"
        table.add_row(*(row[c] for c in columns), style=style)
    console.print(table)


@app.callback()
def _root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING, format="%(message)s")


@app.command()
def metrics() -> None:
    """List the metric catalogue."""
    table = Table(title="Metric catalogue")
    table.add_column("id")
    table.add_column("name")
    table.add_column("unit")
    table.add_column("typical range")
    for m in RISK_METRICS:
        rng = "" if m.typical_range is None else f"{m.typical_range[0]:g} - {m.typical_range[1]:g}"
        table.add_row(m.metric_id, m.name, m.unit.value, rng)
    console.print(table)


@app.command()
def extract(
    report: Annotated[str, typer.Option(help="Path, doc_id under data/reports, or 'all'.")] = "all",
    extractor: Annotated[ExtractorChoice, typer.Option(help="Back-end.")] = ExtractorChoice.RULES,
    metric: Annotated[list[str] | None, typer.Option(help="Restrict to metric ids.")] = None,
    output: Annotated[Path | None, typer.Option(help="Write JSON results here.")] = None,
) -> None:
    """Run the extraction pipeline (or the agent) over one or all reports."""
    settings = Settings()
    specs = [get_metric(m) for m in metric] if metric else list(RISK_METRICS)
    docs = _load_docs(settings, report)
    reports: list[ExtractionReport] = []
    for doc in docs:
        if extractor is ExtractorChoice.AGENT:
            from report_rag.llm.anthropic_client import AnthropicLLM

            agent = ExtractionAgent(AnthropicLLM(settings), settings, metrics=specs)
            result, trace = agent.run(doc)
            console.print(
                f"[dim]{doc.doc_id}: {trace.turns} turns, {len(trace.searches)} searches, "
                f"stopped: {trace.stopped_because}[/dim]"
            )
        else:
            pipeline = RiskExtractionPipeline(
                _make_extractor(extractor, settings), settings, metrics=specs
            )
            result = pipeline.run(doc)
        reports.append(result)
        _print_report(result)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps([r.model_dump(mode="json") for r in reports], indent=2), encoding="utf-8"
        )
        console.print(f"[green]Wrote {output}[/green]")


@app.command()
def evaluate(
    extractor: Annotated[ExtractorChoice, typer.Option(help="Back-end.")] = ExtractorChoice.RULES,
    output: Annotated[Path | None, typer.Option(help="Write a Markdown report here.")] = None,
) -> None:
    """Score an extractor against the gold labels in data/gold."""
    settings = Settings()
    if extractor is ExtractorChoice.AGENT:
        msg = (
            "Agent evaluation is not wired into the corpus runner; use 'extract --extractor agent'."
        )
        raise typer.BadParameter(msg)
    pipeline = RiskExtractionPipeline(_make_extractor(extractor, settings), settings)
    run = evaluate_corpus(pipeline, settings.reports_dir, settings.gold_dir)
    markdown = render_markdown(run)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        console.print(f"[green]Wrote {output}[/green]")
    console.print(f"micro-F1 {run.micro_f1:.3f}  macro-F1 {run.macro_f1:.3f}")
    for s in run.summaries:
        grounding = "n/a" if s.grounding_rate is None else f"{s.grounding_rate:.0%}"
        console.print(
            f"  {s.doc_id}: P {s.precision:.2f} R {s.recall:.2f} F1 {s.f1:.2f} "
            f"grounding {grounding}"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
