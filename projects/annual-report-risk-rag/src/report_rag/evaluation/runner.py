"""Run the pipeline across a corpus with gold labels and render the results."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from report_rag.evaluation.gold import load_gold
from report_rag.evaluation.metrics import EvalSummary, score_report
from report_rag.ingest import load_document
from report_rag.pipeline import RiskExtractionPipeline


class EvalRun(BaseModel):
    """Corpus-level evaluation output."""

    extractor: str
    summaries: list[EvalSummary]

    @property
    def micro_f1(self) -> float:
        """F1 pooled over every (document, metric) pair."""
        tp = sum(s.tp for s in self.summaries)
        fp = sum(s.fp for s in self.summaries)
        fn = sum(s.fn for s in self.summaries)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def macro_f1(self) -> float:
        """Mean of per-document F1."""
        return sum(s.f1 for s in self.summaries) / len(self.summaries) if self.summaries else 0.0


def evaluate_corpus(pipeline: RiskExtractionPipeline, reports_dir: Path, gold_dir: Path) -> EvalRun:
    """Evaluate every ``<doc_id>.json`` gold file that has a matching report."""
    summaries: list[EvalSummary] = []
    for gold_path in sorted(gold_dir.glob("*.json")):
        gold = load_gold(gold_path)
        report_path = _find_report(reports_dir, gold.doc_id)
        doc = load_document(report_path, doc_id=gold.doc_id)
        chunks, _ = pipeline.build_retriever(doc)
        report = pipeline.run(doc)
        sections = {c.chunk_id: c.section for c in chunks}
        summaries.append(score_report(report, gold.by_id(), sections))
    if not summaries:
        msg = f"No gold files found in {gold_dir}"
        raise FileNotFoundError(msg)
    return EvalRun(extractor=pipeline.extractor.name, summaries=summaries)


def _find_report(reports_dir: Path, doc_id: str) -> Path:
    for suffix in (".md", ".txt", ".pdf"):
        candidate = reports_dir / f"{doc_id}{suffix}"
        if candidate.exists():
            return candidate
    msg = f"No report for doc_id '{doc_id}' in {reports_dir}"
    raise FileNotFoundError(msg)


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if pct else f"{value:.3f}"


def render_markdown(run: EvalRun) -> str:
    """Render an evaluation run as a Markdown report."""
    lines = [
        f"# Extraction evaluation - extractor `{run.extractor}`",
        "",
        f"Micro-F1 **{run.micro_f1:.3f}**, macro-F1 **{run.macro_f1:.3f}** over "
        f"{len(run.summaries)} document(s).",
        "",
        "| Document | Metrics | TP | FP | FN | TN | Precision | Recall | F1 "
        "| Mean rel. error | Retrieval recall@k | Grounding rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in run.summaries:
        lines.append(
            f"| {s.doc_id} | {s.n_metrics} | {s.tp} | {s.fp} | {s.fn} | {s.tn} | "
            f"{_fmt(s.precision)} | {_fmt(s.recall)} | {_fmt(s.f1)} | "
            f"{_fmt(s.mean_rel_error, pct=True)} | {_fmt(s.retrieval_recall_at_k, pct=True)} | "
            f"{_fmt(s.grounding_rate, pct=True)} |"
        )
    for s in run.summaries:
        lines += [
            "",
            f"## {s.doc_id}",
            "",
            "| Metric | Outcome | Gold | Predicted | Grounding | Retrieved gold section |",
            "|---|---|---:|---:|---|---|",
        ]
        for o in s.outcomes:
            gold = "null" if o.gold_value is None else f"{o.gold_value:g}"
            pred = "-" if o.predicted_value is None else f"{o.predicted_value:g}"
            hit = "n/a" if o.retrieval_hit is None else ("yes" if o.retrieval_hit else "no")
            lines.append(
                f"| {o.metric_id} | {o.outcome.value.upper()} | {gold} | {pred} "
                f"| {o.grounding.value} | {hit} |"
            )
    lines.append("")
    return "\n".join(lines)
