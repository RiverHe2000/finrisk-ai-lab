"""Evaluation against gold labels: extraction accuracy, retrieval recall, grounding rate."""

from report_rag.evaluation.gold import GoldLabel, GoldSet, load_gold
from report_rag.evaluation.metrics import EvalSummary, MetricOutcome, score_report
from report_rag.evaluation.runner import EvalRun, evaluate_corpus, render_markdown

__all__ = [
    "EvalRun",
    "EvalSummary",
    "GoldLabel",
    "GoldSet",
    "MetricOutcome",
    "evaluate_corpus",
    "load_gold",
    "render_markdown",
    "score_report",
]
