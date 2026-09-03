from __future__ import annotations

from pathlib import Path

import pytest

from report_rag.config import Settings
from report_rag.evaluation import EvalRun, evaluate_corpus, load_gold, render_markdown, score_report
from report_rag.evaluation.metrics import Outcome
from report_rag.extraction.rules import RuleBasedExtractor
from report_rag.pipeline import RiskExtractionPipeline
from report_rag.schemas import Document


def test_gold_files_cover_the_catalogue(settings: Settings) -> None:
    from report_rag.catalogue import METRICS_BY_ID

    for path in settings.gold_dir.glob("*.json"):
        gold = load_gold(path)
        assert set(gold.by_id()) == set(METRICS_BY_ID), path.name


def test_score_report_confusion_matrix(southern_cross: Document, settings: Settings) -> None:
    pipeline = RiskExtractionPipeline(RuleBasedExtractor(), settings)
    chunks, _ = pipeline.build_retriever(southern_cross)
    report = pipeline.run(southern_cross)
    gold = load_gold(settings.gold_dir / "southern_cross_bank_fy2025.json").by_id()
    summary = score_report(report, gold, {c.chunk_id: c.section for c in chunks})
    by_id = {o.metric_id: o for o in summary.outcomes}
    assert by_id["cet1_ratio"].outcome is Outcome.TP
    assert by_id["npl_ratio"].outcome is Outcome.FN  # rejected by the range check
    assert summary.tp + summary.fp + summary.fn + summary.tn == summary.n_metrics == 12
    assert summary.precision == 1.0
    assert 0.9 <= summary.recall < 1.0
    assert summary.retrieval_recall_at_k == 1.0
    assert summary.grounding_rate is not None
    assert summary.mean_abs_error == 0.0


def test_score_report_handles_null_gold(harbour: Document, settings: Settings) -> None:
    pipeline = RiskExtractionPipeline(RuleBasedExtractor(), settings)
    chunks, _ = pipeline.build_retriever(harbour)
    report = pipeline.run(harbour)
    gold = load_gold(settings.gold_dir / "harbour_mutual_fy2025.json").by_id()
    summary = score_report(report, gold, {c.chunk_id: c.section for c in chunks})
    by_id = {o.metric_id: o for o in summary.outcomes}
    assert by_id["lcr"].outcome is Outcome.TN
    assert by_id["traded_var"].outcome is Outcome.TN
    assert by_id["lcr"].retrieval_hit is None
    assert summary.tn == 4  # leverage, LCR, NSFR and VaR are all undisclosed


def test_evaluate_corpus_and_render(settings: Settings, tmp_path: Path) -> None:
    pipeline = RiskExtractionPipeline(RuleBasedExtractor(), settings)
    run = evaluate_corpus(pipeline, settings.reports_dir, settings.gold_dir)
    assert {s.doc_id for s in run.summaries} == {
        "southern_cross_bank_fy2025",
        "harbour_mutual_fy2025",
    }
    assert 0.8 <= run.micro_f1 <= 1.0
    assert 0.0 <= run.macro_f1 <= 1.0
    md = render_markdown(run)
    assert md.startswith("# Extraction evaluation - extractor `rules`")
    assert "| cet1_ratio | TP |" in md
    (tmp_path / "eval.md").write_text(md, encoding="utf-8")


def test_evaluate_corpus_errors_without_gold(settings: Settings, tmp_path: Path) -> None:
    pipeline = RiskExtractionPipeline(RuleBasedExtractor(), settings)
    with pytest.raises(FileNotFoundError, match="No gold files"):
        evaluate_corpus(pipeline, settings.reports_dir, tmp_path)


def test_eval_run_f1_edge_cases() -> None:
    run = EvalRun(extractor="rules", summaries=[])
    assert run.micro_f1 == 0.0
    assert run.macro_f1 == 0.0
