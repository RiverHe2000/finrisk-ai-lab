"""Scoring functions.

Definitions (per document, over the metric catalogue):

* **TP** - gold has a value, extraction accepted, value within tolerance.
* **FP** - extraction accepted but gold is null, or value outside tolerance.
* **FN** - gold has a value but nothing was accepted.
* **TN** - gold is null and nothing was accepted.
* **Retrieval recall@k** - share of disclosed metrics whose gold section appears
  in the top-k retrieved chunks.
* **Grounding rate** - share of ``found=true`` extractions whose quote and value
  were verified against the source (the validator's GROUNDED status).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel

from report_rag.evaluation.gold import GoldLabel
from report_rag.schemas import ExtractionReport, GroundingStatus, ValidatedMetric


class Outcome(StrEnum):
    """Confusion-matrix cell for one metric."""

    TP = "tp"
    FP = "fp"
    FN = "fn"
    TN = "tn"


class MetricOutcome(BaseModel):
    """Per-metric comparison against gold."""

    metric_id: str
    outcome: Outcome
    gold_value: float | None
    predicted_value: float | None
    abs_error: float | None
    rel_error: float | None
    grounding: GroundingStatus
    retrieval_hit: bool | None


class EvalSummary(BaseModel):
    """Aggregate scores for one report."""

    doc_id: str
    extractor: str
    n_metrics: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    value_accuracy: float
    mean_abs_error: float | None
    mean_rel_error: float | None
    retrieval_recall_at_k: float | None
    grounding_rate: float | None
    outcomes: list[MetricOutcome]


def within_tolerance(gold: float, pred: float, rel_tol: float, abs_tol: float) -> bool:
    """True if ``pred`` matches ``gold`` within either tolerance."""
    return math.isclose(gold, pred, rel_tol=rel_tol, abs_tol=abs_tol)


def _retrieval_hit(
    result: ValidatedMetric, gold: GoldLabel, sections: Mapping[str, str]
) -> bool | None:
    if gold.value is None or gold.evidence_section is None:
        return None
    wanted = gold.evidence_section.strip().lower()
    return any(
        sections.get(cid, "").strip().lower() == wanted for cid in result.retrieved_chunk_ids
    )


def score_report(
    report: ExtractionReport,
    gold: Mapping[str, GoldLabel],
    chunk_sections: Mapping[str, str],
    rel_tol: float = 0.005,
    abs_tol: float = 0.05,
) -> EvalSummary:
    """Score a report against gold labels.

    Args:
        report: Pipeline output.
        gold: Gold labels keyed by metric id.
        chunk_sections: ``chunk_id -> section`` map for retrieval recall.
        rel_tol: Relative tolerance for value matches (0.5% by default).
        abs_tol: Absolute tolerance (covers rounding such as 12.3 vs 12.34).
    """
    outcomes: list[MetricOutcome] = []
    counts = dict.fromkeys(Outcome, 0)
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    grounded = 0
    found = 0
    hits = 0
    hit_denominator = 0

    for result in report.results:
        label = gold.get(result.metric.metric_id)
        if label is None:
            continue
        pred = result.metric.value if result.accepted else None
        outcome, abs_err, rel_err = _classify(label.value, pred, rel_tol, abs_tol)
        counts[outcome] += 1
        if abs_err is not None:
            abs_errors.append(abs_err)
        if rel_err is not None:
            rel_errors.append(rel_err)
        if result.metric.found:
            found += 1
            grounded += result.grounding is GroundingStatus.GROUNDED
        hit = _retrieval_hit(result, label, chunk_sections)
        if hit is not None:
            hit_denominator += 1
            hits += hit
        outcomes.append(
            MetricOutcome(
                metric_id=result.metric.metric_id,
                outcome=outcome,
                gold_value=label.value,
                predicted_value=pred,
                abs_error=abs_err,
                rel_error=rel_err,
                grounding=result.grounding,
                retrieval_hit=hit,
            )
        )

    tp, fp, fn, tn = (counts[o] for o in (Outcome.TP, Outcome.FP, Outcome.FN, Outcome.TN))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EvalSummary(
        doc_id=report.doc_id,
        extractor=report.extractor,
        n_metrics=len(outcomes),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        value_accuracy=(tp + tn) / len(outcomes) if outcomes else 0.0,
        mean_abs_error=sum(abs_errors) / len(abs_errors) if abs_errors else None,
        mean_rel_error=sum(rel_errors) / len(rel_errors) if rel_errors else None,
        retrieval_recall_at_k=hits / hit_denominator if hit_denominator else None,
        grounding_rate=grounded / found if found else None,
        outcomes=outcomes,
    )


def _classify(
    gold: float | None, pred: float | None, rel_tol: float, abs_tol: float
) -> tuple[Outcome, float | None, float | None]:
    if gold is None:
        return (Outcome.TN if pred is None else Outcome.FP), None, None
    if pred is None:
        return Outcome.FN, None, None
    abs_err = abs(pred - gold)
    rel_err = abs_err / abs(gold) if gold else None
    outcome = Outcome.TP if within_tolerance(gold, pred, rel_tol, abs_tol) else Outcome.FP
    return outcome, abs_err, rel_err
