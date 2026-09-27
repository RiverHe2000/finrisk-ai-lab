"""Frozen calibration remediation and a separate synthetic temporal assessment."""

from __future__ import annotations

import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ifrs9_pd.config import DataConfig, PipelineConfig
from ifrs9_pd.data.schema import Col
from ifrs9_pd.data.synthetic import generate_portfolio, macro_unemployment_series, split_dev_oot
from ifrs9_pd.model.calibration import calibrate_intercept, fit_factor_loading, ttc_to_pit
from ifrs9_pd.model.ecl import compute_ecl, portfolio_summary
from ifrs9_pd.model.scorecard import PDScorecard, load_model
from ifrs9_pd.pipeline import FittedModel, _ifrs9, evaluate_sample, performing, score_sample
from ifrs9_pd.results import SampleMetrics, frame_to_table, round_floats
from ifrs9_pd.validation.stability import psi

PROTOCOL_VERSION = "synthetic_intercept_remediation_v1"
BASELINE_COMMIT = "e45f526c8f003483c44431844431f92a6db26c58"
ASSESSMENT_DATA = DataConfig(
    n_loans=30_000, seed=20250927, origination_end="2025-12", last_snapshot="2025-12"
)
ASSESSMENT_START = "2025-01"
MACRO_CUTOFF = "2023-12"
LIMITS = {
    "gini_min": 0.50,
    "ks_min": 0.30,
    "hl_p_min": 0.05,
    "binomial_p_min": 0.05,
    "pd_dr_min": 0.85,
    "pd_dr_max": 1.20,
    "score_psi_max": 0.10,
    "gini_drop_max": 0.05,
    "defaults_min": 50.0,
}


def canonical_digest(payload: Any) -> str:
    """Hash JSON independent of indentation, key order and checkout line endings."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    """Hash canonical JSON or LF-normalised text; never rewrite the source file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return canonical_digest(json.loads(text))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    """Hash generated rows, including outcomes, at declared serialisation precision."""
    text = frame.to_csv(
        index=False, float_format="%.12g", date_format="%Y-%m-%d", lineterminator="\n"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FrozenCandidate(BaseModel):
    """Calibration-only receipt sealed before assessment data generation."""

    protocol_version: str
    baseline_commit: str
    source_hashes: dict[str, str]
    protocol_sha256: str
    pipeline_config: PipelineConfig
    assessment_data: DataConfig
    assessment_start: str
    macro_cutoff: str
    macro_mean: float
    macro_std: float
    reference_scorecard: dict[str, Any]
    candidate_scorecard: dict[str, Any]
    added_logit_shift: float
    calibration: dict[str, Any]
    limits: dict[str, float]


def _macro_transform(mean: float, std: float) -> pd.Series:
    return -(macro_unemployment_series() - mean) / std


def _calibration_rows(cfg: PipelineConfig) -> pd.DataFrame:
    _, calibration = split_dev_oot(generate_portfolio(cfg.data), cfg.data.dev_cutoff)
    return calibration


def _receipt(frame: pd.DataFrame) -> dict[str, Any]:
    dates = frame[Col.SNAPSHOT_DATE.value]
    return {
        "n_rows": len(frame),
        "first_snapshot": dates.min().strftime("%Y-%m"),
        "last_snapshot": dates.max().strftime("%Y-%m"),
        "latest_label_maturity": (dates.max() + pd.DateOffset(months=12)).strftime("%Y-%m"),
        "frame_sha256": frame_digest(frame),
        "serialisation": "CSV; LF; floats %.12g; dates YYYY-MM-DD; original column order",
    }


def prepare_remediation(source_dir: Path, protocol: Path, output: Path) -> FrozenCandidate:
    """Fit a single shift using historical calibration rows; never generate the test pool."""
    if output.exists():
        raise FileExistsError(f"Frozen candidate already exists: {output}")
    cfg = PipelineConfig()
    card, binner = load_model(source_dir / "scorecard.json")
    dev, calibration = split_dev_oot(generate_portfolio(cfg.data), cfg.data.dev_cutoff)
    rows = performing(calibration, cfg.staging.default_dpd)
    macro = macro_unemployment_series().loc[: f"{MACRO_CUTOFF}-01"]
    mean, std = float(macro.mean()), float(macro.std())
    baseline = FittedModel(binner, card, _macro_transform(mean, std))
    dev_rows = performing(dev, cfg.staging.default_dpd)
    card.calibration_shift_ = 0.0
    card.factor_loading_ = 1.0
    dev_scored = score_sample(baseline, dev_rows)
    loading, shift = fit_factor_loading(dev_scored.pd_ttc, dev_scored.y, dev_scored.z, baseline.rho)
    card.factor_loading_ = loading
    card.calibration_shift_ = shift
    scored = score_sample(baseline, rows)
    target = float(scored.y.mean())
    delta = calibrate_intercept(
        scored.pd_ttc, target, overlay=lambda p: ttc_to_pit(p, scored.z, baseline.rho)
    )
    candidate = PDScorecard.from_dict(card.to_dict())
    candidate.calibration_shift_ += delta
    fitted = FittedModel(binner, candidate, baseline.macro_z)
    frozen = FrozenCandidate(
        protocol_version=PROTOCOL_VERSION,
        baseline_commit=BASELINE_COMMIT,
        source_hashes={
            name: file_digest(source_dir / name)
            for name in ("scorecard.json", "validation_results.json", "validation_report.md")
        },
        protocol_sha256=file_digest(protocol),
        pipeline_config=cfg,
        assessment_data=ASSESSMENT_DATA,
        assessment_start=ASSESSMENT_START,
        macro_cutoff=MACRO_CUTOFF,
        macro_mean=mean,
        macro_std=std,
        reference_scorecard=card.to_dict(),
        candidate_scorecard=candidate.to_dict(),
        added_logit_shift=delta,
        calibration={
            **_receipt(calibration),
            "performing_rows": len(rows),
            "defaults": int(scored.y.sum()),
            "default_rate": target,
            "baseline_mean_pit_pd": float(scored.pd_pit.mean()),
            "candidate_mean_pit_pd": float(score_sample(fitted, rows).pd_pit.mean()),
        },
        limits=LIMITS,
    )
    payload = frozen.model_dump(mode="json")
    envelope = {"sha256": canonical_digest(payload), "payload": payload}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frozen


def load_frozen(path: Path, source_dir: Path, protocol: Path) -> FrozenCandidate:
    """Check receipt integrity and unchanged original artifacts before any assessment."""
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if canonical_digest(envelope["payload"]) != envelope["sha256"]:
        raise ValueError("Frozen candidate digest mismatch")
    frozen = FrozenCandidate.model_validate(envelope["payload"])
    if (
        frozen.protocol_version != PROTOCOL_VERSION
        or file_digest(protocol) != frozen.protocol_sha256
    ):
        raise ValueError("Frozen protocol mismatch")
    for name, digest in frozen.source_hashes.items():
        if file_digest(source_dir / name) != digest:
            raise ValueError(f"Original artifact changed: {name}")
    return frozen


def generate_assessment(frozen: FrozenCandidate) -> pd.DataFrame:
    """Generate the declared pool once and select its new calendar window without labels."""
    first = pd.Timestamp(f"{frozen.assessment_start}-01")
    maturity = pd.Timestamp(f"{frozen.calibration['latest_label_maturity']}-01")
    if first <= maturity or first <= pd.Timestamp(f"{frozen.macro_cutoff}-01"):
        raise ValueError("Assessment precedes calibration label maturity or macro cutoff")
    pool = generate_portfolio(frozen.assessment_data)
    selected = pool.loc[pool[Col.SNAPSHOT_DATE.value] >= first].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError("Empty assessment window")
    selected[Col.LOAN_ID.value] = (
        f"assessment-{frozen.assessment_data.seed}-" + selected[Col.LOAN_ID.value]
    )
    return selected


def closure_checks(
    baseline: SampleMetrics, candidate: SampleMetrics, score_psi: float, limits: dict[str, float]
) -> dict[str, bool]:
    """Apply the predeclared closure rule without changing limits or selecting models."""
    values = [
        candidate.gini,
        candidate.ks,
        candidate.hl_p_value,
        candidate.binomial_p_value_overall,
        candidate.pd_to_dr_ratio,
        score_psi,
        baseline.gini,
    ]
    return {
        "finite_statistics": all(math.isfinite(v) for v in values),
        "gini": candidate.gini >= limits["gini_min"],
        "ks": candidate.ks >= limits["ks_min"],
        "hl": candidate.hl_p_value >= limits["hl_p_min"],
        "binomial": candidate.binomial_p_value_overall >= limits["binomial_p_min"],
        "pd_dr": limits["pd_dr_min"] <= candidate.pd_to_dr_ratio <= limits["pd_dr_max"],
        "score_psi": score_psi <= limits["score_psi_max"],
        "gini_preserved": baseline.gini - candidate.gini <= limits["gini_drop_max"],
        "enough_defaults": candidate.defaults >= limits["defaults_min"],
    }


def ecl_impact(
    baseline: FittedModel, candidate: FittedModel, book: pd.DataFrame, cfg: PipelineConfig
) -> dict[str, Any]:
    """Measure curve and stage effects on an identical book with identical ECL assumptions."""
    before, _, old_ecl = _ifrs9(baseline, book, cfg)
    after, ts, new_ecl = _ifrs9(candidate, book, cfg)
    fixed_stage = compute_ecl(
        book[Col.EAD.value].to_numpy(dtype=np.float64),
        old_ecl["lgd"].to_numpy(dtype=np.float64),
        ts.marginal,
        cfg.ecl.discount_rate,
        old_ecl["stage"].to_numpy(dtype=np.int64),
        remaining_term=book[Col.REMAINING_TERM.value].to_numpy(dtype=np.int64),
    )
    fixed_total = float(fixed_stage["ecl_applied"].sum())
    transitions = (
        pd.DataFrame({"before": old_ecl["stage"], "after": new_ecl["stage"], "ead": old_ecl["ead"]})
        .groupby(["before", "after"], sort=True)
        .agg(n_loans=("ead", "size"), ead=("ead", "sum"))
        .reset_index()
    )
    return {
        "baseline": before.model_dump(mode="json"),
        "candidate": after.model_dump(mode="json"),
        "candidate_with_baseline_stages": frame_to_table(portfolio_summary(fixed_stage)),
        "pd_curve_effect": fixed_total - before.total_ecl,
        "additional_stage_effect": after.total_ecl - fixed_total,
        "total_change": after.total_ecl - before.total_ecl,
        "stage_migration": frame_to_table(transitions),
    }


def assess_remediation(
    frozen: FrozenCandidate, source_dir: Path, book: pd.DataFrame
) -> dict[str, Any]:
    """Assess a frozen candidate; this function never fits or mutates model parameters."""
    dates = book[Col.SNAPSHOT_DATE.value]
    if (
        book.empty
        or not dates.between(
            f"{frozen.assessment_start}-01", f"{frozen.assessment_data.last_snapshot}-01"
        ).all()
    ):
        raise ValueError("Book contains snapshots outside the frozen assessment window")
    ids = book[Col.LOAN_ID.value]
    if (
        ids.duplicated().any()
        or not ids.str.startswith(f"assessment-{frozen.assessment_data.seed}-").all()
    ):
        raise ValueError("Assessment IDs must be unique and in the frozen seed namespace")
    _, binner = load_model(source_dir / "scorecard.json")
    macro_z = _macro_transform(frozen.macro_mean, frozen.macro_std)
    baseline = FittedModel(binner, PDScorecard.from_dict(frozen.reference_scorecard), macro_z)
    candidate = FittedModel(binner, PDScorecard.from_dict(frozen.candidate_scorecard), macro_z)
    cfg = frozen.pipeline_config
    calibration = _calibration_rows(cfg)
    if frame_digest(calibration) != frozen.calibration["frame_sha256"]:
        raise ValueError("Calibration data receipt mismatch")
    rows = performing(book, cfg.staging.default_dpd)
    seed = frozen.assessment_data.seed
    before = evaluate_sample("assessment_reference", score_sample(baseline, rows), seed)
    after = evaluate_sample("assessment_candidate", score_sample(candidate, rows), seed)
    stability = psi(
        score_sample(baseline, performing(calibration, cfg.staging.default_dpd)).score,
        score_sample(baseline, rows).score,
    )
    checks = closure_checks(before, after, stability, frozen.limits)
    return {
        "protocol_version": frozen.protocol_version,
        "runtime_versions": {
            name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")
        },
        "frozen_payload_sha256": canonical_digest(frozen.model_dump(mode="json")),
        "assessment": _receipt(book),
        "data_kind": "synthetic; new seed and calendar window; same generator family",
        "metrics": {
            "baseline": before.model_dump(mode="json"),
            "candidate": after.model_dump(mode="json"),
        },
        "baseline_score_psi_calibration_to_assessment": stability,
        "limits": frozen.limits,
        "checks": checks,
        "decision": "CLOSED_IN_SIMULATION" if all(checks.values()) else "OPEN",
        "ecl_impact": ecl_impact(baseline, candidate, book, cfg),
        "original_report_rating": "RED (unchanged)",
    }


def render_remediation(result: dict[str, Any]) -> str:
    """Render an evidence-first report without suggesting synthetic results are approvals."""
    before, after = result["metrics"]["baseline"], result["metrics"]["candidate"]
    receipt, impact = result["assessment"], result["ecl_impact"]
    lines = [
        "# Synthetic IFRS 9 remediation assessment v1",
        "",
        f"**Decision: {result['decision']}. Original report: RED, unchanged.**",
        "",
        "One frozen intercept correction, evaluated on one predeclared new synthetic window.",
        "This is not real-data external validation, independent expert review, "
        "or production approval.",
        "",
        f"Frozen payload SHA-256: `{result['frozen_payload_sha256']}`.",
        f"Generated data receipt: `{receipt['frame_sha256']}`.",
        f"Snapshots: {receipt['first_snapshot']} to {receipt['last_snapshot']}; "
        f"simulated outcome maturity through {receipt['latest_label_maturity']}.",
        f"Book: {receipt['n_rows']:,} loans; performing: {after['n']:,}; "
        f"defaults: {after['defaults']}.",
        "",
        "Both arms use macro mean/std frozen on 2016-01 to 2023-12. The reference retains the",
        "original scorecard ranking with a development-only overlay under this repaired transform;",
        "it is not a replay of",
        "the historical report's future-normalised macro path. Candidate parameters were fixed",
        "on the old OOT calibration sample before this new data pool was generated.",
        "",
        "| Metric | Reference | Candidate |",
        "|---|---:|---:|",
    ]
    for key in (
        "mean_pd",
        "default_rate",
        "pd_to_dr_ratio",
        "gini",
        "ks",
        "hl_p_value",
        "binomial_p_value_overall",
        "brier",
    ):
        lines.append(f"| {key} | {before[key]:.6f} | {after[key]:.6f} |")
    lines += ["", "## Frozen closure checks", "", "| Check | Result |", "|---|---|"]
    lines.extend(
        f"| {key} | {'PASS' if passed else 'FAIL'} |" for key, passed in result["checks"].items()
    )
    lines += [
        "",
        f"Baseline-score PSI: {result['baseline_score_psi_calibration_to_assessment']:.6f}.",
        "",
        "## ECL impact on the same book",
        "",
        "Amounts are AUD; assumptions are unchanged and unvalidated.",
        "",
        "| Measure | Reference | Candidate |",
        "|---|---:|---:|",
    ]
    for key in ("total_ead", "total_ecl", "coverage_ratio"):
        lines.append(
            f"| {key} | {impact['baseline'][key]:,.6f} | {impact['candidate'][key]:,.6f} |"
        )
    lines += [
        "",
        f"PD-curve effect with old stages fixed: AUD {impact['pd_curve_effect']:,.2f}.",
        f"Additional stage effect: AUD {impact['additional_stage_effect']:,.2f}.",
        f"Total ECL change: AUD {impact['total_change']:,.2f}.",
        "",
        "| Stage before | Stage after | Loans | EAD |",
        "|---:|---:|---:|---:|",
    ]
    for row in impact["stage_migration"]:
        lines.append(f"| {row['before']} | {row['after']} | {row['n_loans']} | {row['ead']:,.2f} |")
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "Failed checks remain open; do not choose another seed or tune against this window.",
        "The original RED finding is retained. See the frozen protocol for the numerical gates.",
        "JSON includes confidence intervals, grade tests, stage ECL and scenario sensitivities.",
        "LGD 35%, discount 5%, linear EAD, 60-month horizon and subjective scenario weights",
        "are assumptions. Origination PD is reconstructed at matched age. ECL changes measure",
        "conditional model sensitivity, not realised loss improvement or an accounting adjustment.",
        "The synthetic generator re-centres its latent intercept per generated pool; a new seed",
        "does not establish external generalisation. Labels through 2026-12 are simulated.",
    ]
    return "\n".join(lines) + "\n"


def evaluate_remediation(
    frozen_path: Path, source_dir: Path, protocol: Path, output_dir: Path
) -> dict[str, Any]:
    """Verify the sealed candidate, generate the fixed assessment pool, and save its result."""
    frozen = load_frozen(frozen_path, source_dir, protocol)
    result = assess_remediation(frozen, source_dir, generate_assessment(frozen))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assessment.json").write_text(
        json.dumps(round_floats(result, 10), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "assessment.md").write_text(render_remediation(result), encoding="utf-8")
    return result
