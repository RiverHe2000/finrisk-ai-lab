"""Markdown model-validation report (SR 11-7 / APRA CPG 223 structure).

The report is deterministic: it contains no timestamps, so re-running with
the same seed reproduces it byte-for-byte.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ifrs9_pd.results import SampleMetrics, Table, ValidationResults

_RAG_ICON = {"green": "🟢 GREEN", "amber": "🟠 AMBER", "red": "🔴 RED"}
_MONEY = "{:,.0f}"
_PCT = "{:.2%}"
_F3 = "{:.3f}"
_F4 = "{:.4f}"
_GRADE_COLS = ["grade", "n", "defaults", "predicted_pd", "observed_rate", "p_value"]
_GRADE_FMT = {"predicted_pd": _PCT, "observed_rate": _PCT, "p_value": _F3}


def _fmt(value: Any, spec: str | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # noqa: PLR0124 - nan check without numpy
            return "-"
        return spec.format(value) if spec else _F4.format(value)
    return str(value)


def md_table(
    rows: Table, columns: Sequence[str] | None = None, formats: Mapping[str, str] | None = None
) -> str:
    """Render rows as a GitHub-flavoured Markdown table.

    Args:
        rows: Row dictionaries.
        columns: Column order (default: keys of the first row).
        formats: Optional ``str.format`` spec per column.

    Returns:
        The Markdown table text (a placeholder for no rows).
    """
    if not rows:
        return "_no rows_"
    cols = list(columns) if columns else list(rows[0].keys())
    fmt = formats or {}
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(" --- " for _ in cols) + "|"
    body = ["| " + " | ".join(_fmt(r.get(c), fmt.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def _rag(rating: str) -> str:
    return _RAG_ICON.get(rating, rating)


def _kv_table(rows: Sequence[tuple[str, str]]) -> list[str]:
    return ["| Quantity | Value |", "| --- | --- |", *[f"| {k} | {v} |" for k, v in rows]]


def _dev_oot_table(rows: Sequence[tuple[str, str, str]], ratings: Sequence[str] = ()) -> list[str]:
    if ratings:
        header = ["| Metric | Development | Out-of-time | Rating |", "| --- | --- | --- | --- |"]
        body = [
            f"| {m} | {d} | {o} | {_rag(r)} |" for (m, d, o), r in zip(rows, ratings, strict=True)
        ]
    else:
        header = ["| Metric | Development | Out-of-time |", "| --- | --- | --- |"]
        body = [f"| {m} | {d} | {o} |" for m, d, o in rows]
    return header + body


def _ci(value: float, low: float, high: float, digits: int = 3) -> str:
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _executive_summary(res: ValidationResults) -> list[str]:
    dev, oot = res.samples["dev"], res.samples["oot"]
    tl = res.traffic_lights
    rows = [
        (
            "Gini (95% CI)",
            _ci(dev.gini, dev.gini_ci_low, dev.gini_ci_high),
            _ci(oot.gini, oot.gini_ci_low, oot.gini_ci_high),
        ),
        ("KS", f"{dev.ks:.3f}", f"{oot.ks:.3f}"),
        ("Hosmer-Lemeshow p-value", f"{dev.hl_p_value:.3f}", f"{oot.hl_p_value:.3f}"),
        ("Mean PD / observed DR", f"{dev.pd_to_dr_ratio:.2f}", f"{oot.pd_to_dr_ratio:.2f}"),
        ("Score PSI (dev vs OOT)", "-", f"{res.stability.score_psi:.3f}"),
        ("Gini deterioration", "-", f"{res.backtest['gini_deterioration']:.3f}"),
    ]
    ratings = [
        tl["oot_gini"],
        tl["oot_ks"],
        tl["oot_hl_p_value"],
        tl["oot_pd_to_dr_ratio"],
        tl["score_psi"],
        tl["gini_deterioration"],
    ]
    findings = (
        [f"- [{f.severity.upper()}] {f.message}" for f in res.findings]
        if res.findings
        else ["- No amber or red results; all validation tests passed."]
    )
    ifrs = res.ifrs9
    return [
        "## 1. Executive summary",
        "",
        f"**Overall rating: {_rag(res.overall_rating)}**",
        "",
        *_dev_oot_table(rows, ratings),
        "",
        "**Key findings**",
        "",
        *findings,
        "",
        f"The reporting book holds {ifrs.n_loans:,} loans with EAD of "
        f"{_MONEY.format(ifrs.total_ead)} and probability-weighted ECL of "
        f"{_MONEY.format(ifrs.total_ecl)} (coverage {ifrs.coverage_ratio:.2%}).",
    ]


def _overview(res: ValidationResults) -> list[str]:
    dq, model, ecl = res.data_quality, res.config["model"], res.config["ecl"]
    return [
        "## 2. Model overview and scope",
        "",
        "**Purpose.** 12-month and lifetime probability of default for a retail "
        "residential-mortgage portfolio, used for IFRS 9 / AASB 9 stage allocation and "
        "expected-credit-loss measurement.",
        "",
        f"**Portfolio and data.** Synthetic loan-snapshot panel with {dq.n_total:,} rows observed "
        f"between {dq.first_snapshot} and {dq.last_snapshot}; each row carries a 12-month default "
        f"flag. Development sample: snapshots up to {dq.dev_cutoff}; out-of-time (OOT) sample: "
        "later snapshots. Defaulted loans (90+ days past due) are excluded from scorecard "
        "development and validation but included in the reporting book for staging and ECL.",
        "",
        "**Methodology.**",
        "",
        "1. Weight-of-evidence binning with monotonic merging and a separate missing bin; feature "
        f"selection by information value (IV >= {model['min_iv']}"
        + (f", capped at {model['max_iv']})." if model["max_iv"] is not None else ")."),
        "2. L2-regularised logistic regression on WoE features, scaled to points "
        f"(PDO {model['pdo']:.0f}, {model['base_score']:.0f} points at {model['base_odds']:.0f}:1 "
        "odds).",
        "3. Central-tendency calibration to the development-window default rate, imposed on PIT "
        "PDs, with the factor loading of the macro proxy estimated by maximum likelihood.",
        f"4. Vasicek PIT/TTC conversion with asset correlation rho = {model['asset_correlation']} "
        "and a systematic factor proxied by the standardised unemployment gap.",
        f"5. Lifetime term structure over {ecl['horizon_months']} months with an age-aware "
        "seasoning hump and probability-weighted macro scenarios; SICR staging with relative and "
        "absolute PD tests and a 30-dpd backstop; discounted ECL.",
        "",
        "The validation follows the SR 11-7 pillars of conceptual soundness, ongoing monitoring "
        "and outcomes analysis, and the APRA CPG 223 expectations for independent review of "
        "IFRS 9 provisioning models.",
    ]


def _data_quality(res: ValidationResults) -> list[str]:
    dq = res.data_quality
    missing = [{"feature": k, "missing_rate": v} for k, v in dq.missing_rates.items() if v > 0]
    return [
        "## 3. Data quality",
        "",
        "| Sample | Rows | Performing rows | Default rate |",
        "| --- | --- | --- | --- |",
        f"| Total | {dq.n_total:,} | - | {dq.default_rate_total:.2%} |",
        f"| Development (<= {dq.dev_cutoff}) | {dq.n_dev:,} | {dq.n_dev_performing:,} | "
        f"{dq.default_rate_dev:.2%} |",
        f"| Out-of-time (> {dq.dev_cutoff}) | {dq.n_oot:,} | {dq.n_oot_performing:,} | "
        f"{dq.default_rate_oot:.2%} |",
        "",
        "**Default rate by snapshot year**",
        "",
        md_table(dq.default_rate_by_year, ["year", "n", "default_rate"], {"default_rate": _PCT}),
        "",
        "**Missing values** (features without missing values omitted)",
        "",
        md_table(missing, ["feature", "missing_rate"], {"missing_rate": _PCT}),
    ]


def _model_design(res: ValidationResults) -> list[str]:
    md = res.model_design
    iv_rows = [
        {
            **r,
            "feature": f"**{r['feature']}**"
            if r["feature"] in md.selected_features
            else r["feature"],
        }
        for r in md.iv_table
    ]
    scale_fmt = {"pd_lower": _PCT, "pd_upper": _PCT, "pd_mid": _PCT}
    return [
        "## 4. Model design",
        "",
        "**Information value ranking** (selected features in bold)",
        "",
        md_table(iv_rows, ["feature", "kind", "n_bins", "iv"]),
        "",
        "**Coefficients**",
        "",
        md_table(md.coefficients, ["term", "coefficient"]),
        "",
        f"Calibration shift on the logit scale: {md.calibration_shift:.4f} "
        f"(anchor default rate {md.calibration_target_rate:.2%}); Vasicek asset correlation "
        f"{md.asset_correlation:.2f} with fitted factor loading {md.factor_loading:.3f}.",
        "",
        "**Scorecard** (points per bin)",
        "",
        md_table(
            md.scorecard_table,
            ["feature", "bin", "count", "event_rate", "woe", "points"],
            {"event_rate": _PCT, "points": "{:.1f}"},
        ),
        "",
        "**Master scale** (geometric PD bands)",
        "",
        md_table(md.master_scale, ["grade", "pd_lower", "pd_upper", "pd_mid"], scale_fmt),
    ]


def _discrimination(res: ValidationResults) -> list[str]:
    dev, oot = res.samples["dev"], res.samples["oot"]
    rows = [
        ("AUC", f"{dev.auc:.4f}", f"{oot.auc:.4f}"),
        (
            "Gini (bootstrap 95% CI)",
            _ci(dev.gini, dev.gini_ci_low, dev.gini_ci_high, 4),
            _ci(oot.gini, oot.gini_ci_low, oot.gini_ci_high, 4),
        ),
        (
            "KS (bootstrap 95% CI)",
            _ci(dev.ks, dev.ks_ci_low, dev.ks_ci_high, 4),
            _ci(oot.ks, oot.ks_ci_low, oot.ks_ci_high, 4),
        ),
        ("CAP accuracy ratio", f"{dev.cap_accuracy_ratio:.4f}", f"{oot.cap_accuracy_ratio:.4f}"),
        (
            "Rank ordering monotonic",
            _fmt(dev.rank_ordering_monotonic),
            _fmt(oot.rank_ordering_monotonic),
        ),
    ]
    return [
        "## 5. Discrimination",
        "",
        *_dev_oot_table(rows),
        "",
        f"Gini deterioration from development to OOT: {res.backtest['gini_deterioration']:.4f} "
        f"({_rag(res.traffic_lights['gini_deterioration'])}).",
        "",
        f"![ROC curve]({res.figures.get('roc', 'figures/roc_curve.png')})",
        "",
        "**Rank ordering by grade (out-of-time)**",
        "",
        md_table(
            oot.rank_ordering_table,
            ["grade", "n", "defaults", "default_rate"],
            {"default_rate": _PCT},
        ),
    ]


def _grade_tables(sample: SampleMetrics) -> list[str]:
    return [
        f"**Binomial test by grade ({sample.name})**",
        "",
        md_table(sample.binomial_table, _GRADE_COLS, _GRADE_FMT),
        "",
        f"**Jeffreys test by grade ({sample.name})** - p-value is P(DR <= PD); small values flag "
        "underestimation",
        "",
        md_table(sample.jeffreys_table, _GRADE_COLS, _GRADE_FMT),
    ]


def _calibration(res: ValidationResults) -> list[str]:
    dev, oot = res.samples["dev"], res.samples["oot"]
    rows = [
        ("Mean PIT PD", f"{dev.mean_pd:.4%}", f"{oot.mean_pd:.4%}"),
        ("Observed default rate", f"{dev.default_rate:.4%}", f"{oot.default_rate:.4%}"),
        ("PD / DR ratio", f"{dev.pd_to_dr_ratio:.3f}", f"{oot.pd_to_dr_ratio:.3f}"),
        (
            "Binomial p-value (portfolio)",
            f"{dev.binomial_p_value_overall:.3f}",
            f"{oot.binomial_p_value_overall:.3f}",
        ),
        ("Hosmer-Lemeshow statistic", f"{dev.hl_statistic:.3f}", f"{oot.hl_statistic:.3f}"),
        ("Hosmer-Lemeshow p-value", f"{dev.hl_p_value:.3f}", f"{oot.hl_p_value:.3f}"),
        ("Brier score", f"{dev.brier:.5f}", f"{oot.brier:.5f}"),
    ]
    return [
        "## 6. Calibration",
        "",
        *_dev_oot_table(rows),
        "",
        f"![Calibration]({res.figures.get('calibration', 'figures/calibration.png')})",
        "",
        "**Calibration by PD decile (out-of-time)**",
        "",
        md_table(
            oot.calibration_table,
            ["bin", "n", "defaults", "mean_pd", "observed_rate"],
            {"mean_pd": _PCT, "observed_rate": _PCT},
        ),
        "",
        *_grade_tables(oot),
    ]


def _stability(res: ValidationResults) -> list[str]:
    st = res.stability
    csi_rows = [
        {**r, "rating": _rag(res.traffic_lights[f"csi_{r['feature']}"])} for r in st.csi_table
    ]
    return [
        "## 7. Stability",
        "",
        f"Score PSI (development vs out-of-time): **{st.score_psi:.4f}** "
        f"({_rag(res.traffic_lights['score_psi'])}).",
        "",
        md_table(
            st.score_psi_table,
            ["bin", "expected_share", "actual_share", "contribution"],
            {"expected_share": _PCT, "actual_share": _PCT},
        ),
        "",
        "**Characteristic stability index per feature**",
        "",
        md_table(csi_rows, ["feature", "kind", "csi", "rating"]),
        "",
        f"![CSI]({res.figures.get('csi', 'figures/csi.png')})",
        "",
        "![Score distribution]"
        f"({res.figures.get('score_distribution', 'figures/score_distribution.png')})",
    ]


def _ifrs9(res: ValidationResults) -> list[str]:
    ifrs = res.ifrs9
    pit_rows = [
        ("Loans", f"{ifrs.n_loans:,}"),
        ("Mean systematic factor z at snapshot", f"{ifrs.mean_z_now:.3f}"),
        ("Mean 12m TTC PD", f"{ifrs.mean_pd_ttc:.4%}"),
        ("Mean 12m PIT PD", f"{ifrs.mean_pd_pit:.4%}"),
        ("Mean 12m PIT PD at origination (age-matched)", f"{ifrs.mean_pd_at_origination:.4%}"),
        ("Scenario-weighted 12m PD", f"{ifrs.weighted_pd_12m:.4%}"),
        ("Scenario-weighted lifetime PD", f"{ifrs.weighted_pd_lifetime:.4%}"),
    ]
    term_fmt = {c: _PCT for c in ifrs.term_structure[0] if c != "month"}
    stage_fmt = {"ead": _MONEY, "share_of_loans": _PCT, "share_of_ead": _PCT}
    ecl_fmt = {"ead": _MONEY, "ecl": _MONEY, "coverage": _PCT, "share_of_ead": _PCT}
    scen_cols = [
        "scenario",
        "weight",
        "z_shift",
        "mean_pd_12m",
        "mean_pd_lifetime",
        "ecl",
        "ecl_vs_weighted",
    ]
    scen_fmt = {
        "weight": "{:.2f}",
        "z_shift": "{:.2f}",
        "mean_pd_12m": _PCT,
        "mean_pd_lifetime": _PCT,
        "ecl": _MONEY,
        "ecl_vs_weighted": "{:+.1%}",
    }
    return [
        "## 8. IFRS 9 components",
        "",
        "**PIT / TTC** (reporting book)",
        "",
        *_kv_table(pit_rows),
        "",
        "The gap between the TTC PD and the scenario-weighted 12m PD is the non-linearity effect "
        "of the discrete scenario set; a large gap would indicate that the scenarios do not span "
        "the factor distribution.",
        "",
        f"![Term structure]({res.figures.get('term_structure', 'figures/pd_term_structure.png')})",
        "",
        "**Average cumulative PD by month**",
        "",
        md_table(ifrs.term_structure, None, term_fmt),
        "",
        "**Stage distribution**",
        "",
        md_table(
            ifrs.stage_distribution,
            ["stage", "n_loans", "ead", "share_of_loans", "share_of_ead"],
            stage_fmt,
        ),
        "",
        "**ECL by stage**",
        "",
        md_table(
            ifrs.ecl_by_stage,
            ["stage", "n_loans", "ead", "ecl", "coverage", "share_of_ead"],
            ecl_fmt,
        ),
        "",
        f"![ECL by stage]({res.figures.get('ecl_by_stage', 'figures/ecl_by_stage.png')})",
        "",
        "**Scenario sensitivity**",
        "",
        md_table(ifrs.scenario_sensitivity, scen_cols, scen_fmt),
    ]


def _findings(res: ValidationResults) -> list[str]:
    lines = ["## 9. Findings and recommendations", ""]
    if not res.findings:
        lines.append(
            "No findings: every traffic light is green. Recommendation: proceed with standard "
            "monitoring."
        )
    else:
        lines.append(
            md_table(
                [f.model_dump() for f in res.findings],
                ["severity", "metric", "value", "rating", "message"],
                {"value": _F4},
            )
        )
        lines.extend(
            [
                "",
                "Recommendations: high-severity findings block approval until remediated; medium "
                "findings require a documented action plan before the next monitoring cycle; low "
                "findings are noted for monitoring.",
            ]
        )
    lights = [{"test": k, "rating": _rag(v)} for k, v in res.traffic_lights.items()]
    lines.extend(["", "**Traffic lights**", "", md_table(lights, ["test", "rating"])])
    return lines


def _limitations() -> list[str]:
    return [
        "## 10. Limitations and model risk",
        "",
        "- The data are synthetic: the true PD is a known function of the drivers, so real-world "
        "non-linearities, data-quality issues and definition changes are absent.",
        "- LGD is a flat (or region-level) assumption and EAD amortises linearly; no LGD or EAD "
        "models are validated.",
        "- The systematic factor is a single unemployment-based proxy with an assumed asset "
        "correlation; no macro-econometric model is fitted and scenario shifts are judgemental.",
        "- The origination PD used for SICR is reconstructed by re-scoring with delinquency fields "
        "reset and the current loan age kept, rather than retrieved from an origination archive.",
        "- Lifetime is truncated at the term-structure horizon rather than the full contractual "
        "term.",
        "- The calibration anchor is the development-window average default rate, which is a "
        "short proxy for a through-the-cycle rate.",
    ]


def _appendix(res: ValidationResults) -> list[str]:
    return [
        "## 11. Appendix: run metadata",
        "",
        f"- Package: `ifrs9_pd` version {res.package_version}",
        f"- Generated deterministically with seed {res.seed}; no wall-clock timestamp is embedded.",
        "- Metrics: `validation_results.json`; model artefact: `scorecard.json`.",
        "",
        "**Configuration**",
        "",
        "```json",
        json.dumps(res.config, indent=2, sort_keys=True),
        "```",
    ]


def build_validation_report(results: ValidationResults, output_dir: Path) -> Path:
    """Write ``validation_report.md`` into ``output_dir``.

    Args:
        results: Pipeline results.
        output_dir: Directory that also holds ``figures/``.

    Returns:
        Path to the Markdown report.
    """
    sections = [
        ["# IFRS 9 PD model - independent validation report", ""],
        _executive_summary(results),
        _overview(results),
        _data_quality(results),
        _model_design(results),
        _discrimination(results),
        _calibration(results),
        _stability(results),
        _ifrs9(results),
        _findings(results),
        _limitations(),
        _appendix(results),
    ]
    text = "\n\n".join("\n".join(lines) for lines in sections) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation_report.md"
    path.write_text(text, encoding="utf-8")
    return path
