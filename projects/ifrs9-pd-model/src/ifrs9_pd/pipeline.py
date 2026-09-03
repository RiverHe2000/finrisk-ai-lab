"""End-to-end pipeline: data, model, IFRS 9 components, validation, report.

Simplifications worth knowing about (all documented in the report):

* The long-run default rate used as the calibration anchor is the average
  default rate of the development window, and the anchor is imposed on the
  PIT PDs (``mean_i PIT(PD_ttc_i, lambda z_i) = DR_dev``) so that the PD
  compared with observed defaults is the one that is calibrated. The factor
  loading ``lambda`` is estimated jointly by maximum likelihood.
* The origination PD used for the SICR test is obtained by re-scoring each
  loan with its delinquency fields reset and the macro factor at origination
  while keeping its current age (see
  :func:`ifrs9_pd.data.synthetic.origination_view`); a production system
  would store the PD assigned at origination.
* The reporting book for staging and ECL is the out-of-time sample including
  defaulted loans; the scorecard itself is built on performing loans only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ifrs9_pd import __version__
from ifrs9_pd.config import ModelConfig, PipelineConfig
from ifrs9_pd.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, Col, validate_portfolio
from ifrs9_pd.data.synthetic import (
    generate_portfolio,
    macro_unemployment_series,
    origination_view,
    split_dev_oot,
)
from ifrs9_pd.features.binning import WoEBinner, select_features
from ifrs9_pd.model.calibration import fit_factor_loading, macro_z_score, ttc_to_pit
from ifrs9_pd.model.ecl import compute_ecl, portfolio_summary
from ifrs9_pd.model.scorecard import PDScorecard, save_model
from ifrs9_pd.model.staging import assign_stage
from ifrs9_pd.model.term_structure import TermStructure, build_term_structure
from ifrs9_pd.reporting import figures
from ifrs9_pd.reporting.report import build_validation_report
from ifrs9_pd.results import (
    DataQuality,
    Finding,
    IFRS9Results,
    ModelDesign,
    SampleMetrics,
    StabilityResults,
    ValidationResults,
    frame_to_table,
)
from ifrs9_pd.validation.backtest import ScoredSample, out_of_time_backtest
from ifrs9_pd.validation.calibration import (
    binomial_test_by_grade,
    brier_score,
    expected_vs_observed_overall,
    hosmer_lemeshow,
    jeffreys_test,
)
from ifrs9_pd.validation.discrimination import (
    auc,
    bootstrap_ci,
    cap_curve,
    gini,
    ks_statistic,
    rank_ordering,
)
from ifrs9_pd.validation.stability import csi_by_feature, psi_table
from ifrs9_pd.validation.thresholds import Rating, traffic_light

TARGET = Col.DEFAULT_12M.value
CANDIDATE_NUMERIC = [c.value for c in NUMERIC_FEATURES]
CANDIDATE_CATEGORICAL = [c.value for c in CATEGORICAL_FEATURES]
N_BOOTSTRAP = 300


@dataclass(frozen=True)
class FittedModel:
    """Fitted binner and scorecard together with the macro factor series.

    Attributes:
        binner: Fitted WoE binner.
        scorecard: Fitted, calibrated scorecard (carries the factor loading).
        macro_z: Standardised systematic factor proxy by month.
    """

    binner: WoEBinner
    scorecard: PDScorecard
    macro_z: pd.Series

    @property
    def rho(self) -> float:
        """Asset correlation used for PIT conversion."""
        return self.scorecard.cfg.asset_correlation

    def factor_at(self, dates: pd.Series) -> NDArray[np.float64]:
        """Scaled systematic factor ``lambda z`` for each date (zero outside the series)."""
        return self.scorecard.factor_loading_ * factor_at(self.macro_z, dates)


@dataclass(frozen=True)
class Scored:
    """Model outputs for one sample."""

    y: NDArray[np.int64]
    pd_ttc: NDArray[np.float64]
    pd_pit: NDArray[np.float64]
    score: NDArray[np.float64]
    grade: NDArray[np.int64]
    z: NDArray[np.float64]


def performing(df: pd.DataFrame, default_dpd: int) -> pd.DataFrame:
    """Rows that are not yet in default (``current_dpd < default_dpd``)."""
    return df.loc[df[Col.CURRENT_DPD.value] < default_dpd].reset_index(drop=True)


def factor_at(macro_z: pd.Series, dates: pd.Series) -> NDArray[np.float64]:
    """Look up the systematic factor for each date (zero when outside the series)."""
    return macro_z.reindex(pd.DatetimeIndex(dates)).fillna(0.0).to_numpy(dtype=np.float64)


def fit_model(dev: pd.DataFrame, cfg: ModelConfig, macro_z: pd.Series | None = None) -> FittedModel:
    """Fit binning, feature selection, scorecard and calibration on the development sample.

    Args:
        dev: Performing development rows with the target column.
        cfg: Model configuration.
        macro_z: Factor series; defaults to the synthetic macro path.

    Returns:
        The fitted model.
    """
    z_series = macro_z if macro_z is not None else macro_z_score(macro_unemployment_series())
    binner = WoEBinner(CANDIDATE_NUMERIC, CANDIDATE_CATEGORICAL, cfg.n_bins, cfg.min_bin_share)
    binner.fit(dev, TARGET)
    features = select_features(binner.iv_table(), cfg.min_iv, cfg.max_iv)
    scorecard = PDScorecard(cfg, features).fit(binner.transform(dev, features), dev[TARGET])
    z_dev = factor_at(z_series, dev[Col.SNAPSHOT_DATE.value])
    raw_pd = scorecard.predict_proba(binner.transform(dev, features))
    loading, shift = fit_factor_loading(
        raw_pd, dev[TARGET].to_numpy(dtype=np.int64), z_dev, cfg.asset_correlation
    )
    scorecard.factor_loading_ = loading
    scorecard.calibration_shift_ = shift
    return FittedModel(binner, scorecard, z_series)


def score_sample(model: FittedModel, df: pd.DataFrame) -> Scored:
    """Score rows: TTC PD, PIT PD via the factor at snapshot, points and grade."""
    x = model.binner.transform(df, model.scorecard.features)
    pd_ttc = model.scorecard.predict_proba(x)
    z = model.factor_at(df[Col.SNAPSHOT_DATE.value])
    pd_pit = ttc_to_pit(pd_ttc, z, model.rho)
    return Scored(
        y=df[TARGET].to_numpy(dtype=np.int64),
        pd_ttc=pd_ttc,
        pd_pit=pd_pit,
        score=model.scorecard.to_points(pd_ttc),
        grade=model.scorecard.assign_grade(pd_pit),
        z=z,
    )


def evaluate_sample(name: str, s: Scored, seed: int) -> SampleMetrics:
    """Compute discrimination and calibration statistics for one scored sample."""
    hl_stat, hl_p, cal_table = hosmer_lemeshow(s.y, s.pd_pit)
    rank_table, monotonic = rank_ordering(s.y, s.grade)
    overall = expected_vs_observed_overall(s.y, s.pd_pit)
    gini_ci = bootstrap_ci(gini, s.y, s.pd_pit, n_boot=N_BOOTSTRAP, seed=seed)
    ks_ci = bootstrap_ci(ks_statistic, s.y, s.pd_pit, n_boot=N_BOOTSTRAP, seed=seed + 1)
    _, accuracy_ratio = cap_curve(s.y, s.pd_pit)
    return SampleMetrics(
        name=name,
        n=len(s.y),
        defaults=int(s.y.sum()),
        default_rate=float(s.y.mean()),
        mean_pd=float(s.pd_pit.mean()),
        auc=auc(s.y, s.pd_pit),
        gini=gini(s.y, s.pd_pit),
        gini_ci_low=gini_ci[0],
        gini_ci_high=gini_ci[1],
        ks=ks_statistic(s.y, s.pd_pit),
        ks_ci_low=ks_ci[0],
        ks_ci_high=ks_ci[1],
        brier=brier_score(s.y, s.pd_pit),
        hl_statistic=hl_stat,
        hl_p_value=hl_p,
        pd_to_dr_ratio=overall["pd_to_dr_ratio"],
        binomial_p_value_overall=overall["p_value"],
        rank_ordering_monotonic=monotonic,
        rank_ordering_table=frame_to_table(rank_table),
        calibration_table=frame_to_table(cal_table),
        binomial_table=frame_to_table(binomial_test_by_grade(s.y, s.pd_pit, s.grade)),
        jeffreys_table=frame_to_table(jeffreys_test(s.y, s.pd_pit, s.grade)),
        cap_accuracy_ratio=accuracy_ratio,
    )


def _data_quality(
    df: pd.DataFrame,
    dev_all: pd.DataFrame,
    oot_all: pd.DataFrame,
    *,
    n_dev: int,
    n_oot: int,
    cutoff: str,
) -> DataQuality:
    by_year = (
        df.groupby(df[Col.SNAPSHOT_DATE.value].dt.year)[TARGET]
        .agg(n="size", default_rate="mean")
        .rename_axis("year")
        .reset_index()
    )
    return DataQuality(
        n_total=len(df),
        n_dev=len(dev_all),
        n_oot=len(oot_all),
        n_dev_performing=n_dev,
        n_oot_performing=n_oot,
        default_rate_total=float(df[TARGET].mean()),
        default_rate_dev=float(dev_all[TARGET].mean()),
        default_rate_oot=float(oot_all[TARGET].mean()),
        missing_rates={
            c: float(df[c].isna().mean()) for c in CANDIDATE_NUMERIC + CANDIDATE_CATEGORICAL
        },
        first_snapshot=str(df[Col.SNAPSHOT_DATE.value].min().strftime("%Y-%m")),
        last_snapshot=str(df[Col.SNAPSHOT_DATE.value].max().strftime("%Y-%m")),
        dev_cutoff=cutoff,
        default_rate_by_year=frame_to_table(by_year),
    )


def _model_design(model: FittedModel, target_rate: float) -> ModelDesign:
    card = model.scorecard
    return ModelDesign(
        candidate_features=model.binner.features,
        selected_features=card.features,
        iv_table=frame_to_table(model.binner.iv_table()),
        binning_tables={f: frame_to_table(model.binner.binning_table(f)) for f in card.features},
        coefficients=frame_to_table(card.coefficient_table()),
        scorecard_table=frame_to_table(card.scorecard_table(model.binner)),
        master_scale=frame_to_table(card.master_scale.table()),
        calibration_shift=card.calibration_shift_,
        calibration_target_rate=target_rate,
        asset_correlation=model.rho,
        factor_loading=card.factor_loading_,
    )


def _stability(
    dev: pd.DataFrame, oot: pd.DataFrame, s_dev: Scored, s_oot: Scored
) -> StabilityResults:
    table = psi_table(s_dev.score, s_oot.score)
    csi = csi_by_feature(dev, oot, CANDIDATE_NUMERIC, CANDIDATE_CATEGORICAL)
    return StabilityResults(
        score_psi=float(table["contribution"].sum()),
        score_psi_table=frame_to_table(table),
        csi_table=frame_to_table(csi),
    )


def _lgd_vector(book: pd.DataFrame, cfg: PipelineConfig) -> NDArray[np.float64]:
    region = book[Col.REGION.value].astype("str")
    return region.map(lambda r: cfg.ecl.lgd_by_segment.get(r, cfg.ecl.lgd)).to_numpy(
        dtype=np.float64
    )


def _scenario_sensitivity(
    ts: TermStructure,
    book: pd.DataFrame,
    lgd: NDArray[np.float64],
    stage: NDArray[np.int64],
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ead = book[Col.EAD.value].to_numpy(dtype=np.float64)
    term = book[Col.REMAINING_TERM.value].to_numpy(dtype=np.int64)
    rate = cfg.ecl.discount_rate
    weighted = compute_ecl(ead, lgd, ts.marginal, rate, stage, remaining_term=term)
    base_ecl = float(weighted["ecl_applied"].sum())
    rows: list[dict[str, object]] = []
    for name, curves in ts.by_scenario.items():
        ecl = compute_ecl(ead, lgd, curves.marginal, rate, stage, remaining_term=term)
        total = float(ecl["ecl_applied"].sum())
        rows.append(
            {
                "scenario": name,
                "weight": cfg.scenarios.weights[name],
                "z_shift": cfg.scenarios.z_shifts[name],
                "mean_pd_12m": float(curves.cumulative[:, 11].mean()),
                "mean_pd_lifetime": float(curves.cumulative[:, -1].mean()),
                "ecl": total,
                "ecl_vs_weighted": total / base_ecl - 1.0,
            }
        )
    rows.append(
        {
            "scenario": "weighted",
            "weight": 1.0,
            "z_shift": float("nan"),
            "mean_pd_12m": float(ts.cumulative[:, 11].mean()),
            "mean_pd_lifetime": float(ts.cumulative[:, -1].mean()),
            "ecl": base_ecl,
            "ecl_vs_weighted": 0.0,
        }
    )
    return weighted, pd.DataFrame(rows)


def _ifrs9(
    model: FittedModel, book: pd.DataFrame, cfg: PipelineConfig
) -> tuple[IFRS9Results, TermStructure, pd.DataFrame]:
    now = score_sample(model, book)
    orig_x = model.binner.transform(origination_view(book), model.scorecard.features)
    z_orig = model.factor_at(book[Col.ORIGINATION_DATE.value])
    pd_orig = ttc_to_pit(model.scorecard.predict_proba(orig_x), z_orig, model.rho)
    stage = assign_stage(book, now.pd_pit, pd_orig, cfg.staging)
    ts = build_term_structure(
        now.pd_ttc,
        cfg.scenarios,
        z0=now.z,
        rho=model.rho,
        horizon=cfg.ecl.horizon_months,
        peak_month=cfg.ecl.peak_month,
        peak_multiplier=cfg.ecl.peak_multiplier,
        mean_reversion=cfg.ecl.mean_reversion,
        age=book[Col.MONTHS_ON_BOOK.value].to_numpy(dtype=np.int64),
    )
    ecl, sensitivity = _scenario_sensitivity(ts, book, _lgd_vector(book, cfg), stage, cfg)
    summary = portfolio_summary(ecl)
    stage_dist = (
        pd.DataFrame({"stage": stage, "ead": ecl["ead"]})
        .groupby("stage", sort=True)
        .agg(n_loans=("ead", "size"), ead=("ead", "sum"))
        .reset_index()
    )
    stage_dist["share_of_loans"] = stage_dist["n_loans"] / len(book)
    stage_dist["share_of_ead"] = stage_dist["ead"] / stage_dist["ead"].sum()
    curves = {name: c.cumulative.mean(axis=0) for name, c in ts.by_scenario.items()}
    curves["weighted"] = ts.cumulative.mean(axis=0)
    term_table = pd.DataFrame({"month": np.arange(1, cfg.ecl.horizon_months + 1), **curves})
    total = summary.loc[summary["stage"] == "total"].iloc[0]
    results = IFRS9Results(
        n_loans=len(book),
        mean_z_now=float(now.z.mean()),
        mean_pd_ttc=float(now.pd_ttc.mean()),
        mean_pd_pit=float(now.pd_pit.mean()),
        mean_pd_at_origination=float(pd_orig.mean()),
        weighted_pd_12m=float(ts.cumulative[:, 11].mean()),
        weighted_pd_lifetime=float(ts.cumulative[:, -1].mean()),
        term_structure=frame_to_table(
            term_table.iloc[[0, 5, 11, 23, 35, 47, cfg.ecl.horizon_months - 1]]
        ),
        stage_distribution=frame_to_table(stage_dist),
        ecl_by_stage=frame_to_table(summary),
        scenario_sensitivity=frame_to_table(sensitivity),
        total_ead=float(total["ead"]),
        total_ecl=float(total["ecl"]),
        coverage_ratio=float(total["coverage"]),
    )
    return results, ts, ecl


def _traffic_lights(
    samples: dict[str, SampleMetrics],
    backtest: dict[str, float],
    stability: StabilityResults,
    cfg: PipelineConfig,
) -> dict[str, Rating]:
    th = cfg.thresholds
    lights: dict[str, Rating] = {}
    for name, m in samples.items():
        lights[f"{name}_gini"] = traffic_light(m.gini, th.gini)
        lights[f"{name}_ks"] = traffic_light(m.ks, th.ks)
        lights[f"{name}_hl_p_value"] = traffic_light(m.hl_p_value, th.hl_p_value)
        lights[f"{name}_binomial_p_value"] = traffic_light(
            m.binomial_p_value_overall, th.binomial_p_value
        )
        lights[f"{name}_pd_to_dr_ratio"] = traffic_light(m.pd_to_dr_ratio, th.pd_to_dr_ratio)
        lights[f"{name}_rank_ordering"] = "green" if m.rank_ordering_monotonic else "amber"
    lights["gini_deterioration"] = traffic_light(
        backtest["gini_deterioration"], th.gini_deterioration
    )
    lights["score_psi"] = traffic_light(stability.score_psi, th.psi)
    for row in stability.csi_table:
        lights[f"csi_{row['feature']}"] = traffic_light(float(row["csi"]), th.psi)
    return lights


_CORE = ("gini", "ks", "hl_p_value", "rank_ordering", "score_psi", "gini_deterioration")
_GLOBAL = ("gini_deterioration", "score_psi")
_MESSAGES: dict[str, str] = {
    "gini": "Gini of {value:.3f} on the {sample} sample is below the {rating} boundary.",
    "ks": "KS statistic of {value:.3f} on the {sample} sample is below the {rating} boundary.",
    "hl_p_value": (
        "Hosmer-Lemeshow p-value of {value:.3f} on the {sample} sample indicates "
        "miscalibration across PD deciles."
    ),
    "binomial_p_value": (
        "Portfolio-level binomial test p-value of {value:.3f} on the {sample} sample: "
        "predicted and observed default rates differ significantly."
    ),
    "pd_to_dr_ratio": (
        "Mean PD is {value:.2f}x the observed default rate on the {sample} sample "
        "(PD underestimates realised defaults)."
    ),
    "rank_ordering": (
        "Observed default rates are not monotonic across grades on the {sample} sample."
    ),
    "gini_deterioration": "Gini fell by {value:.3f} from development to out-of-time.",
    "score_psi": (
        "Score PSI of {value:.3f} indicates a population shift between development and out-of-time."
    ),
    "csi": (
        "CSI of {value:.3f} for feature '{feature}' indicates a distribution shift; monitor "
        "and consider re-binning."
    ),
}


def _finding_value(
    metric: str,
    samples: dict[str, SampleMetrics],
    backtest: dict[str, float],
    stability: StabilityResults,
) -> float:
    if metric.startswith("csi_"):
        feature = metric.removeprefix("csi_")
        return next(float(r["csi"]) for r in stability.csi_table if r["feature"] == feature)
    if metric == "gini_deterioration":
        return backtest[metric]
    if metric == "score_psi":
        return stability.score_psi
    sample, _, key = metric.partition("_")
    m = samples[sample]
    return {
        "gini": m.gini,
        "ks": m.ks,
        "hl_p_value": m.hl_p_value,
        "binomial_p_value": m.binomial_p_value_overall,
        "pd_to_dr_ratio": m.pd_to_dr_ratio,
        "rank_ordering": float(m.rank_ordering_monotonic),
    }[key]


def _findings(
    lights: dict[str, Rating],
    samples: dict[str, SampleMetrics],
    backtest: dict[str, float],
    stability: StabilityResults,
) -> tuple[list[Finding], Rating]:
    findings: list[Finding] = []
    for metric, rating in lights.items():
        if rating == "green":
            continue
        value = _finding_value(metric, samples, backtest, stability)
        if metric.startswith("csi_"):
            key, sample, feature = "csi", "", metric.removeprefix("csi_")
        elif metric in _GLOBAL:
            key, sample, feature = metric, "", ""
        else:
            sample, _, key = metric.partition("_")
            feature = ""
        core = key in _CORE
        severity = (
            "high"
            if (rating == "red" and core)
            else ("medium" if (rating == "red" or core) else "low")
        )
        message = _MESSAGES[key].format(value=value, sample=sample, rating=rating, feature=feature)
        findings.append(
            Finding(metric=metric, value=value, rating=rating, severity=severity, message=message)
        )
    severities = {f.severity for f in findings}
    overall: Rating = "red" if "high" in severities else ("amber" if severities else "green")
    return findings, overall


def _make_figures(
    output_dir: Path,
    s_dev: Scored,
    s_oot: Scored,
    *,
    oot_metrics: SampleMetrics,
    stability: StabilityResults,
    ifrs9: IFRS9Results,
    ts: TermStructure,
) -> dict[str, str]:
    fig_dir = output_dir / "figures"
    cal = oot_metrics.calibration_table
    curves = {name: c.cumulative.mean(axis=0) for name, c in ts.by_scenario.items()}
    curves["weighted"] = ts.cumulative.mean(axis=0)
    stage_rows = [r for r in ifrs9.ecl_by_stage if r["stage"] != "total"]
    paths = {
        "roc": figures.roc_curve_plot(
            {"development": (s_dev.y, s_dev.pd_pit), "out-of-time": (s_oot.y, s_oot.pd_pit)},
            fig_dir / "roc_curve.png",
        ),
        "calibration": figures.calibration_plot(
            [r["mean_pd"] for r in cal],
            [r["observed_rate"] for r in cal],
            [r["n"] for r in cal],
            fig_dir / "calibration.png",
        ),
        "score_distribution": figures.score_distribution_plot(
            {"development": s_dev.score, "out-of-time": s_oot.score},
            fig_dir / "score_distribution.png",
        ),
        "csi": figures.psi_bar_plot(
            [str(r["feature"]) for r in stability.csi_table],
            [float(r["csi"]) for r in stability.csi_table],
            fig_dir / "csi.png",
            "Characteristic stability index (dev vs OOT)",
        ),
        "ecl_by_stage": figures.ecl_by_stage_plot(
            [str(r["stage"]) for r in stage_rows],
            [float(r["ecl"]) for r in stage_rows],
            [float(r["coverage"]) for r in stage_rows],
            fig_dir / "ecl_by_stage.png",
        ),
        "term_structure": figures.pd_term_structure_plot(curves, fig_dir / "pd_term_structure.png"),
    }
    return {k: f"figures/{p.name}" for k, p in paths.items()}


def run_pipeline(
    cfg: PipelineConfig | None = None,
    output_dir: Path | None = None,
    portfolio: pd.DataFrame | None = None,
    model: FittedModel | None = None,
) -> ValidationResults:
    """Run the full build-and-validate pipeline.

    Args:
        cfg: Pipeline configuration (defaults to :class:`PipelineConfig`).
        output_dir: Where to write the report, JSON, figures and scorecard; if
            ``None`` nothing is written.
        portfolio: Portfolio table to use instead of generating one.
        model: Pre-fitted model to validate instead of fitting one.

    Returns:
        The :class:`ValidationResults` of the run.
    """
    cfg = cfg or PipelineConfig()
    df = validate_portfolio(portfolio if portfolio is not None else generate_portfolio(cfg.data))
    dev_all, oot_all = split_dev_oot(df, cfg.data.dev_cutoff)
    dev = performing(dev_all, cfg.staging.default_dpd)
    oot = performing(oot_all, cfg.staging.default_dpd)
    fitted = model if model is not None else fit_model(dev, cfg.model)

    s_dev, s_oot = score_sample(fitted, dev), score_sample(fitted, oot)
    samples = {
        "dev": evaluate_sample("dev", s_dev, cfg.data.seed),
        "oot": evaluate_sample("oot", s_oot, cfg.data.seed),
    }
    backtest = out_of_time_backtest(
        ScoredSample(s_dev.y, s_dev.pd_pit, s_dev.score),
        ScoredSample(s_oot.y, s_oot.pd_pit, s_oot.score),
    )
    stability = _stability(dev, oot, s_dev, s_oot)
    ifrs9, ts, _ = _ifrs9(fitted, oot_all, cfg)
    lights = _traffic_lights(samples, backtest, stability, cfg)
    findings, overall = _findings(lights, samples, backtest, stability)

    results = ValidationResults(
        package_version=__version__,
        seed=cfg.data.seed,
        config=cfg.model_dump(mode="json"),
        data_quality=_data_quality(
            df, dev_all, oot_all, n_dev=len(dev), n_oot=len(oot), cutoff=cfg.data.dev_cutoff
        ),
        model_design=_model_design(fitted, float(dev[TARGET].mean())),
        samples=samples,
        backtest=backtest,
        stability=stability,
        ifrs9=ifrs9,
        traffic_lights=dict(lights),
        findings=findings,
        overall_rating=overall,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        results.figures = _make_figures(
            output_dir,
            s_dev,
            s_oot,
            oot_metrics=samples["oot"],
            stability=stability,
            ifrs9=ifrs9,
            ts=ts,
        )
        save_model(output_dir / "scorecard.json", fitted.scorecard, fitted.binner)
        results.to_json(output_dir / "validation_results.json")
        build_validation_report(results, output_dir)
    return results
