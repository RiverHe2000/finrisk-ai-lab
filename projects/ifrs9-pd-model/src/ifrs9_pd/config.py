"""Configuration models for the IFRS 9 PD pipeline.

All tunable parameters live here as pydantic v2 models so that a run is fully
described by a JSON-serialisable configuration and can be reproduced
byte-for-byte.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataConfig(BaseModel):
    """Parameters of the synthetic retail-mortgage portfolio generator.

    Attributes:
        n_loans: Number of loan-snapshot rows to generate.
        seed: Seed of the NumPy random generator.
        origination_start: First origination month (inclusive, ``YYYY-MM``).
        origination_end: Last origination month (inclusive, ``YYYY-MM``).
        last_snapshot: Last observable snapshot month; 12 months of outcome
            data are assumed to exist after it.
        dev_cutoff: Snapshots up to and including this month form the
            development sample; later snapshots are out-of-time (OOT).
        missing_rate: Share of rows with a missing ``bureau_score`` and
            (independently) a missing ``debt_to_income``.
        target_default_rate: Approximate portfolio 12-month default rate the
            latent PD is centred on.
    """

    model_config = ConfigDict(frozen=True)

    n_loans: int = Field(default=20_000, ge=100)
    seed: int = Field(default=42, ge=0)
    origination_start: str = "2017-01"
    origination_end: str = "2023-12"
    last_snapshot: str = "2023-12"
    dev_cutoff: str = "2022-06"
    missing_rate: float = Field(default=0.015, ge=0.0, le=0.2)
    target_default_rate: float = Field(default=0.025, gt=0.0, lt=0.5)


class ModelConfig(BaseModel):
    """Scorecard, binning and master-scale parameters.

    Attributes:
        pdo: Points to double the odds.
        base_score: Score assigned at ``base_odds``.
        base_odds: Good:bad odds at ``base_score``.
        n_grades: Number of rating grades on the master scale.
        grade_min_pd: Lower PD anchor of the geometric master scale.
        grade_max_pd: Upper PD anchor of the geometric master scale.
        n_bins: Initial number of quantile bins for numeric features.
        min_bin_share: Minimum share of observations in a bin after merging.
            Set well below the usual 5% so that sparse delinquency indicators
            (``current_dpd > 0`` is under 2% of the performing book) keep a
            bin of their own.
        min_iv: Minimum information value for a feature to be selected.
        max_iv: Optional maximum information value (a leakage guard for real
            data; unset here because the synthetic bureau score is dominant by design).
        regularisation_c: Inverse L2 regularisation strength of the logistic
            regression.
        asset_correlation: Vasicek asset correlation ``rho`` (Basel II
            residential mortgages: 0.15).
    """

    model_config = ConfigDict(frozen=True)

    pdo: float = Field(default=20.0, gt=0.0)
    base_score: float = Field(default=600.0)
    base_odds: float = Field(default=50.0, gt=0.0)
    n_grades: int = Field(default=10, ge=3, le=25)
    grade_min_pd: float = Field(default=0.0003, gt=0.0)
    grade_max_pd: float = Field(default=0.30, lt=1.0)
    n_bins: int = Field(default=10, ge=3, le=50)
    min_bin_share: float = Field(default=0.01, gt=0.0, lt=0.5)
    min_iv: float = Field(default=0.02, ge=0.0)
    max_iv: float | None = Field(default=None, gt=0.0)
    regularisation_c: float = Field(default=1.0, gt=0.0)
    asset_correlation: float = Field(default=0.15, gt=0.0, lt=1.0)


class StagingConfig(BaseModel):
    """Significant-increase-in-credit-risk (SICR) and default rules.

    Attributes:
        relative_pd_threshold: Stage 2 if ``PD_now / PD_orig`` is at least this
            multiple (and the absolute test also passes).
        absolute_pd_threshold_bps: Stage 2 requires ``PD_now - PD_orig`` of at
            least this many basis points, which stops tiny PDs from tripping the
            relative test.
        dpd_backstop: Days past due at which Stage 2 is mandatory
            (IFRS 9 B5.5.19 rebuttable presumption).
        default_dpd: Days past due at which a loan is in default (Stage 3).
    """

    model_config = ConfigDict(frozen=True)

    relative_pd_threshold: float = Field(default=2.0, ge=1.0)
    absolute_pd_threshold_bps: float = Field(default=50.0, ge=0.0)
    dpd_backstop: int = Field(default=30, ge=1)
    default_dpd: int = Field(default=90, ge=1)

    @model_validator(mode="after")
    def _check_dpd_order(self) -> StagingConfig:
        if self.default_dpd <= self.dpd_backstop:
            msg = "default_dpd must exceed dpd_backstop"
            raise ValueError(msg)
        return self


class ECLConfig(BaseModel):
    """Expected-credit-loss inputs.

    Attributes:
        lgd: Portfolio loss-given-default used when no segment override matches.
        lgd_by_segment: Optional LGD overrides keyed by region.
        discount_rate: Annual effective interest rate for discounting.
        horizon_months: Lifetime horizon ``H`` for the term structure.
        peak_month: Month at which the seasoning hazard multiplier peaks.
        peak_multiplier: Value of the seasoning multiplier at its peak.
        mean_reversion: Per-month mean-reversion speed of the scenario
            systematic-factor shift.
    """

    model_config = ConfigDict(frozen=True)

    lgd: float = Field(default=0.35, gt=0.0, le=1.0)
    lgd_by_segment: dict[str, float] = Field(default_factory=dict)
    discount_rate: float = Field(default=0.05, ge=0.0)
    horizon_months: int = Field(default=60, ge=12, le=360)
    peak_month: int = Field(default=24, ge=1)
    peak_multiplier: float = Field(default=1.4, ge=1.0)
    mean_reversion: float = Field(default=0.05, ge=0.0, le=1.0)


class ScenarioConfig(BaseModel):
    """Macro-economic scenarios used for probability weighting.

    ``z_shifts`` are shifts of the Vasicek systematic factor at month 1;
    negative values increase PD. The defaults are dispersed enough that the
    probability-weighted 12-month PD is close to the unconditional (TTC) PD,
    which is the usual sanity check on a discrete scenario set.

    Attributes:
        weights: Scenario probability weights (must sum to one).
        z_shifts: Systematic factor shift per scenario.
    """

    model_config = ConfigDict(frozen=True)

    weights: dict[str, float] = Field(
        default_factory=lambda: {"base": 0.5, "upside": 0.25, "downside": 0.25}
    )
    z_shifts: dict[str, float] = Field(
        default_factory=lambda: {"base": 0.0, "upside": 1.0, "downside": -1.5}
    )

    @model_validator(mode="after")
    def _check_scenarios(self) -> ScenarioConfig:
        if set(self.weights) != set(self.z_shifts):
            msg = "weights and z_shifts must define the same scenarios"
            raise ValueError(msg)
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            msg = "scenario weights must sum to 1"
            raise ValueError(msg)
        return self


class MetricThreshold(BaseModel):
    """Green/amber boundaries for one validation metric.

    For ``higher_is_better`` metrics a value ``>= green`` is green, a value
    ``>= amber`` is amber and anything lower is red. The comparisons flip for
    lower-is-better metrics such as PSI.

    Attributes:
        green: Boundary between green and amber.
        amber: Boundary between amber and red.
        higher_is_better: Direction of the metric.
    """

    model_config = ConfigDict(frozen=True)

    green: float
    amber: float
    higher_is_better: bool = True

    @model_validator(mode="after")
    def _check_order(self) -> MetricThreshold:
        ordered = self.green >= self.amber if self.higher_is_better else self.green <= self.amber
        if not ordered:
            msg = "green boundary must be stricter than amber boundary"
            raise ValueError(msg)
        return self


class ValidationThresholds(BaseModel):
    """Traffic-light thresholds used throughout the validation suite.

    Rationale: Gini and KS bands follow common retail-scorecard practice
    (Basel Committee Working Paper 14 discusses discriminatory-power
    benchmarks; many banks treat Gini below 0.40 as weak for retail).
    PSI bands 0.10 / 0.25 are the industry rule of thumb for population
    shift. Statistical tests use the conventional 5% significance level and
    a 1% level for red.
    """

    model_config = ConfigDict(frozen=True)

    gini: MetricThreshold = MetricThreshold(green=0.50, amber=0.40)
    ks: MetricThreshold = MetricThreshold(green=0.30, amber=0.20)
    psi: MetricThreshold = MetricThreshold(green=0.10, amber=0.25, higher_is_better=False)
    hl_p_value: MetricThreshold = MetricThreshold(green=0.05, amber=0.01)
    binomial_p_value: MetricThreshold = MetricThreshold(green=0.05, amber=0.01)
    gini_deterioration: MetricThreshold = MetricThreshold(
        green=0.05, amber=0.10, higher_is_better=False
    )
    pd_to_dr_ratio: MetricThreshold = MetricThreshold(green=0.85, amber=0.70)


class PipelineConfig(BaseModel):
    """Bundle of every configuration section consumed by the pipeline."""

    model_config = ConfigDict(frozen=True)

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    staging: StagingConfig = StagingConfig()
    ecl: ECLConfig = ECLConfig()
    scenarios: ScenarioConfig = ScenarioConfig()
    thresholds: ValidationThresholds = ValidationThresholds()
