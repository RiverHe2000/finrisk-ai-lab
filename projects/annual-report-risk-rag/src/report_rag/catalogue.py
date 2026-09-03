"""Catalogue of prudential and credit-risk metrics the pipeline extracts.

The list is deliberately opinionated towards what an Australian ADI discloses
under APRA's capital (APS 110/111), liquidity (APS 210) and credit-quality
reporting, which is what a model-risk or credit-risk team would ask for.
"""

from __future__ import annotations

from report_rag.schemas import MetricSpec, Unit

RISK_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        metric_id="cet1_ratio",
        name="Common Equity Tier 1 ratio",
        description="CET1 capital as a percentage of total risk-weighted assets (Level 2).",
        unit=Unit.PERCENT,
        query="Common Equity Tier 1 CET1 capital ratio percent risk-weighted assets",
        aliases=("CET1 ratio", "CET1 capital ratio", "Common Equity Tier 1 ratio"),
        typical_range=(4.5, 25.0),
    ),
    MetricSpec(
        metric_id="tier1_ratio",
        name="Tier 1 capital ratio",
        description="Tier 1 capital (CET1 plus Additional Tier 1) as a percentage of RWA.",
        unit=Unit.PERCENT,
        query="Tier 1 capital ratio percent additional tier 1",
        aliases=("Tier 1 ratio", "Tier 1 capital ratio"),
        typical_range=(6.0, 30.0),
    ),
    MetricSpec(
        metric_id="total_capital_ratio",
        name="Total capital ratio",
        description="Total regulatory capital (Tier 1 + Tier 2) as a percentage of RWA.",
        unit=Unit.PERCENT,
        query="Total capital ratio percent tier 2 regulatory capital",
        aliases=("Total capital ratio", "Total capital adequacy ratio"),
        typical_range=(8.0, 35.0),
    ),
    MetricSpec(
        metric_id="leverage_ratio",
        name="Leverage ratio",
        description="Tier 1 capital divided by total exposures (APS 110 leverage ratio).",
        unit=Unit.PERCENT,
        query="leverage ratio percent Tier 1 capital total exposures",
        aliases=("Leverage ratio",),
        typical_range=(2.0, 15.0),
    ),
    MetricSpec(
        metric_id="lcr",
        name="Liquidity Coverage Ratio",
        description="High-quality liquid assets over net cash outflows in a 30-day stress.",
        unit=Unit.PERCENT,
        query="Liquidity Coverage Ratio LCR percent high quality liquid assets net cash outflows",
        aliases=("LCR", "Liquidity Coverage Ratio"),
        typical_range=(100.0, 250.0),
    ),
    MetricSpec(
        metric_id="nsfr",
        name="Net Stable Funding Ratio",
        description="Available stable funding over required stable funding.",
        unit=Unit.PERCENT,
        query="Net Stable Funding Ratio NSFR percent available stable funding",
        aliases=("NSFR", "Net Stable Funding Ratio"),
        typical_range=(100.0, 200.0),
    ),
    MetricSpec(
        metric_id="rwa_total",
        name="Total risk-weighted assets",
        description="Total RWA across credit, market and operational risk, in AUD millions.",
        unit=Unit.AUD_MILLION,
        query="total risk-weighted assets RWA million credit risk operational risk market risk",
        aliases=("Total RWA", "risk-weighted assets", "Risk weighted assets"),
        typical_range=(1_000.0, 2_000_000.0),
    ),
    MetricSpec(
        metric_id="npl_ratio",
        name="Non-performing loans ratio",
        description="Gross impaired plus 90+ days past due loans as a share of gross loans.",
        unit=Unit.PERCENT,
        query="non-performing loans ratio 90 days past due impaired percent gross loans",
        aliases=(
            "non-performing loans ratio",
            "non-performing exposures ratio",
            "90+ days past due and impaired",
        ),
        typical_range=(0.0, 15.0),
    ),
    MetricSpec(
        metric_id="provision_coverage",
        name="Provision coverage ratio",
        description="Total provisions for credit impairment as a percentage of credit RWA.",
        unit=Unit.PERCENT,
        query="provision coverage ratio total provisions credit risk-weighted assets percent",
        aliases=("provision coverage", "collective provision coverage", "provisions to credit RWA"),
        typical_range=(0.1, 5.0),
    ),
    MetricSpec(
        metric_id="loan_impairment_expense",
        name="Loan impairment expense",
        description="Credit impairment charge for the year, in AUD millions.",
        unit=Unit.AUD_MILLION,
        query="loan impairment expense credit impairment charge million for the year",
        aliases=("loan impairment expense", "credit impairment charge", "impairment expense"),
        typical_range=(-5_000.0, 20_000.0),
    ),
    MetricSpec(
        metric_id="stage3_ecl",
        name="Stage 3 expected credit losses",
        description="IFRS 9 / AASB 9 stage 3 (credit-impaired) provisions, in AUD millions.",
        unit=Unit.AUD_MILLION,
        query="Stage 3 expected credit loss provision credit-impaired lifetime ECL million",
        aliases=("Stage 3 ECL", "stage 3 provisions", "credit-impaired provisions"),
        typical_range=(0.0, 50_000.0),
    ),
    MetricSpec(
        metric_id="traded_var",
        name="Traded market risk VaR",
        description="Average 1-day 99% Value-at-Risk for the trading book, in AUD millions.",
        unit=Unit.AUD_MILLION,
        query="Value at Risk VaR traded market risk average 99% one-day million",
        aliases=("VaR", "Value-at-Risk", "traded market risk VaR"),
        typical_range=(0.0, 500.0),
    ),
)

METRICS_BY_ID: dict[str, MetricSpec] = {m.metric_id: m for m in RISK_METRICS}


def get_metric(metric_id: str) -> MetricSpec:
    """Look up a metric by id, raising ``KeyError`` with the known ids on miss."""
    try:
        return METRICS_BY_ID[metric_id]
    except KeyError as exc:
        known = ", ".join(sorted(METRICS_BY_ID))
        msg = f"Unknown metric '{metric_id}'. Known metrics: {known}"
        raise KeyError(msg) from exc
