# Changelog

All notable changes to this repository are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-09-03

### Added

* Monorepo scaffold: uv workspace, shared ruff / mypy (strict) / pytest configuration,
  pre-commit hooks, GitHub Actions CI on Python 3.11-3.13 with a coverage gate.
* `annual-report-risk-rag`: hybrid BM25 + dense retrieval, Claude structured-output extraction,
  grounding validator, tool-calling agent mode, rule-based baseline, gold-label evaluation harness,
  two synthetic annual reports with labels, committed baseline evaluation report.
* `ifrs9-pd-model`: synthetic mortgage portfolio generator, WoE binning, logistic scorecard,
  intercept calibration and Vasicek PIT/TTC conversion, lifetime PD term structure with scenario
  weighting, SICR staging, ECL, and an independent validation suite (discrimination, calibration,
  stability, out-of-time backtest) that writes a model-validation report.
