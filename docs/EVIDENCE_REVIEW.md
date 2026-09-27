# Evidence review — 2026-09-27

The portfolio connects auditable extraction with independent credit-model validation.
The RAG baseline is a small synthetic mechanism check; the IFRS9 report is a seeded
synthetic model-validation study. Neither is an observed production outcome.

## What the reviewer can inspect

- [Current RAG evaluation](../projects/annual-report-risk-rag/reports/eval_rules.md):
  13 accepted values match gold, seven disclosed values are missed, and precision is 1.000
  on this small fixture. Micro-F1 is 0.788. Lower recall is retained after stricter checks.
- [IFRS9 validation report](../projects/ifrs9-pd-model/reports/validation_report.md):
  the original RED calibration finding remains unchanged. A subsequent, separately
  [frozen remediation study](../projects/ifrs9-pd-model/docs/REMEDIATION_PROTOCOL.md)
  now reports a new synthetic 2025 assessment: candidate not accepted, finding OPEN.
  It repairs the comparison arms' macro preprocessing look-ahead, preserves the
  negative result, and reports conditional ECL impact on the same book.
- [Adversarial evidence tests](../projects/annual-report-risk-rag/tests/test_grounding.py):
  decimal truncation, sign changes, Unicode/separated minus signs, wrong units, wrong periods,
  half-year mismatches, another metric's value and an unknown metric in a later clause.

## Corrected evidence contract

Previously a string-substring check could accept 12 from a quote containing 12.4%, and
periods were not checked. The validator now parses complete signed numeric tokens,
requires the extraction unit to match an explicit adjacent unit, binds the number to a
preceding known metric and rejects a later competing quantity. A claimed period must
occur unambiguously in the quote. Missing periods are not invented. Unsupported or
ambiguous constructions abstain, even when that reduces benchmark recall.

Alias matching, annual-label normalisation and typical ranges remain heuristics. These
checks do not establish full semantic entailment, reporting scope, fiscal conventions,
currency identity or robustness on real annual-report tables. A live-model comparison
and external-report gold set remain future work requiring a separate evaluation.

## Reproduce without credentials

From the repository root:

```bash
uv sync --all-packages
uv run report-rag evaluate --extractor rules --output /tmp/rag-review.md
uv run pytest --cov
```

On Windows, use a writable local output path in place of `/tmp/rag-review.md`.
The historical [v1 result](../projects/annual-report-risk-rag/reports/eval_rules_legacy_v1.md)
is preserved unchanged; the current result is regenerated with the corrected validator.
The tests require no network. Local verification used Python 3.12; when its installed
NumPy stubs use Python 3.12 syntax, run `mypy --python-version 3.12` for that environment.

Local verification passed ruff lint/format, strict mypy with the Python 3.12 target,
and all 197 offline tests at 97.58% branch coverage. An existing report-rebuild test
was corrected to compare against the original output, using explicit UTF-8 on Windows.
The CI workflow already supports manual dispatch; no cloud service or live model was used.
