# Synthetic IFRS 9 remediation assessment v1

**Decision: OPEN. Original report: RED, unchanged.**
Candidate disposition: **not_accepted**.

One frozen intercept correction, evaluated on one predeclared new synthetic window.
This is not real-data external validation, independent expert review, or production approval.

Frozen payload SHA-256: `949c44f9040745e4c4aec47fe3cc762a4a670f4d80215748a43a56cb41fa9d49`.
Generated data receipt: `6e0cfb091476c3e584a02b41bcb6cceaf4291d9db778b076ca2e99dd44c59b92`.
Snapshots: 2025-01 to 2025-12; simulated outcome maturity through 2026-12.
Book: 6,176 loans; performing: 6,154; defaults: 88.

Both arms use macro mean/std frozen on 2016-01 to 2023-12. The reference retains the
original scorecard ranking with a development-only overlay under this repaired transform;
it is not a replay of the historical report's future-normalised macro path.
Candidate parameters were fixed
on the old OOT calibration sample before this new data pool was generated.

| Metric | Reference | Candidate |
|---|---:|---:|
| mean_pd | 0.013393 | 0.017560 |
| default_rate | 0.014300 | 0.014300 |
| pd_to_dr_ratio | 0.936583 | 1.227996 |
| gini | 0.652767 | 0.652785 |
| ks | 0.484893 | 0.484893 |
| hl_p_value | 0.000081 | 0.000645 |
| binomial_p_value_overall | 0.541498 | 0.052070 |
| brier | 0.013532 | 0.013720 |

## Frozen closure checks

| Check | Result |
|---|---|
| binomial | PASS |
| enough_defaults | PASS |
| finite_statistics | PASS |
| gini | PASS |
| gini_preserved | PASS |
| hl | FAIL |
| ks | PASS |
| pd_dr | FAIL |
| score_psi | PASS |

Baseline-score PSI: 0.003377.

## ECL impact on the same book

Amounts are AUD; assumptions are unchanged and unvalidated.

| Measure | Reference | Candidate |
|---|---:|---:|
| total_ead | 2,953,173,622.880000 | 2,953,173,622.880000 |
| total_ecl | 34,193,140.354745 | 41,094,227.760828 |
| coverage_ratio | 0.011578 | 0.013915 |

PD-curve effect with old stages fixed: AUD 6,959,347.31.
Additional stage effect: AUD -58,259.91.
Total ECL change: AUD 6,901,087.41.

| Stage before | Stage after | Loans | EAD |
|---:|---:|---:|---:|
| 1 | 1 | 5726 | 2,735,985,620.18 |
| 1 | 2 | 8 | 3,879,620.36 |
| 2 | 1 | 1 | 758,243.55 |
| 2 | 2 | 419 | 200,106,816.92 |
| 3 | 3 | 22 | 12,443,321.87 |

## Interpretation and limits

Failed checks remain open; do not choose another seed or tune against this window.
The original RED finding is retained. See the frozen protocol for the numerical gates.
The next research action is to diagnose calibration shape by grade and macro regime;
an intercept-only correction cannot guarantee decile calibration. Any new candidate
needs a separately frozen protocol and fresh assessment data. This window is now used
and must not be advertised as untouched in follow-up model selection.
JSON includes confidence intervals, grade tests, stage ECL and scenario sensitivities.
LGD 35%, discount 5%, linear EAD, 60-month horizon and subjective scenario weights
are assumptions. Origination PD is reconstructed at matched age. ECL changes measure
conditional model sensitivity, not realised loss improvement or an accounting adjustment.
The synthetic generator re-centres its latent intercept per generated pool; a new seed
does not establish external generalisation. Labels through 2026-12 are simulated.
