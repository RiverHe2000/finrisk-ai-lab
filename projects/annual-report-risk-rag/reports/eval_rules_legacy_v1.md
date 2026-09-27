# Extraction evaluation - extractor `rules`

Micro-F1 **0.947**, macro-F1 **0.945** over 2 document(s).

| Document | Metrics | TP | FP | FN | TN | Precision | Recall | F1 | Mean rel. error | Retrieval recall@k | Grounding rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| harbour_mutual_fy2025 | 12 | 7 | 0 | 1 | 4 | 1.000 | 0.875 | 0.933 | 0.0% | 100.0% | 100.0% |
| southern_cross_bank_fy2025 | 12 | 11 | 0 | 1 | 0 | 1.000 | 0.917 | 0.957 | 0.0% | 100.0% | 91.7% |

## harbour_mutual_fy2025

| Metric | Outcome | Gold | Predicted | Grounding | Retrieved gold section |
|---|---|---:|---:|---|---|
| cet1_ratio | TP | 15.8 | 15.8 | grounded | yes |
| tier1_ratio | TP | 15.8 | 15.8 | grounded | yes |
| total_capital_ratio | TP | 17.6 | 17.6 | grounded | yes |
| leverage_ratio | TN | null | - | not_found | n/a |
| lcr | TN | null | - | not_found | n/a |
| nsfr | TN | null | - | not_found | n/a |
| rwa_total | TP | 3120 | 3120 | grounded | yes |
| npl_ratio | FN | 0.42 | - | not_found | yes |
| provision_coverage | TP | 0.71 | 0.71 | grounded | yes |
| loan_impairment_expense | TP | 3.9 | 3.9 | grounded | yes |
| stage3_ecl | TP | 4.1 | 4.1 | grounded | yes |
| traded_var | TN | null | - | not_found | n/a |

## southern_cross_bank_fy2025

| Metric | Outcome | Gold | Predicted | Grounding | Retrieved gold section |
|---|---|---:|---:|---|---|
| cet1_ratio | TP | 12.4 | 12.4 | grounded | yes |
| tier1_ratio | TP | 14.2 | 14.2 | grounded | yes |
| total_capital_ratio | TP | 18.9 | 18.9 | grounded | yes |
| leverage_ratio | TP | 5.3 | 5.3 | grounded | yes |
| lcr | TP | 134 | 134 | grounded | yes |
| nsfr | TP | 121 | 121 | grounded | yes |
| rwa_total | TP | 215640 | 215640 | grounded | yes |
| npl_ratio | FN | 0.94 | - | out_of_range | yes |
| provision_coverage | TP | 1.58 | 1.58 | grounded | yes |
| loan_impairment_expense | TP | 612 | 612 | grounded | yes |
| stage3_ecl | TP | 1480 | 1480 | grounded | yes |
| traded_var | TP | 18.6 | 18.6 | grounded | yes |
