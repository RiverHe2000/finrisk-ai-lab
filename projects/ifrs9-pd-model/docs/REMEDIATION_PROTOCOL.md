# Intercept remediation protocol v1

Frozen before generating or scoring the new synthetic assessment window. This is
a model-risk case study, not real borrower data, independent expert validation,
an IFRS 9 compliance opinion or a production approval.

## Finding and proposed action

The original seed-42 report remains RED and is retained byte-for-byte. Its
2022-07 to 2023-12 out-of-time sample has PD/default-rate 0.76 and a
Hosmer-Lemeshow p-value near 0.003. That sample is now explicitly reused for
calibration; it is no longer an independent test of the remediation.

Load the committed scorecard and retain its bins, features, coefficients and asset
correlation. Freeze macro mean and sample standard deviation
using only January 2016 to December 2023. Both assessment arms use this past-only
normalisation. Refit the reference overlay loading and development calibration
shift on original performing development rows using that fixed transform, removing
the inherited future-normalised overlay. The raw logistic coefficients and intercept
remain unchanged. The
candidate fits exactly one additive
TTC-logit shift on performing calibration rows, solving for mean PIT PD equal to
their observed default rate. Do not refit rank ordering, select another method,
change scenarios, or adjust thresholds after viewing the new-window result.

## Frozen sequence and temporal boundary

1. `remediation-prepare` verifies the original report and model hashes, reproduces
   seed-42 calibration rows and writes `frozen_candidate.json`, including the
   complete protocol, candidate model, calibration statistics and input hashes.
   It does not generate new-window data. It refuses to overwrite an existing file.
2. Commit the protocol and frozen candidate before the assessment run.
3. `remediation-evaluate` checks that frozen input against the original artifacts
   and reconstructs the candidate without fitting. Generate one 30,000-row pool
   using seed **20250927**, original origination start **2017-01**, origination end
   and last snapshot **2025-12**, and target mean latent PD **0.025**. Keep only
   snapshots **2025-01 to 2025-12**. Prefix loan IDs so they cannot collide with
   seed-42 IDs. All observations are fresh synthetic borrowers, not a longitudinal
   extension of the historical loans. There is no seed search or result tuning.
4. The latest calibration snapshot is December 2023; its 12-month labels mature
   in December 2024, before the first new-window snapshot. New-window outcomes
   through December 2026 are **simulated**, including future months relative to
   the date of this study. They are not observed outcomes.

The new-window reference is therefore a frozen-ranking reference under repaired
macro normalisation and development overlay, not an exact replay of the historical report's scoring
transform. This avoids giving either assessment arm future-derived transformation
parameters. The original report is not reclassified under the new transform.

The generator re-centres its latent intercept on each complete generated pool to
the configured target. The assessment window is then a calendar-selected subset;
its realised and expected default rates need not be 2.5%. This is a same-family
synthetic temporal challenge, not evidence of external generalisation. The legacy
macro proxy estimated its mean and standard deviation on the full 2016-2025 path;
that was a look-ahead limitation of model preprocessing, not only of the data
generator. This study repairs the assessment transform as described above. Future
macro values are applied individually at their snapshot dates without updating
the frozen normalisation. No future labels or features estimate model parameters.

## Decision rule fixed before assessment

Report both baseline and candidate on exactly the same performing new-window rows.
Close the narrow calibration finding in this simulation only if all pass:

- Gini >= 0.50; KS >= 0.30; HL p >= 0.05; portfolio binomial p >= 0.05.
- Mean PD / observed default rate in **[0.85, 1.20]**. The additional upper bound
  prevents excessive conservatism passing a one-sided underprediction check.
- Baseline-score PSI, using calibration-derived bins, <= 0.10.
- Candidate Gini falls by no more than 0.05 versus baseline on the same new rows.
- At least 50 new-window defaults; all statistics finite. Sparse samples remain open.

These are portfolio demonstration thresholds, not claimed regulatory mandates.
Report all failures and retain an OPEN decision when any fails. Passing does not
resolve the model's other limitations or change the original RED report.

## ECL impact and remaining assumptions

Use the identical complete new-window book for both variants, including Stage 3.
Keep LGD 35%, discount rate 5%, linear EAD amortisation, 60-month lifetime cap,
SICR thresholds, seasoning and base/upside/downside weights (50%/25%/25%) fixed.
Report total and stage ECL, coverage, scenario sensitivity, and a stage migration
matrix. Recalculate candidate ECL with baseline stages to separate the PD-curve
effect from the additional stage-change effect. Update current and age-matched
origination PD consistently under the candidate calibration.

LGD, EAD, macro scenarios, recovery timing and the lifetime cap are assumptions,
not validated submodels. Origination PD is a reconstructed age-matched view, not
a stored origination estimate. ECL deltas are conditional model sensitivity, not
an estimate of realised loss improvement or an approved accounting adjustment.
