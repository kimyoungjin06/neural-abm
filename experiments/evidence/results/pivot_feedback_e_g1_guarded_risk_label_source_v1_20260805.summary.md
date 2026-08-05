# E/G1 guard-first risk and label source v1

Date: 2026-08-05

Independent verification:
`PASS_INDEPENDENT_G1_GUARD_FIRST_RISK_AND_LABEL_SOURCE_ROW_REPLAY_ESTIMATION_NOT_RUN`

## Outcome

The explicitly approved guard-only exception was applied before any G1 label
logic. The production source contains 84 frozen response-cohort files and
72,715,221 guarded author-opportunity rows. Independent replay found zero
guard overlaps, zero label-logic mismatches, zero cohort-constant mismatches,
and zero author-order or duplicate errors.

The post-promotion mode check found that the intermediate `private/` parent had
been created as `0775` even though the payload directory was already `0700` and
every file was `0600`. That one parent was tightened to `0700`; no artifact
byte, receipt, report, manifest, or cohort SHA-256 changed. Independent
verification ran only after all production directories were `0700`.

| Accounting item | Independently replayed value |
| --- | ---: |
| Pre-guard risk rows | 95,392,671 |
| Guard-excluded rows | 22,677,450 |
| Guarded risk rows | 72,715,221 |
| Entry rows | 17,411 |
| Non-entry rows | 72,697,810 |
| Persistence rows | 3,184 |
| One-off entry rows | 14,227 |
| Alternative-pivot entry rows | 10,086 |

## Frozen support result

The source materialization passed, but the registered execution is terminally
blocked before sampling or estimation. The direct frozen policy
`FIELD_EPISODE_PARTITION_MINIMUM_500_WITH_100_PER_OUTCOME_CLASS_V1` requires at
least 100 entry events per field-episode unit, forbids pooling to rescue a
failed episode, and assigns
`EXCLUDE_UNIT_AND_FAIL_REGISTERED_COVERAGE_REQUIREMENT`.

| Episode | Field | Topic | Shock year | Total | Entries | Entry shortfall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 32 | 14085 | 2015 | 1,673,160 | 58 | 42 |
| 25 | 11 | 14195 | 2015 | 3,624,912 | 28 | 72 |

The resulting registered status is
`BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION`.
No scope or threshold was changed, no feature/control source was materialized,
and no sampling, weighting, estimator, bootstrap, or hypothesis decision ran.

The config field `minimum_source_year: 2010` denotes the earliest three-year
activity-window year for the 2013 response cohort. The lifetime no-prior-target
check intentionally scans eligible selected-field work before 2010 as well; it
does not truncate prior target history at 2010.

## Access boundary

The exact 8,174,698-row author-cold guard was the only private G4 file read.
No holdout outcome row, other private G4 file, model-selection score, or sealed
artifact was read; the seal remained unopened and `g4_open=false`.

This result establishes neither an association nor a null, predictive result,
mechanism, transport claim, causal effect, or scientific effect.
