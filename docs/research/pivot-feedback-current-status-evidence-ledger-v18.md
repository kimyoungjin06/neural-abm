# Pivot-feedback current-status and evidence ledger v18

Date: 2026-08-05

Status:
`CURRENT_HUMAN_FACING_INDEX_E_G1_TERMINAL_SUPPORT_BLOCKED_AFTER_GUARD_FIRST_CONFIRMATORY_LABELS_NO_ESTIMATION`

## The approved exception was executed

The previous E/G1 block was an authorization contradiction, not an empirical
result. The explicit exception permitted only the exact 8,174,698-row
author-cold guard as an operator-only anti-join before any G1 label, control,
sampling, weighting, or estimator logic. Its source identity and all-true
`exclude_from_g1` projection were independently verified. No other private G4
file or sealed-root artifact was authorized or read.

## Guard-first materialization passed

The resumed run covered the frozen 2013–2016 G1 scope: fields 16, 32, and 11;
28 certified episodes; and offsets 0, 1, and 2, producing 84 cohort files. The
production source is at
`/ssd/pivot-feedback-e-g1-guarded-risk-label-source-v1-20260805`.

Independent full-row replay verified:

| Accounting item | Value |
| --- | ---: |
| Pre-guard risk rows | 95,392,671 |
| Guard-excluded rows | 22,677,450 |
| Guarded risk/label rows | 72,715,221 |
| Entry rows | 17,411 |
| Non-entry rows | 72,697,810 |
| Persistence rows | 3,184 |
| One-off entry rows | 14,227 |
| Alternative-pivot entry rows | 10,086 |

The independent replay found zero guard overlaps, label-logic mismatches,
cohort-constant mismatches, author duplicates, or ordering errors.

The first independent mode check stopped before reading cohort content because
the intermediate `private/` parent was `0775`; its payload child was already
`0700` and all payload files were `0600`. The exact parent was tightened to
`0700` without changing any artifact byte or SHA-256, after which the full
independent replay passed. Also, `minimum_source_year: 2010` is the earliest
three-year activity-window year; the lifetime no-prior-target check correctly
scans eligible history before 2010 rather than truncating it.

## The registered support floor produced the terminal result

This is no longer an authorization or audit block. It is the first empirical
support result from the resumed confirmatory execution. The direct frozen rule
`FIELD_EPISODE_PARTITION_MINIMUM_500_WITH_100_PER_OUTCOME_CLASS_V1` requires
500 total opportunities, 100 entry events, and 100 non-entry opportunities per
field-episode unit before negative sampling. It forbids pooling a failed episode
into other episodes and assigns
`EXCLUDE_UNIT_AND_FAIL_REGISTERED_COVERAGE_REQUIREMENT`.

| Episode | Field | Topic | Shock year | Total | Entries | Required | Shortfall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 32 | 14085 | 2015 | 1,673,160 | 58 | 100 | 42 |
| 25 | 11 | 14195 | 2015 | 3,624,912 | 28 | 100 | 72 |

The other 26 episodes passed this floor. Because the two failed units violate
registered coverage, the terminal status is
`BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION`.

## Why execution stops here

Continuing to feature construction or estimation cannot repair the already
observed coverage failure. Lowering 100, pooling episodes, dropping the two
units while retaining the registered claim, or selecting a new scope after
seeing the outcomes would change the frozen decision rule. Those actions are
not authorized on this confirmatory cohort.

This support floor is a precommitted minimum screen, not a universal scientific
law or a claim that estimates from 58 or 28 events are intrinsically impossible.
The strict consequence comes from the registered coverage claim. A future
analysis may use a narrower claim and a redesigned support rule only in a new,
disjoint, outcome-blind, preauthorized successor cohort.

## Active state and access boundary

The independently verified
[`current state v4`](../../experiments/evidence/results/pivot_feedback_current_state_v4_20260805.summary.md)
records the confirmatory read rather than erasing it:

| Boundary | Verified state |
| --- | --- |
| Guarded confirmatory rows materialized | 72,715,221 |
| Feature/control rows materialized | 0 |
| Sampling or weighting runs | 0 |
| Estimators or bootstraps | 0 |
| Holdout outcome rows read | 0 |
| Other private G4 files read | 0 |
| Model-selection scores read | 0 |
| Seal opens | 0 |
| `g4_open` | `false` |

F and G0 compatibility remain passed; E and G1 remain failed/blocked; M is not
run; H remains byte-sealed and unopened. No association, null, prediction,
mechanism, transport, causal effect, or scientific effect has been established.
