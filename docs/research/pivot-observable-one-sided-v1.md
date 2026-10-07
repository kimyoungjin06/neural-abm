# Pivot one-sided observable test v1 (N09)

Date: 2026-10-07

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_observable_one_sided_v1.yaml`](../../experiments/configs/pivot_observable_one_sided_v1.yaml)

## Why this study exists

[N08](../../experiments/evidence/results/pivot_observable_confirmation_v1_20261001.summary.md)
tested the two persistence-trend observables against their N05 thresholds.
Neither was confirmed in every cell, so by the registered rule neither went to
the empirical stage as a two-sided discriminant. Every error, however, was a
missed weak channel. Across the 96 predictions of N05 and N08, neither
observable called an inactive cell active.

That one-sided reading was formed after N08's results were seen. This study
registers it and tests it:

> An observable past its threshold implies that learned policy is changing who
> enters.

## Design

The thresholds are N08's, unchanged and checked against the hash-bound N08
contract at load:

| Observable | Past the threshold means |
|---|---|
| Cohort persistence drop | below 0.023175 |
| Outcome selection trend | above 0.098417 |

The entry schedules are new in shape as well as in value. None repeats an N01,
N05, or N08 schedule:

| Schedule | Admission of a ready agent at open step *u* |
|---|---|
| Hazard 0.15 and 0.45 | constant |
| Surge, peak 5, width 1 | Gaussian above a 0.05 floor |
| Surge, peak 10, width 5 | Gaussian above a 0.05 floor |
| Decay, τ = 3 | front-loaded: falls from 1 toward the floor |
| Ramp, 8 steps | slow rise from the floor to 1, then flat |

Each schedule is crossed with the four failure delays, for 24 cells. The arms,
worlds, learners, seeds policy, and N01 entry-channel rule are those of N08,
with fresh seeds. A flat schedule must reproduce the N01 arms exactly.

## Registered decisions

For each observable, and for "either observable past its threshold":

| Disposition | Condition |
|---|---|
| `SUFFICIENT_SIGN_CONFIRMED` | No false actives |
| `SUFFICIENT_SIGN_FAILS` | At least one false active |
| `NOT_INFORMATIVE` | Fewer than 5 flagged cells or fewer than 5 inactive cells |

Sensitivity, the share of active cells flagged, is reported but not decided.
N08's two-sided accuracy and its no-learning control are reported as before.
An observable whose sufficient sign is confirmed, with the control reading
inactive, goes to the empirical stage as a one-sided indicator. A trend past
the threshold would then be read as a sign that learning acts on entry, and its
absence would not be read as evidence against it.

## Boundary

The executable must be committed before the development profile runs, and a
deterministic rerun must reproduce every artifact byte for byte. No post-result
change to schedules, thresholds, or rules is authorized.

The schedules and worlds are synthetic. The study reads no empirical rows, fits
nothing to data, selects no model, and changes no generic NABM API.
