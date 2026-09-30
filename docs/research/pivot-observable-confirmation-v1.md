# Pivot observable confirmation v1 (N08)

Date: 2026-10-01

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_observable_confirmation_v1.yaml`](../../experiments/configs/pivot_observable_confirmation_v1.yaml)

## Why this study exists

[N05](../../experiments/evidence/results/pivot_observable_discriminant_v1_20260930.summary.md)
found two data-observable candidates that separated every cell by whether
learned policy changed who entered:

- the **cohort persistence drop**, first-cohort persistence minus that of later
  cohorts; and
- the **outcome selection trend**, the rank correlation between an entrant's
  entry time and whether they persist.

Both need only entry timing and persistence. N05 ruled that neither is
confirmed until a third registered test, because the persistence drop had failed
on the N01 hazard cells. This is that test.

A post-hoc look at the recorded N05 rows, taken before this freeze, adds a
reason to include a control. The fixed rule, which does not learn, showed the
"inactive" pattern in every cell: later cohorts persisted less. The observables
therefore seem to read learning, not entry timing alone. This study registers
that as a check.

## Design

The study uses new entry schedules and fresh seeds:

- two constant hazards, admitting a ready agent with probability 0.35 or 0.7
  (N01 used 1, 0.5, and 0.25); and
- four attention surges, with peaks at open step 4 or 8 and widths 2 or 4, above
  a 0.05 floor (N05 used peaks 3, 6, and 9 with widths 1.5 and 3).

Each schedule is crossed with the four failure delays (0, 1, 2, and 4 steps),
for 24 cells. Everything else is the N05 design. It uses the same arms, worlds,
and learners, and N01's entry-channel rule. A flat schedule must reproduce the
N01 arms exactly. The contract rejects any schedule that repeats an N01 or N05
one.

## Registered tests

The thresholds are fixed from N05, at the midpoints of its separations:

| Observable | Predicts active when | Threshold |
|---|---|---:|
| Cohort persistence drop | below | 0.023175 |
| Outcome selection trend | above | 0.098417 |

Each observable is confirmed if it classifies every cell correctly. It holds if
at least 75% are correct, and fails otherwise. A joint rule, active only when
both say active, is scored the same way. The post-failure entry share at its
N01 threshold is reported as a comparator.

**No-learning control.** In every cell, the fixed rule's own observables must
read inactive. If they read active anywhere, the observables may be tracking
entry timing rather than learning.

**Use on data.** Only an observable that is confirmed in every cell, with the
control reading inactive everywhere, is carried to the empirical stage.

## Boundary

The executable must be committed before the development profile runs, and a
deterministic rerun must reproduce every artifact byte for byte. No post-result
change to schedules, thresholds, or rules is authorized.

The schedules and worlds are synthetic. The study reads no empirical rows, fits
nothing to data, selects no model, and changes no generic NABM API.
