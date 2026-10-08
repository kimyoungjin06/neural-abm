# Pivot empirical ladder recovery v2 (N11a v2)

Date: 2026-10-08

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_empirical_ladder_recovery_v2.yaml`](../../experiments/configs/pivot_empirical_ladder_recovery_v2.yaml)

Amends: [v1](pivot-empirical-ladder-recovery-v1.md) (`af43b16`, recorded
`d114272`). Its contract, script, and result are bound by hash.

## Why v2

[v1](../../experiments/evidence/results/pivot_empirical_ladder_recovery_v1_20261008.summary.md)
passed identification with one limit. In W6, entrants were drawn from the
start toward the episode's eventual success relation, without learning. That
anticipation also produced an NABM win in 8 of 8 replicates.

The two mechanisms differ in timing:

- **Learning** can act only once evidence is visible, from the fifth quarter
  after the first entries.
- **Anticipation** acts from the first quarter.

## What changes

| | v1 | v2 |
|---|---|---|
| Ladder | B1–B6 | Adds **B7, anticipatory**: B2 plus the episode's eventual success relation, applied from the first quarter |
| NABM win | Beats the chosen non-learning model and B2 | Must also beat B7, with a positive mean and a positive 5% bound |
| W6, anticipation | Reported | **Must not falsely recover** |
| W7, learning with anticipation | — | Reported |
| Seeds | 20261101 | 20261102, fresh |

B7 is fitted on the early-success labels of every entrant in the episode.
That relation is available to an analyst afterwards, but not to agents at the
time of entry. Learning must beat B7. Its timing, tied to the evidence, is
what separates it from anticipation.

The episode structure, mechanisms, learners, fitting, bootstrap, profiles, and
reported sign test are v1's, checked at load.

## Decision

| Disposition | Condition |
|---|---|
| `IDENTIFICATION_PASSES` | W1 and W2 at least 7 of 8 wins; W3, W4, W5, and W6 at most 1 of 8 |
| `IDENTIFICATION_FAILS` | Otherwise. The empirical claim must then name anticipatory selection as an equal alternative |

## Boundary

The executable must be committed before the development profile runs, and a
rerun must be byte identical. The study is synthetic, reads no empirical
row, and selects no model for data.
