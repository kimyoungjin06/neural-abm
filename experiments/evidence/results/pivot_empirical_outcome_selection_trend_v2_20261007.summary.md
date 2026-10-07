# Pivot empirical outcome-selection trend v2: stage T

Date: 2026-10-07

Status: `STAGE_T_THRESHOLD_TRANSFERRED_ON_TWELVE_LEVELS_EMPIRICAL_STAGES_NOT_AUTHORIZED`

Precommit: `58c87f4`, amending v1 (`49f5345`). Design:
[`pivot-empirical-outcome-selection-trend-v2.md`](../../../docs/research/pivot-empirical-outcome-selection-trend-v2.md)

## What ran

Stage T reran the v1 transfer rule with each synthetic run's 18 full-horizon
entry steps cut into twelve bins. That is the clock of publication-date
quarters across three response years. All 13,824 fine-clock values were
rebuilt exactly, and a rerun reproduced both artifacts byte for byte. No
empirical row was read.

## Registered result

The threshold **transferred**.

| | Value |
|---|---|
| Twelve-level coarse threshold, the largest inactive cell | 0.0918 (N08 surge, peak 8, width 2, failure lag 2) |
| **Data threshold** | **0.098417**, the fine-clock threshold, because the coarse one is lower |
| Largest no-learning control value | −0.051 |
| Cells undefined on twelve levels | 0 of 72 |

## Reported, not decided

| Clock | Active cells past the threshold |
|---|---|
| Synthetic, 18 steps | 29 of 33 |
| v1, 3 years | 12 of 33 |
| **v2, 12 quarters** | **29 of 33** |

The four misses include N09's weak hazard 0.15, failure lag 4 cell, which the
fine clock also missed.

## Reading

Twelve levels recover the sensitivity lost on three. The data threshold is the
one N09 confirmed out of sample on the fine clock, not an in-sample maximum.

Three things stage T cannot test, and S1 will report them:

- the share of January 1 entry dates;
- how accurate the publication dates are;
- the remaining gap between "returned within twelve quarters" and the
  synthetic "not exited".

S1, S2, and S3 remain unauthorized. No row value was decoded, and no G1, G4,
or model-selection row was read.
