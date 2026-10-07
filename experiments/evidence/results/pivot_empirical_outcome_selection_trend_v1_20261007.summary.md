# Pivot empirical outcome-selection trend v1: stage T

Date: 2026-10-07

Status: `STAGE_T_THRESHOLD_TRANSFERRED_EMPIRICAL_STAGES_NOT_AUTHORIZED`

Precommit: `49f5345`. Design:
[`pivot-empirical-outcome-selection-trend-v1.md`](../../../docs/research/pivot-empirical-outcome-selection-trend-v1.md)

## What ran

Stage T recomputed the outcome selection trend from the stored runs of N05,
N08, and N09 (4,608 runs each). Each run's 18 full-horizon entry steps were cut
into three equal bins, the clock of three yearly response cohorts. Every run's
fine-clock value was rebuilt from its stored cohorts first, and all 13,824
matched. A deterministic rerun reproduced both artifacts byte for byte. No
empirical row was read.

## Registered result

The threshold **transferred**.

| | Value |
|---|---|
| Fine-clock threshold | 0.098417 |
| Coarse threshold, the largest inactive cell | 0.105635 (N08 surge, peak 8, width 2, failure lag 2) |
| **Data threshold** | **0.105635** |
| Largest no-learning control value | −0.031 |
| Cells | 72: 33 active, 39 inactive |
| Cells undefined on the coarse clock | 16, all inactive (15 saturated) |

## Reported, not decided

| | Fine clock | Coarse clock |
|---|---|---|
| Active cells past the threshold | 29 of 33 | **12 of 33** |
| Smallest active minus largest inactive | | −0.130 |

## Reading

This reading is post hoc and descriptive.

The three-level clock removes most of the signal when entry is concentrated
early. Then most entrants fall in the first bin, and the coarse value is 2% to
30% of the fine one. The 12 active cells that still pass are low-hazard
clocks (0.15, 0.35) and late-peak clocks (peaks 8 and 9). The 16 undefined
cells are mostly saturated: everyone enters in one bin. An empirical episode
like that would be invalid.

The coarse threshold is the largest inactive value, so the synthetic cells
show no false actives by construction. That is an in-sample property, not
the out-of-sample confirmation N09 gave the fine clock.

## What this means for the empirical stages

- A positive empirical reading would still be a sign.
- On this clock, the indicator caught about one active cell in three. A result
  that is not positive is even less informative than registered.
- A finer empirical entry clock, such as publication-date quarters within the
  three response years, could recover sensitivity. It would need a new
  versioned precommit, still before any empirical row is read.

S1, S2, and S3 remain unauthorized. No row value was decoded, and no G1, G4,
or model-selection row was read.
