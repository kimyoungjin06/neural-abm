# Information-spread policy ladder: stage D2 controls

Date: 2026-10-14

Status: `D2_RULE_FAILS_THE_POWER_CRITERION_POSITIVE_CONTROLS_NOT_POSITIVE`

Specification: the "D2 specification" section of
[info-spread-policy-ladder-design-v1.md](../../../docs/research/info-spread-policy-ladder-design-v1.md),
committed in `ccac2fd` before the run. No study cell was scored.

## What ran

The run covered 240 cells, each with its own world seed:

- 3 controls (NEG, POS-flat, POS-flip);
- 2 regimes × 2 learner families;
- 20 cells each, with 16 paired replicates per cell.

It took 78 minutes on 36 CPU workers of the GPU server. Δ is first minus
second:

- in the POS controls, adaptive L0 minus the four-donor trajectory table;
- in NEG, table 1 minus table 2.

## Result

| Control / regime / family | Material (R1 and R2) | R1 alone | R2 alone | Mean Δ share precision [cell q10, q90] | Replicates with Δ < 0 | Mean Δ false reach |
|---|---:|---:|---:|---|---:|---:|
| NEG / A / linear | 1/20 | 2 | 3 | −0.000 [−0.018, +0.018] | 0.55 | −1.1 |
| NEG / A / MLP | 0/20 | 3 | 2 | +0.002 [−0.022, +0.026] | 0.50 | −2.1 |
| NEG / B / linear | 1/20 | 1 | 2 | +0.002 [−0.013, +0.020] | 0.45 | −3.8 |
| NEG / B / MLP | 0/20 | 2 | 4 | +0.002 [−0.018, +0.021] | 0.51 | −2.3 |
| POS-flat / A / linear | 1/20 | 2 | 10 | −0.012 [−0.032, +0.011] | 0.65 | +6.9 |
| POS-flat / A / MLP | 3/20 | 3 | 12 | −0.027 [−0.050, −0.008] | 0.76 | +23.3 |
| POS-flat / B / linear | 4/20 | 5 | 15 | −0.024 [−0.038, −0.008] | 0.83 | +28.3 |
| POS-flat / B / MLP | 10/20 | 14 | 15 | −0.045 [−0.061, −0.025] | 0.92 | +54.7 |
| POS-flip / A / linear | 1/20 | 3 | 9 | −0.010 [−0.044, +0.017] | 0.61 | +6.3 |
| POS-flip / A / MLP | 5/20 | 6 | 11 | −0.020 [−0.054, +0.011] | 0.71 | +24.9 |
| POS-flip / B / linear | 3/20 | 5 | 14 | −0.021 [−0.043, +0.002] | 0.74 | +28.8 |
| POS-flip / B / MLP | 7/20 | 9 | 15 | −0.035 [−0.062, −0.009] | 0.79 | +55.0 |

Counts of 0 of 20 bound the cell rate below about 14% (one-sided 95%).

## Reading against the specification

- **False positives: within the criterion.** NEG is material in 0 or 1 of 20
  cells for every regime and family.
  - R2 alone flags 2 to 4 of 20, i.e. 10–20%.
  - With 16 replicates, the bootstrap interval for R2 is therefore too
    liberal to stand alone.
- **Power: the criterion fails.** POS-flip is material in 1 to 7 of 20 cells,
  against the 16 required. This holds in every regime and family.
- **The positive controls were not positive in the intended sense.**
  - The adaptive arm does differ from the table.
  - But it is worse: lower share precision and larger false reach in both
    shift controls.
  - The gap is largest for per-agent MLPs, where L0 is lower in 79–92% of
    replicates.

## Why the controls did not behave as positives

These are candidate explanations. No factor was varied on its own.

- **The table pools more than the run.**
  - T0-traj averages four donors' policies, so it carries four runs of
    evidence and the variance reduction of an ensemble.
  - The comparison therefore mixes the closed loop with ensembling. It does
    not isolate the closed loop.
- **The learners cannot track a shift quickly.**
  - The registered learners refit on all visible outcomes, with no
    forgetting, as in the Pivot ladder.
  - Fact-checks arrive four steps after an item enters, and refits happen
    every four steps.
  - After a shift, pre-shift evidence dilutes the new relation, so the
    closed loop helps little within the run.

## What this does and does not say

The proposed rule, at 16 replicates per cell, misses closed-loop differences
of the size these controls produce. The controls did not provide a case where
the closed loop is known to help.

In these synthetic shift worlds, a four-donor table matched or beat the
adaptive learners on share precision. That is a result about the controls,
not about the study cells, which were not scored.

D3 is not frozen. The rule and the controls need revision, and the choice is
the user's.
