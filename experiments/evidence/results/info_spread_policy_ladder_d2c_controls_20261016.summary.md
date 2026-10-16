# Information-spread policy ladder: stage D2c controls

Date: 2026-10-16

Status: `D2C_ADEQUATE_REVISED_DESIGN_READY_FOR_D3`

Specification: "D2c specification" in
[the design brief](../../../docs/research/info-spread-policy-ladder-design-v1.md),
fixed before the run. Code: `d6c5d44`. No study cell was scored.

## What ran

D2c used D2b's controls with the five changes the user approved on
2026-10-16:

- exact (Newton) linear learners, pooled and per agent;
- one training row per agent and item under public fact-checks;
- pooled MLPs on GPU 2;
- 24 workers with per-task progress;
- the group rule alone.

The run covered 240 cells:

- 3 controls;
- 2 regimes × 2 families;
- 20 cells each, with 16 replicates per cell.

It took 85 minutes. A full-load pilot had put the estimate at 40–80 minutes.

## Result: group rule

The table shows the mean cell Δ in share precision, with its cell-resampled
95% interval.

| Control / group | Pair | Mean cell Δ | Interval | Material |
|---|---|---:|---|---|
| NEG, all 4 groups | T0-1 vs T0-1′, T0-4 vs T0-4′ | −0.007 to +0.007 | All cover 0 | 0 of 8 |
| POS-sign / A / linear_exact | L0 − T0-1 | +0.137 | [+0.118, +0.156] | Yes |
| POS-sign / A / MLP | L0 − T0-1 | +0.123 | [+0.108, +0.138] | Yes |
| POS-sign / B / partial_linear_exact | L0 − T0-1 | +0.127 | [+0.108, +0.145] | Yes |
| POS-sign / B / partial_MLP | L0 − T0-1 | +0.131 | [+0.116, +0.147] | Yes |
| POS-sign / all 4 groups | L0 − T0-4 | +0.082 to +0.096 | All exclude 0 | 4 of 4 |

In POS-sign runs whose sign is −1, the exact pooled linear learner reaches a
precision of 0.594. In D2b the step-capped learner reached 0.26.

POS-flip:

| Pair | Mean cell Δ | Before the shift | After the shift |
|---|---:|---:|---:|
| L0-window − L0 | +0.031 to +0.051 | −0.018 to −0.002 | +0.101 to +0.123 |
| L0 − T0-1 | +0.010 to +0.029 | | |
| L0 − T0-4 | −0.011 to +0.002 | −0.015 to −0.033 | Up to +0.017 |

- The L0 − T0-1 difference is material in 3 of 4 groups.
- The L0 − T0-4 difference is material only in A/MLP.

## Reading against the specification

- **The revised design is adequate.**
  - No NEG pair is material.
  - L0 − T0-1 is material and positive in POS-sign in all four groups.
- **The changes worked.**
  - The exact linear learner removed D2b's single failure.
  - Removing duplicate rows and moving pooled MLPs to the GPU cut the run
    from 10.3 hours to 85 minutes.
- **Two further readings, still on controls only.**
  - When a run carries information that donors lack, as with its own sign,
    the closed loop beats even the four-donor table.
  - When the information arrives as a shift, tracking it needs forgetting.
    Without forgetting, the four-donor table's ensembling before the shift
    offsets the closed loop's gain after it.

## Next

D3 can freeze the study with these choices: the learners, observability,
comparators, and rule tested here. That needs the user's go-ahead. Until
then, no study cell is run.
