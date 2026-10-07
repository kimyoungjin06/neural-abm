# Pivot one-sided observable test v1 result (N09)

Date: 2026-10-07

Status: `SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED`

## Result

**The one-sided claim held for both observables.** The N08 thresholds were kept
unchanged. On six entry-schedule shapes new to the ladder, each observable
flagged 11 cells as active, and all 11 were active. Each caught 11 of the 13
active cells, a sensitivity of 85%.

**Only the outcome selection trend passed the no-learning control.** The fixed
rule does not learn:

- Its **outcome selection trend** never passed the threshold, staying negative
  in all 24 cells.
- Its **cohort persistence drop** passed the threshold in 8 cells: every cell
  of the lowest constant hazard (0.15) and of the narrow surge (peak 5,
  width 1). Under those schedules, later cohorts barely persist less even
  without learning, so a small drop does not show that learning acts on entry.

Under the registered rule, applied to each observable with its own control
count, only the **outcome selection trend** goes to the empirical stage, as a
one-sided indicator. The persistence drop does not. Read together, the combined
control disposition fails, because of the drop.

**Two-sided, both observables again mostly held**, at 22 of 24 cells with two
missed active cells each. The post-failure entry share comparator classified 19
of 24, with four false actives and one miss.

The run used `c0a8b1a`: 4,608 runs over 6 schedules, 4 failure delays, 3 fit
worlds, and 4 arms, with 16 replicates of 256 agents over 24 steps. A flat
schedule reproduced the N01 arms exactly, and a second execution reproduced all
four artifacts byte for byte. The cells split 13 active and 11 inactive.

## Cells

Bold values are past the threshold. The trend's threshold is above 0.098417, and
the drop's is below 0.023175. "Pairs" counts the family-geometry pairs that
passed N01's entry-channel rule.

| Schedule | Delay | Status | Pairs | Outcome trend | Persistence drop | Fixed-rule trend | Fixed-rule drop |
|---|---:|---|---:|---:|---:|---:|---:|
| hazard 0.15 | 0 | **active** | 2 | **+0.290** | **−0.369** | −0.053 | **+0.004** |
| hazard 0.15 | 1 | **active** | 4 | **+0.308** | **−0.118** | −0.053 | **+0.004** |
| hazard 0.15 | 2 | **active** | 3 | **+0.201** | **+0.010** | −0.053 | **+0.004** |
| hazard 0.15 | 4 | **active** | 2 | −0.176 | +0.507 | −0.053 | **+0.004** |
| hazard 0.45 | 0 | **active** | 2 | **+0.348** | **−0.375** | −0.116 | +0.064 |
| hazard 0.45 | 1 | **active** | 1 | +0.052 | +0.026 | −0.116 | +0.064 |
| hazard 0.45 | 2 | absent | 0 | −0.036 | +0.083 | −0.116 | +0.064 |
| hazard 0.45 | 4 | absent | 0 | −0.498 | +0.575 | −0.116 | +0.064 |
| surge 5, 1 | 0 | **active** | 1 | **+0.218** | **−0.282** | −0.075 | **+0.020** |
| surge 5, 1 | 1 | **active** | 2 | **+0.122** | **+0.017** | −0.075 | **+0.020** |
| surge 5, 1 | 2 | absent | 0 | −0.051 | +0.328 | −0.075 | **+0.020** |
| surge 5, 1 | 4 | saturated | 0 | −0.330 | +0.456 | −0.075 | **+0.020** |
| surge 10, 5 | 0 | **active** | 2 | **+0.279** | **−0.344** | −0.122 | +0.051 |
| surge 10, 5 | 1 | **active** | 1 | **+0.110** | **+0.014** | −0.122 | +0.051 |
| surge 10, 5 | 2 | absent | 0 | −0.024 | +0.109 | −0.122 | +0.051 |
| surge 10, 5 | 4 | saturated | 0 | −0.414 | +0.536 | −0.122 | +0.051 |
| decay τ = 3 | 0 | **active** | 2 | **+0.223** | **−0.326** | −0.156 | +0.208 |
| decay τ = 3 | 1 | absent | 0 | −0.127 | +0.141 | −0.156 | +0.208 |
| decay τ = 3 | 2 | absent | 0 | −0.146 | +0.155 | −0.156 | +0.208 |
| decay τ = 3 | 4 | saturated | 0 | −0.730 | +0.658 | −0.156 | +0.208 |
| ramp 8 | 0 | **active** | 2 | **+0.241** | **−0.344** | −0.126 | +0.033 |
| ramp 8 | 1 | **active** | 1 | **+0.122** | **+0.004** | −0.126 | +0.033 |
| ramp 8 | 2 | absent | 0 | −0.021 | +0.112 | −0.126 | +0.033 |
| ramp 8 | 4 | saturated | 0 | −0.396 | +0.519 | −0.126 | +0.033 |

## Post-hoc checks

These checks were not registered, and nothing new was run.

**N08's "only weak channels are missed" reading did not hold.** The hazard 0.15
cell with a four-step delay had two passing pairs and was missed by both
observables. The other miss, hazard 0.45 with a one-step delay, had one passing
pair.

**There are still no false actives.** Across N05, N08, and N09, 144 predictions
were made by the two observables, and none called an inactive cell active.

## What this means for the empirical stage

- **The empirical target is a one-sided indicator.** It is the rank
  correlation, within a pivot episode, between when researchers enter and
  whether they persist. A clearly positive correlation would indicate that
  learning changes who enters. No correlation, or a negative one, is not
  evidence against it.
- **The persistence drop is not carried.** A shrinking cohort-persistence
  decline can arise from entry timing alone.
- **The threshold is on the model's scale.** 0.098 is defined for these
  worlds' persistence horizon. Its transfer to data requires the empirical
  persistence definition and horizon to be fixed first, in the empirical
  preregistration.

## Boundary

The schedules and worlds are synthetic and stylized. The study read no empirical
rows, fitted nothing to data, selected no model, and changed no generic NABM API.
