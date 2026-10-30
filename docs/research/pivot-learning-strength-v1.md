# Pivot learning strength v1: Study 2's loop at a strength the data allow

Date: 2026-10-30

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_learning_strength_v1.yaml`](../../experiments/configs/pivot_learning_strength_v1.yaml)

This study answers O39 of
[consolidation v13](pivot-policy-ladder-consolidation-v13.md). The user
directed it on 2026-10-30.

## Question

[Study 2](../case-studies/researcher-pivot/README.md#r5--imitative-learning-amplifies-baseline-and-grant-pivoting)
reports that imitative outcome learning raises failed pivots against the
frozen arm:
- +0.281 [0.235, 0.322] at baseline;
- +0.289 [0.268, 0.308] under seed grants;
- −0.009 [−0.024, 0.007] under hype.

The case study calls the first two a self-reinforcing outcome-learning loop
(R5).

[Experiment 4](pivot-peer-outcome-contrast-v1.md) measured Study 2's arms
and the OpenAlex coauthor data with one estimator. Post hoc, the imitative
arm's visible outcome gap was at least 6.95 log-odds, against an upper bound
of 0.79 in the data (C93). Study 2's learning rate of 1.5 makes an agent
react to a neighbor's outcome far more strongly than coauthors' entries react
to it in those data.

**Does the loop survive when the learning rate is cut until the same
estimator finds a gap the data allow? Does it survive when outcomes become
visible only after a delay?**

## Design

### Study 2 over a grid of learning rates

Study 2 runs unchanged through experiment 4's two slots, imported from that
study's frozen script by hash.

- **Rates.** The tracked rate 1.5 halved seven times:
  1.5, 0.75, 0.375, 0.1875, 0.09375, 0.046875, 0.0234375, 0.01171875.
  Every value is exact in binary.
- **What the rate does.** It is the step size of the vicarious cross-entropy
  update. It scales both the pull toward observed outcomes and the pull of
  the prior anchor; the anchor's strength, 0.25, is unchanged. Nothing else
  in Study 2's config changes.
- **Arms.**
  - Imitative and cautionary at every rate.
  - Frozen once per scenario and delay, because it never learns.
- **Scope.** Three scenarios, 30 replicates, and outcome delays 0 and 3, all
  as in experiment 4.
- **Pairing.** Study 2's component-keyed seed streams are unchanged. Every
  rate and arm therefore runs on the same populations, networks, and noise.

### The estimator

Experiment 4's estimator is imported unchanged with its contract settings:
- the exposure model, the state model, and the entry-order model;
- the 10-event rule for undefined contrasts;
- its 2,000 bootstrap draws.

The draws for a scenario and delay are experiment 4's, shared by every rate.
Rates are therefore compared on the same replicate draws.

It measures the frozen arm and the imitative arm at every rate.

### Calibration

- **The quantity.** The imitative arm's visible outcome gap, `S − F`. This
  is the quantity behind C93, and it is defined at both delays. It reads
  imitative learning directly: entry after a neighbor's visible success
  against entry after a visible failure.
- **The bound.** The data's upper 95% bound, 0.789381, read from experiment
  4's record. It is the largest gap the data do not exclude, so it is the
  most generous reading for Study 2. The data's point estimate, 0.228246, is
  reported as a secondary calibration and decides nothing.
- **An allowed rate.** The imitative gap, from the full sample, is defined in
  at least one scenario. It is at or below the bound in every scenario where
  it is defined.
- **The data-allowed rate.** For each delay, the largest grid rate such that
  it and every smaller rate are allowed. If the smallest rate is not
  allowed, there is none.

The calibration takes the data bound at face value. The bound comes from
coauthors and topic quarters, and the gaps from Study 2 neighbors and agent
steps, on one log-odds scale. The study does not map one scale onto the
other.

### Outcome

At the data-allowed rate, the outcome is R5's own comparison:
- the imitative arm's failed pivot rate minus the frozen arm's, paired by
  replicate;
- with Study 2's own interval, a 10,000-resample percentile bootstrap of the
  paired mean.

Also reported, for every rate, scenario, and delay:
- productive pivot deltas;
- the cautionary arm's deltas, for R6 and R7;
- each arm's pivot, failed, and productive rates and productive share;
- every quantity of experiment 4's estimator, including the entry-order
  gradient.

## Checks

These come before any estimate. A failure stops the run with no output.
- **Study 2 runs.** At the tracked rate and delay 0, the rerun reproduces
  all 270 tracked replicate runs exactly.
- **Study 2 comparisons.** At the same point, it reproduces all 12 tracked
  comparison rows exactly.
- **Frozen arm.** It is identical at delays 0 and 3.

These come before the disposition. A failure writes the rows with status
`O39_INVALID_THE_E4_ESTIMATOR_IS_NOT_REPRODUCED` and no disposition.
- **The estimator.** At the tracked rate, the rows of the frozen and
  imitative arms equal experiment 4's record within 1e-5.
- **Convergence.** Every full-sample fit converges.

## Decision

The loop is read at baseline and under grants, where R5 claims it. The
headline delay is 3. O39 asks about delayed visibility, and in the data an
outcome is visible only five quarters after entry. Delay 0 gets the same
rule as a secondary disposition.

**Material.** A failed-pivot delta of 0.05:
- six of 120 researchers;
- about a sixth of R5's registered deltas;
- the size of the smallest effect the case study reports as a finding
  (R6's productive-pivot delta, −0.050).

| Class | 95% interval |
|---|---|
| `MATERIAL` | Low at or above 0.05 |
| `BELOW_MATERIAL_DETECTED` | High below 0.05, low above 0 |
| `BELOW_MATERIAL_NOT_DETECTED` | High below 0.05, low at or below 0 |
| `UNRESOLVED` | Otherwise |

| Baseline | Grants | Status |
|---|---|---|
| No data-allowed rate | | `O39_NOT_CALIBRATED_THE_SMALLEST_RATE_EXCEEDS_THE_DATA_BOUND` |
| Material | Material | `O39_THE_LOOP_HOLDS_AT_THE_DATA_ALLOWED_STRENGTH` |
| Below material | Below material | `O39_THE_LOOP_FALLS_BELOW_MATERIAL_AT_THE_DATA_ALLOWED_STRENGTH` |
| Below material | Material | `O39_THE_LOOP_HOLDS_ONLY_UNDER_GRANTS` |
| Material | Below material | `O39_THE_LOOP_HOLDS_ONLY_AT_BASELINE` |
| Otherwise | | `O39_UNRESOLVED_AT_THE_DATA_ALLOWED_STRENGTH` |

**What a disposition says.** Whether Study 2's registered loop is still
there when its learning is no stronger than the coauthor data allow under one
estimator. It does not calibrate Study 2, identify learning in real
researchers, or say which rule they use.

## Expectations, stated before the run

These are recorded and do not change the disposition.

| Id | Statement |
|---|---|
| X1 | The imitative visible gap rises with the rate: Spearman at least 0.9 under grants and hype at both delays |
| X2 | The data-allowed rate at delay 3 is at most 0.1875, at least eight times below the tracked rate |
| X3 | At that rate the baseline loop falls below material. Frozen baseline agents pivot rarely (4.9%) and leave few outcomes to learn from |
| X4 | At that rate the grant loop is still detected: the interval's low end is above 0 |

## Execution

- **Host.** The workstation, with 32 workers, for both runs.
- **Threads.** `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
  `MKL_NUM_THREADS` are set to 1 and checked. Each simulation worker runs
  torch on one thread.
- **Runtime.** 3 to 20 minutes per run. A timing pilot on a scratch seed,
  which printed no estimate, measured 6 to 8 s per 30-replicate simulation
  task and 2.6 ms per bootstrap fit: 102 tasks and 324,000 fits, about 26
  CPU minutes before worker start-up.
- **Identity.** `o39_summary.json` and `o39_receipt.json` must be byte
  identical between the development and reproduction runs.

## Boundary

- **Data.** No empirical row is read. The data bound is a number in
  experiment 4's recorded result.
- **Study 2.** Its code is unchanged. Only the learning rate and the
  experiment-4 slots differ from the tracked study.
- **Scope.** No LLM call and no change to a generic NABM API. No model is
  selected.
