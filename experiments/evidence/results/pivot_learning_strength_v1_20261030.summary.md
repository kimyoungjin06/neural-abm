# Pivot learning strength v1 (O39): result

Date: 2026-10-30

Status: `O39_THE_LOOP_FALLS_BELOW_MATERIAL_AT_THE_DATA_ALLOWED_STRENGTH`

Design: [pivot-learning-strength-v1.md](../../../docs/research/pivot-learning-strength-v1.md),
frozen at `42b682ef93e61c5d19c0e24631f7849dbde094c2`.
Machine-readable record:
[pivot_learning_strength_v1_20261030.summary.json](pivot_learning_strength_v1_20261030.summary.json).

## Execution

- **Host:** the workstation, with 32 workers and one BLAS thread per
  process.
- **Wall time:** 172 s for the development run and 224 s for the
  reproduction run, inside the quoted 3 to 20 minutes.
- **Reproduction:** both identity artifacts are byte identical.
- **Data read:** none. The data bound is a number in experiment 4's record.

### Checks

- **Study 2 runs:** at the tracked rate and delay 0, all 270 tracked
  replicate runs were reproduced exactly.
- **Study 2 comparisons:** at the same point, all 12 tracked comparison rows
  were reproduced exactly.
- **Frozen arm:** identical at delays 0 and 3 in all 90 runs.
- **Experiment 4's estimator:** at the tracked rate, all 378 compared values
  equal experiment 4's record; the largest difference is 0.
- **Convergence:** all 162 full-sample fits and every bootstrap fit
  converged.

## Calibration

The imitative arm's visible outcome gap, `S − F` in log-odds, by learning
rate and outcome delay:

| Rate | Baseline, 0 | Grants, 0 | Hype, 0 | Baseline, 3 | Grants, 3 | Hype, 3 |
|---|---:|---:|---:|---:|---:|---:|
| 1.5 | +9.54 | +6.95 | +7.07 | +11.15 | +10.08 | +10.06 |
| 0.75 | +6.53 | +6.38 | +6.16 | +7.79 | +7.84 | +8.67 |
| 0.375 | +4.34 | +4.04 | +3.67 | +5.35 | +4.62 | +5.93 |
| 0.1875 | +2.51 | +2.75 | +1.95 | undefined | +2.95 | +3.67 |
| 0.09375 | undefined | +1.60 | +0.93 | undefined | +1.51 | +2.16 |
| 0.046875 | undefined | +0.82 | +0.42 | undefined | +0.54 | +1.18 |
| 0.0234375 | undefined | +0.45 | +0.18 | undefined | +0.19 | +0.72 |
| 0.01171875 | undefined | +0.18 | +0.08 | undefined | −0.14 | +0.42 |

- **Undefined cells.** At low rates, baseline agents rarely pivot, and
  fewer than 10 entries follow a visible failure.
- **The data-allowed rate is 0.0234375 at both delays,** one sixty-fourth of
  the tracked rate.
  - It is the largest rate whose gap stays at or below the data's upper
    bound, 0.789, in every defined scenario.
  - At delay 0, the next rate up fails under grants (+0.82); at delay 3, it
    fails under hype (+1.18).

## The registered decision

At the data-allowed rate, the imitative arm's failed pivots against the
frozen arm, paired by replicate:

| Delay | Baseline | Class | Grants | Class |
|---|---|---|---|---|
| 3 (headline) | +0.0003 [+0.0000, +0.0008] | Below material, not detected | +0.0161 [+0.0117, +0.0208] | Below material, detected |
| 0 | +0.0003 [+0.0000, +0.0008] | Below material, not detected | +0.0222 [+0.0169, +0.0275] | Below material, detected |

**R5's loop falls below the material size of 0.05 in both scenarios, at
both delays.** At the tracked rate it is +0.281 at baseline and +0.289
under grants.

| Expectation | Result |
|---|---|
| X1: the gap rises with the rate | Held. Spearman 1.0 under grants and hype at both delays |
| X2: the data-allowed rate at delay 3 is at most 0.1875 | Held: 0.0234375 |
| X3: the baseline loop falls below material | Held |
| X4: the grant loop is still detected | Held: the low end is +0.0117 |

## The loop across the grid

Imitative minus frozen failed pivot rate:

| Rate | Baseline, 0 | Grants, 0 | Baseline, 3 | Grants, 3 |
|---|---:|---:|---:|---:|
| 1.5 | +0.281 | +0.289 | +0.138 | +0.301 |
| 0.75 | +0.232 | +0.257 | +0.099 | +0.272 |
| 0.375 | +0.104 | +0.212 | +0.036 | +0.234 |
| 0.1875 | +0.021 | +0.171 | +0.011 | +0.166 |
| 0.09375 | +0.004 | +0.109 | +0.003 | +0.074 |
| 0.046875 | +0.001 | +0.048 | +0.001 | +0.030 |
| 0.0234375 | +0.000 | +0.022 | +0.000 | +0.016 |
| 0.01171875 | +0.000 | +0.011 | +0.000 | +0.007 |

### Secondary calibration: the data's point estimate

With the point estimate, 0.228, as the bound:
- **Delay 0:** the data-allowed rate is 0.01171875. There the grant loop is
  +0.011.
- **Delay 3:** there is none. Under hype the gap is +0.42 even at the
  smallest rate.

## Post hoc, descriptive, not registered

- **How much stronger learning must be for a material loop.** Counting from
  the tracked rate down, the loop stays material:
  - under grants, down to 0.09375 at both delays, four times the
    data-allowed rate;
  - at baseline, down to 0.375 at delay 0 and 0.75 at delay 3, 16 and 32
    times that rate.
- **What learning adds at the data-allowed rate.** Under grants it adds
  more productive pivots than failed ones:
  - delay 0: +0.0328 productive against +0.0222 failed;
  - delay 3: +0.0217 against +0.0161.

  At the tracked rate it adds more failed ones: +0.289 failed against
  +0.156 productive at delay 0.
- **Delay alone, at the tracked rate.**
  - It halves the baseline loop, from +0.281 to +0.138.
  - It leaves the grant loop, +0.301, unchanged.
  - It turns the hype difference positive: +0.0506 [+0.0383, +0.0625].
- **Failure-only learning (R6, R7) also shrinks.**
  - At the data-allowed rate, hype failures fall by −0.0200
    [−0.0247, −0.0153] at delay 0 and by −0.0119 [−0.0164, −0.0081] at
    delay 3. At the tracked rate the falls are −0.1436 and −0.0583.
  - Under grants, productive pivots fall by −0.0047 and −0.0028, against
    −0.0694 and −0.0369 at the tracked rate.
- **The estimator still sees learning at the data-allowed rate.**
  - Its learning-attributable visible gap is +0.526 [+0.387, +0.677] under
    grants and +0.241 [+0.169, +0.320] under hype at delay 0.
  - At delay 3 it is +0.615 [+0.424, +0.837] and +0.661 [+0.404, +0.911].
  - The simulation has hundreds of entries per state. The data had 17
    after a visible success and 60 after a visible failure.
- **The entry-order gradient at that rate is the frozen arm's.**
  - Grants: −1.32 against −1.37 at delay 3. Hype: −0.54 against −0.74.
  - The falling gradient under grants comes from Study 2's structure, not
    from learning. The data show none, +0.010.

## What this means

- **R5 depends on learning far stronger than the coauthor data allow under
  this estimator.**
  - At a rate whose outcome dependence the data do not exclude, imitative
    learning adds about 2 failed pivots per 100 researchers under grants,
    with an upper bound below 3. At baseline it adds none that is detected.
  - On the grid, the smallest rate with a material loop is four times the
    data-allowed rate under grants, and 16 to 32 times at baseline.
- **The calibration is generous to Study 2.** It takes the data's upper
  bound. With the point estimate, the delay-3 calibration allows no grid
  rate at all.
- **This narrows R5. It does not refute it.** R5 is a property of Study 2
  at its hand-set rate. Whether real researchers' learning is as weak as
  the calibrated rate depends on putting coauthor topic quarters and Study 2
  steps on one scale, which this study does not justify.

## Boundary

- **Synthetic.** No empirical row is read; one recorded number from the
  discovery partition sets the bound.
- **One estimator, two designs.** The calibration puts Study 2's neighbor
  steps and OpenAlex coauthor quarters on one log-odds scale, and it takes
  the data bound at face value.
- **Not a calibration of Study 2.** Only the learning rate moves. The
  thresholds, network, and other hand-set constants stay as tracked.
- **Scope.** No LLM call, and no change to Study 2's code or to a generic
  NABM API.
