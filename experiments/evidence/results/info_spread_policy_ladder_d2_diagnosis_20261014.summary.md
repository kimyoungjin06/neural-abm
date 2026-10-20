# Information-spread policy ladder: diagnosis of D2

Date: 2026-10-14

Status: `EXPLORATORY_DIAGNOSIS_ENSEMBLING_SUPPORTED_FORGETTING_PARTLY_FIT_BUDGET_CANDIDATE`

Question: in the D2 shift controls, why was the adaptive arm (L0) worse than
the four-donor table (T0-4)? D2 named two candidate causes:

- (a) ensembling;
- (b) learners that do not forget.

This diagnosis varies each on its own. Code: `aade5f6`, with the D1 learner
settings. It ran 80 cells:

- both shift controls (flat, flip);
- 2 regimes × 2 learner families;
- 10 cells each, with 16 replicates per cell.

It took 41 minutes on 36 CPU workers. No study cell was scored.

## Arms

| Arm | What it is |
|---|---|
| L0-full | The D2 adaptive learner. It refits on every visible outcome |
| L0-window | The same learner, refit only on outcomes visible in the last 12 steps |
| T0-1 | A trajectory table from one donor, i.e. the evidence of one run |
| T0-4 | The D2 table, averaging four donors |

Share precision is reported for two item cohorts: items born before the run's
shift step and items born after it. Run-wide precision mixes the two cohorts
in proportions that differ by arm. Intervals are 95% bootstrap intervals over
cells.

## (a) Ensembling: supported

Differences in share precision. "Pre" is the pre-shift cohort and "post" the
post-shift cohort.

| Group | T0-1 − T0-4, pre | L0-full − T0-1, pre | L0-full − T0-1, post |
|---|---|---|---|
| flat / A / linear | −0.011 [−0.029, +0.005] | −0.005 [−0.021, +0.014] | −0.014 [−0.039, +0.008] |
| flat / A / MLP | −0.025 [−0.056, +0.005] | −0.005 [−0.034, +0.029] | −0.011 [−0.035, +0.010] |
| flat / B / linear | −0.016 [−0.031, −0.003] | −0.014 [−0.030, +0.002] | −0.012 [−0.024, −0.002] |
| flat / B / MLP | −0.040 [−0.059, −0.023] | −0.010 [−0.028, +0.010] | −0.016 [−0.031, −0.001] |
| flip / A / linear | −0.012 [−0.030, +0.004] | −0.004 [−0.021, +0.015] | −0.002 [−0.020, +0.019] |
| flip / A / MLP | −0.027 [−0.058, +0.003] | −0.004 [−0.033, +0.030] | +0.003 [−0.067, +0.059] |
| flip / B / linear | −0.016 [−0.031, −0.002] | −0.014 [−0.032, +0.002] | −0.004 [−0.012, +0.005] |
| flip / B / MLP | −0.040 [−0.059, −0.023] | −0.009 [−0.027, +0.011] | −0.026 [−0.062, +0.010] |

- **Ensembling helps.** The single-donor table is below the four-donor table
  before the shift in all 8 groups. The interval excludes zero in the 4
  Regime B groups.
- **The closed loop alone does little.** At matched evidence, L0 against the
  single-donor table stays within ±0.026. Its interval includes zero in 14 of
  16 cohort contrasts. The exceptions are the post-shift cohort of the flat
  control in Regime B, at −0.012 and −0.016.
- **The ensemble's effective weight is closer to the truth.** Before the
  shift the four-donor table's credibility weight is about 3.5, against a
  true 3. Single learners sit at 4.1 to 4.5. Per-agent MLPs reach about 7.5,
  so they are overconfident, and averaging donors' probabilities shrinks
  them.
- **Most of D2's deficit is therefore ensembling, not the closed loop.**

## (b) No forgetting: partly supported

Effective credibility weight, by steps from the shift. The truth is +3 before
the shift and −3 after the flip.

| Flip group | Arm | [−4, 0) | [4, 8) | [8, 12) | [16, 20) |
|---|---|---:|---:|---:|---:|
| A / MLP | L0-full | 4.36 | 3.87 | 3.16 | 2.33 |
| A / MLP | L0-window | 4.93 | 3.45 | 1.49 | −1.41 |
| B / MLP | L0-full | 7.49 | 6.53 | 5.19 | 3.65 |
| B / MLP | L0-window | 7.66 | 5.82 | 2.36 | −2.15 |
| A / linear | L0-full | 4.18 | 4.06 | 3.87 | 3.67 |
| A / linear | L0-window | 4.23 | 3.79 | 3.47 | 3.16 |

- **The full learner tracks a shift slowly.** Pre-shift evidence dilutes the
  new relation.
- **For MLPs, forgetting tracks the flip and pays off after it.** Post-shift
  precision rises by 0.111 in A and 0.088 in B, with intervals excluding
  zero.
- **For linear learners, forgetting does little.** The gain is +0.012 in A
  and +0.003 in B.
- **Forgetting costs evidence.** Pre-shift precision falls by 0.004 to 0.054,
  most in Regime B, where each learner sees only its own outcomes.
- **Under the flat shift, forgetting does not help.** The change is −0.002 to
  −0.043 after the shift.

## (c) A new candidate: the fit budget

Linear learners barely leave the hand rule's weight of 4, with or without
forgetting. Before the shift they stay at 4.1 against a true 3. After the flip
they end at 3.2 to 3.9 against a true −3.

- Each refit starts from a zero residual and runs 60 Adam steps at a learning
  rate of 0.05. That caps how far a weight can move from the prior at
  roughly 3 logit units.
- The D1 settings were chosen for speed and recorded without justification.
  They are weaker than the Pivot ladder's: 300 epochs, learning rate 0.02,
  weight decay 1e-4, hidden width 16, and 300 tree rounds.
- A rerun of the flip control with the Pivot settings, on the same cells, is
  under way. It tests this candidate.

## What this does and does not say

- D2's adaptive arm lost to the four-donor table mostly through ensembling.
  At matched evidence, the closed loop alone made little difference in these
  controls.
- Tracking a shift needs both forgetting and a flexible learner, and
  forgetting has an evidence cost.
- The fit budget may have kept the learners near the prior, which would
  weaken every learning arm. This is not yet tested.
- None of this scores a study cell.
