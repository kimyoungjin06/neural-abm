# Information-spread policy ladder: fit-budget check of the D2 diagnosis

Date: 2026-10-14

Status: `EXPLORATORY_FIT_BUDGET_PARTLY_CONFIRMED_ENSEMBLING_AND_FORGETTING_FINDINGS_HOLD`

The [D2 diagnosis](info_spread_policy_ladder_d2_diagnosis_20261014.summary.md)
raised a third candidate cause. The D1 learner settings may have kept the
learners near the prior:

| Setting | D1 | Pivot ladder |
|---|---|---|
| Epochs | 60 | 300 |
| Learning rate | 0.05 | 0.02 |
| Weight decay | 1e-3 | 1e-4 |
| MLP hidden width | 8 | 16 |

This check reran the flip control with the Pivot settings, on the same 10
cells per group, so each group is paired with its D1-settings result.

- Code: `95b7000`.
- Scale: 40 cells, 16 replicates each.
- Runtime: 2 h 45 min on 36 CPU workers.
- No study cell was scored.

## Effective credibility weight, by steps from the shift

The truth is +3 before the shift and −3 after it.

| Group | Arm | D1, [−4, 0) | D1, [16, 20) | Pivot, [−4, 0) | Pivot, [16, 20) |
|---|---|---:|---:|---:|---:|
| A / linear | L0-full | 4.18 | 3.67 | 4.15 | 3.07 |
| A / linear | L0-window | 4.23 | 3.16 | 4.23 | 1.96 |
| B / linear | L0-full | 4.08 | 3.61 | 4.54 | 3.38 |
| B / linear | L0-window | 4.09 | 3.02 | 4.45 | 2.24 |
| A / MLP | L0-full | 4.36 | 2.33 | 8.00 | 3.12 |
| A / MLP | T0-4 | 3.17 | 2.23 | 4.07 | 2.93 |
| B / MLP | L0-full | 7.49 | 3.65 | 14.30 | 8.07 |
| B / MLP | T0-4 | 3.50 | 2.59 | 5.20 | 4.17 |

## Contrasts in share precision, Pivot settings

Each cell gives the mean with its 95% bootstrap interval over cells.

| Group | T0-1 − T0-4, pre | L0-full − T0-1, pre | L0-full − T0-1, post | L0-window − L0-full, pre | L0-window − L0-full, post |
|---|---|---|---|---|---|
| A / linear | −0.026 [−0.053, −0.002] | +0.002 [−0.023, +0.031] | +0.004 [−0.027, +0.036] | −0.010 [−0.013, −0.007] | +0.028 [+0.016, +0.041] |
| A / MLP | −0.045 [−0.064, −0.023] | +0.007 [−0.022, +0.033] | −0.018 [−0.075, +0.042] | −0.011 [−0.014, −0.008] | +0.121 [+0.116, +0.126] |
| B / linear | −0.027 [−0.044, −0.012] | −0.013 [−0.030, +0.007] | −0.002 [−0.018, +0.013] | −0.040 [−0.044, −0.037] | +0.016 [+0.011, +0.021] |
| B / MLP | −0.048 [−0.067, −0.028] | −0.006 [−0.024, +0.013] | −0.029 [−0.068, +0.010] | −0.051 [−0.054, −0.048] | +0.080 [+0.076, +0.085] |

## Reading

- **(c) Fit budget: partly confirmed.**
  - With the Pivot settings, linear learners move further after the flip.
    Post-shift precision rises for every arm.
  - Linear learners still sit at 4.1 to 4.5 before the shift, against a true
    3.
  - The larger budget makes MLPs more overconfident. Before the shift their
    weight is 8.0 in A and 14.3 in B, against 4.4 and 7.5 with the D1
    settings. Their pre-shift precision falls.
- **(a) Ensembling: holds, and grows.** The single-donor table is below the
  four-donor table before the shift in all four groups, each interval
  excluding zero.
- **The closed loop alone is still indistinguishable.** At matched evidence,
  every interval of L0 against the single-donor table covers zero (8 of 8).
- **(b) Forgetting: holds, and widens.** The forgetting learner raises
  post-shift precision in all four groups, now including linear learners. It
  costs 0.010 to 0.051 of pre-shift precision.
- **Untested.** Linear learners stay near the prior before the shift even with
  the larger budget. One candidate is selective labels: an outcome is seen
  only for a shared item, and the exploration floor is 2%.

## What this says for the redesign

1. **D2's loss to the four-donor table was mostly ensembling.**
   - Two different questions need two comparators.
   - The closed-loop question needs a comparator at matched evidence, the
     single-donor table.
   - "Can any table written before the run match NABM?" is a different
     question, and in these controls the four-donor table answered yes.
2. **A closed-loop advantage under a shift needs forgetting and enough fit
   budget.** Both have costs, so they are design choices to register, not
   defaults.
3. **High variance is itself a finding.** Per-agent and high-capacity learners
   are overconfident, and an ensemble corrects that. Regime B rests on
   per-agent learning, so its learners need regularization or partial
   pooling before D3.
