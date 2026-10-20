# Information-spread policy ladder: stage D2b controls

Date: 2026-10-15

Status: `D2B_NOT_ADEQUATE_GROUP_RULE_WORKS_POOLED_LINEAR_CANNOT_REVERSE`

Specification: "D2b specification" in
[the design brief](../../../docs/research/info-spread-policy-ladder-design-v1.md),
fixed before the run. Code: `f0ea165`. No study cell was scored.

## What ran

The run covered 240 cells:

- 3 controls (NEG, POS-sign, POS-flip);
- 2 regimes × 2 learner families;
- 20 cells each, with 16 replicates per cell.

Learners trained on public fact-checks, with the Pivot settings. Regime B
used partial pooling.

The run took 10.3 hours on 40 CPU workers, against a quoted hour. Pooled-MLP
training took 95% of the run time, for two reasons:

- Under public fact-checks, every re-evaluated decline became a training row.
- The Pivot settings refit 300 epochs from scratch at every refit.

## Result: group rule

The table shows the mean cell Δ in share precision, with its cell-resampled
95% interval.

| Control / group | Pair | Mean cell Δ | Interval | Material |
|---|---|---:|---|---|
| NEG, all 4 groups | T0-1 vs T0-1′, T0-4 vs T0-4′ | −0.008 to −0.001 | All intervals cover 0 | 0 of 8 |
| POS-sign / A / linear | L0 − T0-1 | +0.000 | [−0.014, +0.015] | No |
| POS-sign / A / MLP | L0 − T0-1 | +0.142 | [+0.129, +0.156] | Yes |
| POS-sign / B / partial linear | L0 − T0-1 | +0.120 | [+0.108, +0.133] | Yes |
| POS-sign / B / partial MLP | L0 − T0-1 | +0.148 | [+0.136, +0.160] | Yes |
| POS-sign / A / linear | L0 − T0-4 | −0.034 | [−0.045, −0.022] | Yes, negative |
| POS-sign / other 3 groups | L0 − T0-4 | +0.083 to +0.098 | All exclude 0 | Yes |

POS-sign precision split by each run's own sign:

| Group | Sign | L0 | T0-1 | T0-4 |
|---|---|---:|---:|---:|
| A / linear | −1 | 0.260 | 0.279 | 0.283 |
| A / linear | +1 | 0.647 | 0.627 | 0.691 |
| A / MLP | −1 | 0.576 | 0.388 | 0.436 |
| B / partial linear | −1 | 0.542 | 0.380 | 0.381 |
| B / partial MLP | −1 | 0.585 | 0.397 | 0.443 |

POS-flip:

- Forgetting tracks the flip in every group. Against L0, post-shift
  precision rises by 0.059 to 0.109.
- Where L0 − T0-4 is material, the four-donor table is ahead
  (ensembling).

## Reading against the specification

- **NEG: passes.** The group rule calls no NEG pair material.
- **POS-sign: fails in one group, so the design is not adequate.**
  - The adaptive arm beats both tables strongly in three groups.
  - In A/linear it does not. In runs whose sign is −1, its precision is
    0.260, against 0.54 to 0.59 in the other groups. It never learns the
    reversed relation.
- **Why A/linear fails: the fit budget again.**
  - From a zero residual, 300 Adam steps at learning rate 0.02 move a weight
    by at most about 6 logits.
  - Reversing the prior's +4 to −3 needs 7.
  - Partial pooling stacks two linear stages, and the MLP has more
    parameters, so both can reverse.
  - The failure is in the learner's optimizer, not the closed-loop rule.
- **The per-cell rules are unsuitable.**
  - In a POS-sign cell the replicates have random signs, so the effect is
    large in about half of them and near zero in the rest.
  - R1 needs 12 of 16 and rarely holds. R1 alone also flags up to 7 of 20
    NEG cells.
  - R2, the early-shock slope, does not address a sign effect.
  - The group rule is the one that works.

## What D3 needs before it can freeze

1. **A linear learner that is not step-capped.** For example, an exact
   ridge-logistic fit (Newton or IRLS) on the prior offset. The Adam version
   stays available for Pivot parity tests.
2. **One training row per agent and item.** A re-evaluated decline should not
   multiply an agent's experience of one fact-check. This needs the user's
   decision.
3. **Efficiency.**
   - Train the pooled MLP on the GPU, or warm-start it.
   - Use about 24 workers.
   - Log progress per task.
   - Measure throughput under full load before quoting a time.
4. **The group rule as the decision rule.** The per-cell rules are dropped.
5. **A rerun of the controls (D2c)** with these changes, before any study cell
   is run.
