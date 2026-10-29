# Pivot linear-learner diagnosis v1 (O35)

Date: 2026-10-29

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_linear_learner_diagnosis_v1.yaml`](../../experiments/configs/pivot_linear_learner_diagnosis_v1.yaml)

## Why this study exists

In the linear world, the productive label is a deterministic linear
threshold of the features: `productive_fit ≥ 0.40`, where `productive_fit` is a
linear score. The pooled linear learner is a logistic residual on the prior
logit, so it is correctly specified there. The pooled MLP is the same residual
plus a tanh hidden layer, so it nests the linear learner exactly. Yet under
the hand rule in the entry-matched control, the linear learner's probe Brier
was 0.060, against the MLP's 0.013.

[Consolidation v10](pivot-policy-ladder-consolidation-v10.md) opened this as
O35. It reopens O12 from v2, which was not carried after v2. Every capacity
comparison in the ladder depends on the answer, and the user confirmed the
study on 2026-10-29.

## Two candidate explanations

- **The learner.** The best forecast of a deterministic threshold is a step. A
  logistic policy approaches a step only with large weights. Adam at a
  learning rate of 0.02 moves each residual parameter by at most about 6 in
  the registered 300 steps. The MLP multiplies two such moves across 16 units,
  so it can sharpen faster. If this is the cause, the linear learner ranks
  almost as well as the MLP but is under-confident. Converging its registered
  objective would then close the gap.
- **The evidence.** In the closed loop, each learner's own policy decides who
  enters, and so what evidence it sees. The linear arm may see different, less
  informative entrants.

## Design

The hand rule fills the prior slot. The design follows the v3 lifecycle:
- the field-public cross-fit with four folds;
- immediate visibility;
- three fit worlds;
- 16 replicates of 256 agents; and
- fresh seeds for the study (20261211) and the probe cohort (20261212).

For each replicate and world, the linear arm and the MLP arm are each run
once. During the run, a subclass of the v2 probed learner fills the learner
slot. It only lists itself, and the slot is restored afterwards. At the end of
the run, each fold learner's final training buffer and the seed of its last
refit are recorded.

Each buffer is then refitted offline and scored on the replicate's probe
cohort:

| Refit | What it is |
|---|---|
| `registered_linear` | The registered linear learner, as in the run |
| `registered_mlp` | The registered MLP |
| `linear_epochs_x10` | The linear learner with 3,000 Adam steps |
| `linear_learning_rate_x10` | The linear learner at a learning rate of 0.2 |
| `linear_converged` | The registered linear objective minimized to convergence by L-BFGS: mean cross-entropy plus half the weight decay (0.0001) times the squared residual norm, which is what Adam's weight decay optimizes |
| `linear_converged_weak_penalty` | The same, at a weight decay of 0.000001 (descriptive) |

Fitting the MLP on the linear arm's buffers, and the linear learner on the
MLP arm's buffers, crosses learner with evidence.

## Checks

The run stops as an invalid implementation if any of these fail:
- the recorded fold learners reproduce the run's probe Brier exactly;
- the source arm's registered refit on each final buffer reproduces the
  recorded probe output exactly;
- the learner slot is restored and the prior slot is the hand rule after
  every task; and
- every `linear_converged` fit reaches a maximum absolute gradient of 1e-6.

## Registered decisions

The unit is a replicate, scored as the mean over its four fold learners, as in
the run's probe Brier. Each difference is judged by its paired 95% t interval
over 16 replicates. A difference is material if the interval excludes zero
and its absolute mean is at least 0.01. The primary world is the linear world;
the other two are reported the same way.

**Source of the gap.** The learner gap is the registered linear learner's
Brier minus the registered MLP's, on the same evidence, for each source arm.
The evidence gap is a fit's Brier on the linear arm's evidence minus its
Brier on the MLP arm's evidence, for each registered fit.

| Outcome | Condition |
|---|---|
| `LEARNER` | Both learner gaps material and positive, no evidence gap material |
| `EVIDENCE` | An evidence gap material, and not both learner gaps |
| `BOTH` | Both learner gaps material and positive, and an evidence gap material |
| `NEITHER` | Otherwise |

**Optimization budget**, on the linear arm's own evidence:

| Outcome | Condition |
|---|---|
| `NO_GAP_TO_EXPLAIN` | The learner gap on this evidence is not material and positive |
| `OPTIMIZATION_BUDGET` | The upper 95% bound of `linear_converged` minus `registered_mlp` is at most 0.005 |
| `PARTLY_OPTIMIZATION` | Not the above, but `registered_linear` minus `linear_converged` is material and positive |
| `NOT_OPTIMIZATION` | Otherwise |

The study status is `O35_<source>_<optimization>` in the linear world. Log
loss, AUC, sharpness, the converged residual norm, and the evidence count and
productive share are reported for every cell.

## Expectations before the run

These follow from the argument above. Each will be recorded as held or
failed, and a failed expectation is recorded as a rejected claim.

- **E1.** In the linear world the source is `LEARNER`.
- **E2.** In the linear world the optimization outcome is
  `OPTIMIZATION_BUDGET`.
- **E3.** On the linear arm's evidence, the registered MLP's probe AUC exceeds
  the registered linear learner's by less than 0.02. The gap is then
  sharpness, not ranking.

## Boundary

The executable must be committed before the development profile runs. A
deterministic rerun on the same host must reproduce every artifact byte for
byte. No post-result change to the refits, the thresholds, or the decisions is
authorized.

The worlds are synthetic, and the hand rule is the only prior. The study makes
no LLM or paid API calls, reads no empirical rows, changes no generic NABM API,
and selects no model. Whichever outcome it records describes these learners
under these settings. It does not establish neural necessity.
