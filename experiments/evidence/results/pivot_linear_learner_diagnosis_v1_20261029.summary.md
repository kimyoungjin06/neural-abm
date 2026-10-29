# Pivot linear-learner diagnosis v1 (O35): result

Date: 2026-10-29

Status: `O35_LEARNER_PARTLY_OPTIMIZATION`

Design: [pivot-linear-learner-diagnosis-v1.md](../../../docs/research/pivot-linear-learner-diagnosis-v1.md),
frozen at `8c7ffd8b4e26625988e7cee4eae6f69614ae1164`.
Machine-readable record:
[pivot_linear_learner_diagnosis_v1_20261029.summary.json](pivot_linear_learner_diagnosis_v1_20261029.summary.json).

## Execution

- **Host:** the workstation, on CPU with 32 workers.
- **Runs:** 96 tasks, taking 113 s for the development run and 108 s for the
  reproduction run.
- **Reproduction:** all four artifacts are byte identical.
- **Checks:** every registered check passed.
  - The recorded fold learners reproduced each run's probe Brier.
  - Each source arm's refit reproduced every recorded probe output exactly.
  - The slots were restored after every task.
  - Every converged fit reached the gradient tolerance.

## Linear world (primary)

All values are mean probe Brier over 16 replicates, on the linear arm's own
evidence unless stated.

| Fit | Probe Brier | Probe AUC |
|---|---:|---:|
| registered linear | 0.0573 | 0.9926 |
| linear, 10x epochs | 0.0249 | 0.9982 |
| linear, 10x learning rate | 0.0260 | 0.9981 |
| linear, converged at the registered penalty | 0.0247 | 0.9982 |
| linear, converged at a 100x weaker penalty (descriptive) | 0.0108 | 0.9992 |
| registered MLP | 0.0117 | 0.9992 |

**The source is the learner, not the evidence: `LEARNER`.**
- **Learner gap** (registered linear minus registered MLP on the same
  evidence):
  - on the linear arm's evidence: +0.046 [+0.042, +0.049];
  - on the MLP arm's evidence: +0.053 [+0.047, +0.060].
- **Evidence gap** (the same fit on the linear arm's evidence minus the MLP
  arm's). Neither gap is material:
  - registered linear: −0.009 [−0.014, −0.004];
  - registered MLP: −0.0012 [−0.0024, −0.0001].

**Most of the gap is the optimization budget: `PARTLY_OPTIMIZATION`.**
- Converging the registered linear objective lowers the Brier by +0.033
  [+0.030, +0.035]. That closes 71% of the gap.
- 10x the epochs or 10x the learning rate gets within 0.002 of the converged
  fit.
- The converged fit is still 0.013 [0.011, 0.015] worse than the MLP. That is
  outside the registered 0.005 margin, so the outcome is not
  `OPTIMIZATION_BUDGET`.
- **The rest is the penalty, descriptively.** At the registered weight decay,
  the converged residual norm is 30. At a 100x weaker penalty it is 149, and
  that fit is 0.0009 [0.0005, 0.0013] better than the MLP. The MLP reaches a
  steep slope through products of smaller parameters, which the same L2
  penalty charges less.
- **The ranking was never the problem.** The MLP's AUC exceeds the registered
  linear learner's by only 0.0066 [0.0048, 0.0084].

## Interaction worlds (reported, the linear policy is misspecified)

| World | Learner gap, on own / MLP evidence | Converged minus registered | MLP minus linear AUC |
|---|---|---|---|
| smooth | +0.066 / +0.092 | −0.001 | +0.180 |
| axis | +0.051 / +0.077 | +0.001 | +0.193 |

- Converging changes nothing in either world (`NOT_OPTIMIZATION`). The MLP
  ranks far better there, so its edge is the policy class.
- Both worlds are `BOTH`: the linear learner does better on its own arm's
  evidence than on the MLP arm's.

## Expectations

- **E1 held.** The source is `LEARNER`.
- **E2 failed.** The outcome is `PARTLY_OPTIMIZATION`, not
  `OPTIMIZATION_BUDGET`. Converging closes 71% of the gap. The rest comes from
  the registered weight decay, which limits the linear learner's slope more
  than the MLP's.
- **E3 held.** The AUC gap is 0.0066.

## What this means

- In the linear world, the linear learner loses to the MLP because of how it
  is trained, not because of what it can represent or what it saw. About 71%
  of the gap is the 300-step Adam budget, and most of the rest is the shared
  L2 penalty acting on a linear parameterization.
- In the interaction worlds, the MLP's edge is its policy class. The capacity
  conclusions about those worlds stand. Comparisons of the linear learner with
  others under the registered settings carry a training handicap that is
  largest where its class is right.

## Boundary

The worlds are synthetic, the hand rule is the only prior, and the
fits are the registered ones and five variants of the linear fit. The weak
penalty and the 10x variants are descriptive. The study selects no setting
and does not establish neural necessity.
