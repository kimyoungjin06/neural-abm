# Pivot matched-marginal discriminator v1 stop

Date: 2026-09-28

Status: `INVALID_IMPLEMENTATION_STOP_NO_RESULT`

## Outcome

The frozen development run at `b960fe0` passed precommit verification and
stopped in replicate 0 at the registered scalar invariance check for
`interaction_orientation`. The contract names this failure
`INVALID_IMPLEMENTATION_STOP`. No output directory was created. No artifact was
written, and no metric was printed or inspected. Replicate 0 metrics existed
only in process memory and were discarded when the process exited.

None of the three registered questions was evaluated. This record makes no
claim about feature-conditional, nonlinear, or neural-specific need.

## Defect

The v1 scalar update reduces `mean(train_labels - prediction)`. Aligned and
reversed worlds have the same success count but different label vectors, so the
elementwise differences round differently and the mean can differ in its final
bit. The learner was therefore not implemented as a function of the success
count alone, which the contract requires.

The diagnosis compared scalar predictions only with each other. It computed no
label metric.

| Profile | Replicates | Linear pair non-identical | Interaction pair non-identical |
|---|---:|---|---|
| Smoke, 64 agents | 2 | none | none |
| Development, 256 agents | 16 | 3, 4, 5, 6, 7, 8, 10, 15 | 0, 1, 10, 12 |

The maximum prediction difference is `1.11e-16`. The frozen invariance test
passed because it used only the 64-agent smoke cohort for replicate 0, where
both reductions happen to round identically.

## Correction checked

Computing the update as `success_fraction - mean(prediction)`, with the
fraction taken from the training success count, produced zero invariance
failures in every smoke and development replicate. Predictions differ from v1
by at most `1.11e-16`, so the correction changes the arithmetic, not the
learner.

## Successor

A v2 contract corrects only the scalar reduction and adds an invariance test
over every smoke and development replicate. Seeds, cohorts, worlds, the model
ladder, hyperparameters, thresholds, and the exact invariance requirement are
unchanged. v2 must be committed before it runs.

## Boundary

This is an execution stop record. It reads zero empirical rows, performs no fit
or predictive validation, selects no model, and establishes no neural necessity
or scientific effect.
