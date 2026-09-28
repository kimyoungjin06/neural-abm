# Pivot matched-marginal discriminator v2

Date: 2026-09-28

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_matched_marginal_discriminator_v2.yaml`](../../experiments/configs/pivot_matched_marginal_discriminator_v2.yaml)

## Why v2 exists

The frozen v1 development run stopped at its registered scalar invariance
check with `INVALID_IMPLEMENTATION_STOP`. No artifact was written and none of
the three questions was evaluated. The stop record is
[`pivot_matched_marginal_discriminator_v1_20260928.summary.md`](../../experiments/evidence/results/pivot_matched_marginal_discriminator_v1_20260928.summary.md).

The defect was arithmetic. v1 averaged `train_labels - prediction`, so aligned
and reversed worlds with equal success counts could round to different offsets
in the final bit. v2 reduces the labels to their success count before the
update:

```text
offset += eta * (success_count / n - mean(sigmoid(base_logit + offset)))
          - prior_anchor * offset
```

The scalar learner now depends on the labels only through their count, so its
predictions are byte-identical across orientations by construction. On every
smoke and development replicate, v2 predictions differ from v1 by at most
`1.11e-16`.

## What is unchanged

The [v1 design](pivot-matched-marginal-discriminator-v1.md) carries forward
unchanged: purpose, cohorts and seeds, matched-marginal worlds, label
assignment, the five-model ladder and its hyperparameters, targets, thresholds,
replicate rules, profiles, and boundaries. A contract test pins these sections
to the v1 contract.

v1 produced no evaluated result, so reusing its seeds exposes no outcome.

## Decision rule, stated exactly

The v1 prose could be read as a per-replicate threshold. The frozen v1 code
applies a different rule, and v2 keeps that rule and names it in the contract
as `improvement_rule`. For each comparison and world:

- the mean paired Brier improvement over 16 replicates must meet the threshold;
- the improvement must be strictly positive in at least 12 of 16 replicates;
  and
- both worlds in the pair must pass.

The thresholds are unchanged: `0.02` for explicit linear over scalar Delta on
the linear worlds, `0.01` for the residual MLP over explicit linear on the
interaction worlds, and `0.005` for the residual MLP over the histogram tree on
the interaction worlds. The neural comparison also requires that the MLP's
worst per-replicate Brier regression against explicit linear on the linear
worlds be at most `0.01`.

## Added check

v1 tested invariance only on the 64-agent smoke cohort for replicate 0. v2
tests exact invariance on every smoke and development replicate before any run.

## Boundary and next step

The executable must be committed before the development profile runs. The run
requires that full commit SHA, clean precommit paths, and exact source hashes,
including the v1 contract and stop record. Outputs are exclusive-create
artifacts. A deterministic rerun into a separate directory must reproduce all
four artifacts byte for byte before the result is recorded.

No post-result change to learning rates, architecture, worlds, thresholds, or
cohorts is authorized. A result can distinguish policy capacity only on this
frozen synthetic generator. It cannot select an empirical model or change the
generic NABM API.
