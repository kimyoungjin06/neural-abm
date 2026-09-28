# Pivot matched-marginal discriminator v3

Date: 2026-09-28

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_matched_marginal_discriminator_v3.yaml`](../../experiments/configs/pivot_matched_marginal_discriminator_v3.yaml)

## Why v3 exists

This successor was designed after the v2 result was opened. The
[v2 result](../../experiments/evidence/results/pivot_matched_marginal_discriminator_v2_20260928.summary.md)
passed all three registered comparisons, but a post-hoc check found two
confounds:

1. The residual MLP beat the explicit linear baseline by almost as much on the
   linear worlds as on the interaction worlds. Most of the registered
   nonlinearity margin came from the baseline's plain gradient-descent fit and
   compressed predictions, not from nonlinearity.
2. The MLP's advantage over the histogram tree was larger on the linear worlds
   than on the interaction worlds. It reflected a general fit to smooth
   boundaries against one frozen tree setting, not a better representation of
   the interaction.

v3 removes both confounds and then asks the question the Pivot ABM needs
answered: which policy family should it carry forward? Seeds are fresh, and the
opened v2 seed namespaces are rejected by the contract.

## Design changes

**Matched linear baseline.** `matched_linear` is the v2 residual MLP with its
tanh hidden branch removed. It shares the frozen-logit initialization, Adam
optimizer, 300 epochs, learning rate, and weight-decay candidates. The MLP and
the linear baseline now differ only by the hidden branch.

**Tuned baselines.** The matched linear model, the tree, and the MLP each pick
a setting by 4-fold inner cross-validation on the training cohort only. The
candidates are registered in advance: three weight decays for the two neural
optimizers, and twelve depth, leaf-size, and boosting-schedule settings for the
tree. The v2 settings are among them. Selection minimizes mean validation
Brier, ties go to the earlier candidate, and the chosen setting is refit on the
full training cohort. Folds stratify by label and assign agents round-robin by
identifier, so both orientations of a pair use the same folds.

**Axis-aligned control.** A third world pair uses the same four features as
the smooth interaction, with each product replaced by a corner function:

```text
corner(x, y) = sign(x) * sign(y) * min(|x|, |y|)
score = max(corner(resource - 0.5, network - 0.5),
            -corner(skill - 0.5, openness - 0.5))
```

The top-half region of this score is a union of axis-aligned boxes. That
matches the inductive bias of a tree, as the smooth world matches the MLP's. A
family that wins only one geometry has shown a fit to that geometry, not a
general advantage.

**Difference in differences.** Every interaction-world gap is also computed
on the linear world of the same orientation and replicate, and the linear-world
gap is subtracted. A gap that is equally present where no interaction exists
cannot pass.

## Model ladder

| Model | Role | Tuned |
|---|---|---|
| `fixed_rule` | Frozen nine-feature logistic predecessor | No |
| `scalar_delta` | One global logit offset from the success count | No |
| `matched_linear` | Residual MLP without the hidden branch | 3 weight decays |
| `histogram_tree` | Classical nonlinear baseline | 12 settings |
| `residual_mlp` | Frozen logit plus linear and 16-unit tanh residuals | 3 weight decays |

## Registered decisions

Each comparison uses the v2 improvement rule. The mean paired Brier
improvement must meet the threshold and be strictly positive in at least 12 of
16 replicates.

| Decision | Comparison | Statistic | Worlds | Threshold |
|---|---|---|---|---:|
| Feature-conditional need | matched linear over scalar | raw gap | both linear | 0.02 |
| Nonlinear representation need | MLP over matched linear | DiD | four interaction | 0.01 |
| Family dominance | MLP versus tree, each direction | raw gap | four interaction | 0.005 |
| Neural interaction specificity | MLP over tree | DiD | both smooth | 0.005 |

Nonlinear need is reported by geometry. A geometry passes when both of its
orientations pass. The disposition is `SUPPORTED_BOTH_GEOMETRIES`,
`SUPPORTED_SMOOTH_ONLY`, `SUPPORTED_AXIS_ONLY`, or `NOT_SUPPORTED`.

A family dominates only if it wins all four interaction worlds. Otherwise the
disposition is `GENERATOR_DEPENDENT_OR_TIED`. A neural-specific advantage
requires MLP dominance and interaction specificity.

## ABM routing

The result routes the next Pivot ABM step by the first matching rule:

| Condition | Route |
|---|---|
| Feature-conditional need not supported | Scalar adaptation policy |
| Nonlinear need not supported in either geometry | Linear conditional policy |
| Neural-specific advantage supported | Neural nonlinear policy, with the tree as comparator |
| Tree dominates all four worlds | Classical nonlinear policy; neural retired from the policy layer |
| Otherwise | Paired classical and neural nonlinear arms |

Nonlinear need supported in one geometry counts as supported for routing. The
route names a policy family to carry into the ABM. The ABM integration needs
its own precommit.

## Boundary and next step

The executable must be committed before the development profile runs. The run
requires that full commit SHA, clean precommit paths, and exact source hashes.
A deterministic rerun into a separate directory must reproduce all four
artifacts byte for byte before the result is recorded.

No post-result change to candidates, worlds, thresholds, folds, or cohorts is
authorized. The result distinguishes policy families on synthetic generators
only. It cannot select an empirical model, establish neural necessity for Pivot
behavior, or change the generic NABM API.
