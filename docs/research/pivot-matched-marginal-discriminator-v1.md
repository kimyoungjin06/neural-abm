# Pivot matched-marginal discriminator v1

Date: 2026-08-07

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_matched_marginal_discriminator_v1.yaml`](../../experiments/configs/pivot_matched_marginal_discriminator_v1.yaml)

## Purpose

The adaptive baseline ladder established that a one-state delta rule reproduces
the delayed overshoot, purge, and selective-missingness directions of the
current neural logistic policy. That result distinguishes fixed from adaptive
behavior, but it does not make feature-conditional inference necessary.

This successor holds the aggregate success fraction exactly fixed while
reversing which feature profiles succeed. It asks three nested questions:

1. Does the case require more than scalar adaptation?
2. Does it require nonlinear rather than linear conditional inference?
3. Does a neural nonlinear policy outperform a strong classical nonlinear
   policy on disjoint predictive targets?

The study is a repository-only synthetic policy-layer discriminator. It is not
an empirical fit, a full persistence-and-exit ABM run, or a generic NABM model
selection exercise.

## Exact matched-marginal construction

Each replicate contains a public synthetic training cohort and a disjoint
synthetic evaluation cohort. The cohorts use different frozen seed namespaces.
Within a cohort and replicate, all worlds share byte-identical feature rows and
event counts. Exactly half the rows are productive in every world.

Labels are assigned to the top exact half of a frozen score, with ascending
agent identifier as the tie-break. The reversed world uses the exact complement
of that assignment, which is the negative-score orientation without tie drift. The
linear pair reverses a support-and-skill profile. The interaction pair reverses
a nonlinearly separable support-by-network versus skill-by-openness surface.

Because features and marginal labels are fixed within a pair, a one-state
scalar learner receives the same evidence target in both orientations. Its
predictions must therefore be exactly invariant. Feature-aware learners may
change only because the label-feature association changes.

## Model ladder

| Model | Role |
|---|---|
| `fixed_rule` | Frozen nine-feature logistic predecessor |
| `scalar_delta` | One global adaptive logit offset |
| `explicit_linear_logit` | Analytic, non-backpropagated linear residual |
| `histogram_tree` | Classical nonlinear nonneural baseline |
| `residual_mlp` | Frozen-logit prior plus linear and tanh residuals |

The linear baseline prevents a Delta-versus-Neural difference from being
misread as neural necessity when it only demonstrates feature use. The tree
baseline prevents a nonlinear MLP advantage over a linear model from being
misread as neural necessity when a conventional nonlinear learner performs as
well.

All differentiable policies start from the same frozen logistic prediction.
The scalar, linear, tree, and MLP settings are frozen before any development
scorecard is produced. No post-result learning-rate, architecture, world,
threshold, or cohort sweep is authorized in v1.

## Disjoint targets and decisions

Primary evaluation uses held-out Brier score, log loss, ROC AUC, and conditional
separation. Paired comparisons retain replicate-level improvements and signs.

Feature-conditional need requires the explicit linear model to improve Brier
score over scalar Delta by at least `0.02` in at least 12 of 16 replicates in
each linear orientation. Nonlinear need requires the residual MLP to improve
over the explicit linear model by at least `0.01` with the same replicate rule
in both interaction orientations.

A neural-specific advantage is narrower: the MLP must also improve over the
histogram tree by at least `0.005` in both interaction orientations and may not
regress by more than `0.01` on the linear worlds. Failure of this comparison
supports nonlinearity at most, not neural necessity.

The thresholds govern only this frozen synthetic generator. They do not select
an empirical policy, establish a scientific effect, or promote a case-owned
model into the generic NABM API.

## Boundary and next step

The executable must be committed before the development profile can run. The
run requires that full commit SHA, clean precommit paths, and exact predecessor
source hashes. Outputs are exclusive-create artifacts.

After a frozen result, a full ABM integration is authorized only if the policy
layer distinction is both valid and substantively useful. Otherwise the
simpler surviving policy family remains the correct baseline for Pivot.
