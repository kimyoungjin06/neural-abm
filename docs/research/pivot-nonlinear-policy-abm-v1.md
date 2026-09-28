# Pivot nonlinear policy ABM v1

Date: 2026-09-28

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_nonlinear_policy_abm_v1.yaml`](../../experiments/configs/pivot_nonlinear_policy_abm_v1.yaml)

## Why this study exists

The [matched-marginal v3 result](../../experiments/evidence/results/pivot_matched_marginal_discriminator_v3_20260928.summary.md)
showed that nonlinear policies are needed on synthetic interaction worlds, and
that tree and MLP policies split by boundary geometry. Its frozen routing rule
sends both families into the Pivot ABM as paired arms.

Two facts found while designing that step shape this study:

1. **The current ABM gives nonlinear policies nothing to learn.** Pivot's
   structural productive fit is a clipped linear combination of the features,
   so a linear policy is already correctly specified.
2. **Agents see very little evidence.** An imitative agent in the adaptive
   ladder saw about three neighbor outcomes over a whole run. The v3 learners
   trained on 256 rows.

This study therefore adds interaction-fit worlds and a field-wide evidence
regime, keeping the current world and the current evidence scale as nested
controls.

## Fit worlds

| World | Structural fit |
|---|---|
| `linear_fit` | Exact study-1 productive fit. Nested control. |
| `smooth_interaction_fit` | Rank-matched to the v3 smooth interaction score |
| `axis_interaction_fit` | Rank-matched to the v3 axis-aligned interaction score |

Rank matching sorts a replicate's agents by the world score and hands out that
population's linear fit values in the same order. Each replicate keeps its exact
multiset of fit values, so the productive base rate and the fit component of the
exit score are unchanged. Only which agents are structurally productive changes.
Entry, readiness, networks, and noise do not read the fit, so they are unchanged
too.

## Evidence regimes

| Regime | Buffer | Learner |
|---|---|---|
| `neighbor_local` | Each agent's cumulative public neighbor outcomes, excluding its own | One per agent |
| `field_public` | All public outcomes in the field, including the agent's own | One shared by all agents |

The local regime keeps the predecessor's evidence scale. The field regime
models a shared public record. With 256 agents it reaches the v3 evidence
scale.

## Policy arms

| Arm | Evidence | Learner |
|---|---|---|
| `fixed_rule` | none | Frozen prior |
| `online_logistic` | local | Exact predecessor imitative logistic |
| `scalar_batch` | local, field | v2 count-based scalar offset |
| `linear_batch` | local, field | v3 matched linear |
| `tree_batch` | local, field | Newton-boosted trees from the prior logit |
| `mlp_batch` | local, field | v3 residual MLP |

Batch learners refit from the frozen prior on the full buffer whenever it
grows. With an empty buffer they output the prior exactly. The linear and MLP
learners use the weight decay most often selected in v3. The v3 histogram tree
cannot start from a prior, so it cannot define a policy before evidence
arrives. `tree_batch` instead boosts 300 depth-3 regression trees from the
frozen prior logit, using the 300-round schedule v3 favored. Its leaf values
use an L2 penalty. With a handful of events it cannot split and reduces to a
regularized offset.

Because `tree_batch` is not the v3 tree, a **family transfer check** runs
first. It uses fresh v3-style cohorts, with 256 rows and four replicates. It
asks whether the four batch learners reproduce the v3 pattern: both nonlinear
families beat linear, the MLP wins the smooth world, and the tree wins the axis
world. If they do not, the ABM family comparisons are reported with restricted
interpretation.

## Nesting

In `linear_fit`, the fixed and online-logistic arms must equal the persistence
predecessor exactly, in every final value and trajectory. Any mismatch stops the
run as an invalid implementation.

## Registered decisions

All decisions use the field regime unless stated. They use the paired rule from
v3: the mean must meet the threshold and be strictly positive in at least 12 of
16 replicates. DiD subtracts the same gap in `linear_fit` for the same
visibility, evidence regime, and replicate. A family-geometry pair passes when
both visibility regimes pass.

| Decision | Metric | Comparison | Threshold |
|---|---|---|---:|
| Policy learning | in-situ policy Brier | tree or MLP over linear, DiD | 0.01 |
| Behavioral consequence | productive share of entrants | tree or MLP over linear, DiD | 0.02 |
| ABM geometry crossover | in-situ policy Brier | MLP over tree on smooth; tree over MLP on axis | 0.005 |
| Evidence scale | in-situ policy Brier | the policy-learning rule under local evidence | 0.01 |

In-situ policy Brier compares every agent's final policy output with its
structural productive label. It measures what the agents learned. The productive
share of entrants measures whether that learning changed who entered.

The overall result is one of:

- `NONLINEAR_LEARNING_AND_BEHAVIOR_BOTH_CHANGE`
- `NONLINEAR_LEARNING_CHANGES_BUT_BEHAVIOR_DOES_NOT`
- `NO_NONLINEAR_LEARNING_IN_ABM`

The crossover is reported as `CROSSOVER_TRANSFERS_TO_ABM`,
`MLP_DOMINATES_IN_ABM`, `TREE_DOMINATES_IN_ABM`, or `TIED_OR_MIXED`. Evidence
scale is reported as `EVIDENCE_SCALE_GATES_NONLINEAR_LEARNING`,
`NONLINEAR_LEARNING_SURVIVES_LOCAL_EVIDENCE`, or
`NO_NONLINEAR_LEARNING_UNDER_FIELD_EVIDENCE`.

## Design

The development profile uses 256 agents, 18 steps, 2 burn-in steps, and 16
paired replicates. It has two visibility regimes, immediate and four-step
failure delay, and three fit worlds. Each combination runs 10 arms, for 960
runs. Population, network, and local noise are identical across arms and worlds
within a replicate. Seeds are fresh.

Tasks run in parallel worker processes with one thread each. A test confirms
that parallel and serial execution give identical rows.

## Boundary

The executable must be committed before the development profile runs, and a
deterministic rerun must reproduce every artifact byte for byte. No
post-result change to learners, worlds, thresholds, or cohorts is authorized.

This is a synthetic generative comparison. It reads no empirical rows, fits
nothing empirical, selects no model, and changes no generic NABM API. The fit
worlds are stylized alternatives, not claims about how real pivots succeed.
Which geometry real Pivot success follows is an empirical question this study
cannot answer.
