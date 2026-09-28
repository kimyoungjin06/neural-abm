# Pivot nonlinear policy ABM v1 result

Date: 2026-09-28

Status: `SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED`

## Result

The registered dispositions are:

| Decision | Registered disposition |
|---|---|
| Overall | `NONLINEAR_LEARNING_CHANGES_BUT_BEHAVIOR_DOES_NOT` |
| ABM geometry crossover | `TREE_DOMINATES_IN_ABM` |
| Evidence scale | `NONLINEAR_LEARNING_SURVIVES_LOCAL_EVIDENCE` |

Post-hoc checks on the frozen rows show that only the first holds as a
statement about the Pivot mechanism. Even the first hides the more interesting
pattern. Three registered design choices were wrong for this ABM, and they are
listed below. Corrected, the result reads:

- **Nonlinear policies change who enters when failure is visible
  immediately.** In both interaction geometries, field-wide tree and MLP
  policies raise the productive share of entrants beyond what they add in the
  linear world. All four family-geometry pairs pass the registered threshold
  under immediate visibility. Entry decisions are made before an agent's own
  outcome is public, so this effect is out of sample.
- **A four-step failure delay swamps policy precision.** In every field-arm
  run under delayed failure, all 256 agents enter. No policy can change who
  enters, so the registered rule, which required both visibility regimes,
  failed.
- **Neighbor evidence is too thin for feature learning.** With about four
  events per agent, the linear, tree, and MLP learners are worse than the fixed
  prior in every interaction-world replicate.
- **The ABM does not test which nonlinear family generalizes better.** The
  registered policy Brier is mostly in sample, and trees memorize best. The
  out-of-sample family transfer check reproduces the v3 crossover.

The run used `4ffc5dd`: 960 runs over 3 fit worlds, 2 visibility regimes, 10
arms, and 16 replicates of 256 agents. The fixed and online-logistic arms
exactly nested the persistence predecessor in the linear world. A second
execution reproduced all five artifacts byte for byte.

## Family transfer check

Out of sample, on fresh v3-style cohorts, the ABM learners reproduce v3. Every
condition held in all four replicates.

| World | Scalar | Linear | Tree | MLP |
|---|---:|---:|---:|---:|
| Linear | 0.250 | 0.063 | 0.053 | 0.031 |
| Smooth interaction | 0.250 | 0.225 | 0.109 | 0.072 |
| Axis interaction | 0.250 | 0.227 | 0.071 | 0.125 |

The prior-anchored boosted tree is therefore a competent stand-in for the v3
tree.

## Registered decisions

Field regime unless stated. DiD subtracts the same gap in the linear world.
A pair passes when both visibility regimes pass.

| Decision | Pair | Immediate | Failure lag 4 | Pass |
|---|---|---:|---:|---|
| Policy learning | tree, smooth | 0.095 (16) | 0.113 (16) | yes |
| Policy learning | tree, axis | 0.120 (16) | 0.125 (16) | yes |
| Policy learning | MLP, smooth | 0.086 (16) | 0.105 (16) | yes |
| Policy learning | MLP, axis | 0.086 (16) | 0.098 (16) | yes |
| Entry behavior | tree, smooth | 0.037 (16) | 0.000 (0) | no |
| Entry behavior | tree, axis | 0.062 (16) | 0.000 (0) | no |
| Entry behavior | MLP, smooth | 0.023 (13) | 0.000 (0) | no |
| Entry behavior | MLP, axis | 0.022 (14) | 0.000 (0) | no |
| Crossover | tree over MLP, smooth | 0.010 (12) | 0.012 (16) | yes |
| Crossover | tree over MLP, axis | 0.035 (16) | 0.031 (16) | yes |
| Local learning | tree, smooth | 0.041 (16) | 0.042 (16) | yes |
| Local learning | tree, axis | 0.046 (16) | 0.045 (16) | yes |
| Local learning | MLP, smooth | -0.011 (3) | -0.013 (1) | no |
| Local learning | MLP, axis | -0.014 (4) | -0.018 (1) | no |

Cells show the mean, with positive replicates out of 16 in parentheses.
Thresholds: 0.01 for policy learning, 0.02 for entry behavior, 0.005 for the
crossover.

## Field-regime productive share of entrants, immediate visibility

| World | Scalar | Linear | Tree | MLP |
|---|---:|---:|---:|---:|
| Linear fit | 0.324 | 0.398 | 0.396 | 0.420 |
| Smooth interaction fit | 0.334 | 0.355 | 0.390 | 0.400 |
| Axis interaction fit | 0.333 | 0.348 | 0.409 | 0.393 |

Once interaction structure matters, the linear policy loses about 0.04 to 0.05
of entrant precision. The nonlinear policies recover most of it, the tree more
than the MLP.

## Post-hoc checks

These checks were not registered. They reuse the frozen rows and run nothing new.

**The field policy Brier is mostly in sample.** The shared learner's buffer
contains 71% to 74% of the evaluated agents under immediate visibility, and
100% under delayed failure. There, every field arm reached full entry in all 48
runs. The registered Brier then scores memorization of the public record. The
boosted trees reach a Brier of 0.0001 to 0.003, with AUC 1.000, even in the
smooth world, where they lose out of sample. The registered tree dominance is a
memorization result, not a generalization result.

**Every local feature learner is worse than the prior in interaction worlds.**
Policy Brier minus the fixed prior's, neighbor evidence:

| Visibility | World | Scalar | Linear | Tree | MLP |
|---|---|---:|---:|---:|---:|
| Immediate | Linear fit | -0.022 | -0.029 | -0.007 | +0.004 |
| Immediate | Smooth interaction | -0.000 | +0.070 | +0.051 | +0.114 |
| Immediate | Axis interaction | +0.001 | +0.078 | +0.055 | +0.126 |
| Failure lag 4 | Linear fit | -0.024 | -0.055 | -0.021 | -0.024 |
| Failure lag 4 | Smooth interaction | -0.001 | +0.045 | +0.036 | +0.089 |
| Failure lag 4 | Axis interaction | -0.001 | +0.048 | +0.036 | +0.097 |

The linear, tree, and MLP learners are worse than the prior in all 16
replicates of every interaction-world cell. Four events cannot identify an
interaction, and they mislead a feature learner when the prior is wrong. The
tree passed the registered local rule only because its reference, the linear
learner, degrades more. Neighbor-scale evidence prevents useful feature
learning, which is the opposite of the registered disposition.

**High-capacity field learners purge failed stock by memorizing identities.**
Field tree and MLP arms leave almost no active failed agents in any world,
including the linear one. At most 0.013 remain, against 0.030 to 0.049 for the
linear arm. The shared record includes each agent's own public failure, and the
exit score reads the policy output. This is identity memorization. It is not
interaction learning.

## Design defects

1. The field policy Brier includes the evaluated agent's own public outcome.
   It should score predictions made before that outcome is known.
2. The local DiD uses the linear learner as reference. When data are too thin,
   that reference degrades differently across worlds. The fixed prior is the
   stable reference.
3. The entry-behavior rule required both visibility regimes. Delayed failure
   saturates entry, so the rule could not pass for any policy. Visibility
   regimes should be reported separately, with saturation reported as its own
   outcome.
4. Letting the field learner see an agent's own outcome allows identity
   memorization through the exit score.

## NABM conclusion

For the Pivot flagship case:

- nonlinear policy families matter for entry only when the field shares
  outcome evidence widely and failure is visible early;
- delayed failure visibility produces saturated entry that no policy precision
  can correct, consistent with the earlier delayed-feedback overshoot;
- neighbor-scale evidence cannot support feature learning of any kind in
  interaction worlds; and
- the choice between tree and neural policies remains a question of boundary
  geometry, which this ABM did not test out of sample.

No model is selected and neural necessity is not established. A result-informed
successor should score policies at decision time, use the fixed prior as the
local reference, report visibility regimes separately, and exclude the agent's
own outcome from the field record it acts on. v1 is not retuned.

## Boundary

This scorecard uses deterministic synthetic ABM runs only. It reads zero
empirical rows, performs no empirical fit, changes no generic NABM API, selects
no model, establishes no neural necessity or scientific effect, and authorizes
no package promotion.
