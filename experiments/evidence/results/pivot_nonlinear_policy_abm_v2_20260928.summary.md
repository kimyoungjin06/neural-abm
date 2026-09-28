# Pivot nonlinear policy ABM v2 result

Date: 2026-09-28

Status: `SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED`

## Result

With v1's four design defects fixed, the registered outcomes are:

| Visibility | Learning | Crossover | Evidence scale | Entry | Exit |
|---|---|---|---|---|---|
| Immediate | out of sample | transfers | gates useful learning | active: tree | active: tree |
| Failure lag 2 | out of sample | transfers | gates useful learning | saturated | active: tree, axis |
| Failure lag 4 | out of sample | transfers | gates useful learning | saturated | active: tree, axis |

In words:

- **Field-wide nonlinear policies learn interaction structure out of sample.**
  Scored on probe agents the ABM never simulated, tree and MLP policies beat
  the linear policy by more than they do in the linear world. This holds for
  both geometries, all three visibility regimes, and all 16 replicates.
- **The v3 geometry crossover survives inside the ABM.** The MLP wins the
  smooth world in 16 of 16 replicates. The tree wins the axis world in 15 or 16
  of 16. v1's tree dominance was a memorization artifact.
- **Evidence scale decides whether learning helps at all.** With field-wide
  evidence, every feature learner beats the fixed prior in every replicate.
  With neighbor evidence, every one is worse than the prior in every replicate.
- **Learning changes who enters only when failure is visible immediately.**
  From a two-step failure delay onward, the linear, tree, and MLP field arms put
  every agent in, in all 48 runs each.
- **Entry and exit follow the crossover, but only the tree's gains are
  interaction-specific.** In raw terms, the MLP gives the best entrant
  precision in the smooth world and the tree in the axis world. The MLP fails
  the DiD because it also beats the linear policy in the linear world.

The run used `c56d741`: 1,440 runs over 3 visibility regimes, 3 fit worlds, 10
arms, and 16 replicates of 256 agents. The fixed and online-logistic arms
exactly nested the persistence predecessor. No field fold learner saw its own
fold. The family transfer check passed every condition in 4 of 4 replicates. A
second execution reproduced all five artifacts byte for byte.

## Registered decisions

All values are means over 16 replicates, with strictly positive replicates in
parentheses. DiD subtracts the same gap in the linear world.

**Field nonlinear learning.** Probe Brier improvement over linear, DiD.
Threshold 0.01.

| Pair | Immediate | Lag 2 | Lag 4 |
|---|---:|---:|---:|
| Tree, smooth | 0.065 (16) | 0.077 (16) | 0.077 (16) |
| Tree, axis | 0.140 (16) | 0.146 (16) | 0.146 (16) |
| MLP, smooth | 0.046 (16) | 0.067 (16) | 0.066 (16) |
| MLP, axis | 0.027 (16) | 0.041 (16) | 0.041 (16) |

**Field geometry crossover.** Raw probe Brier gap. Threshold 0.005.

| Comparison | Immediate | Lag 2 | Lag 4 |
|---|---:|---:|---:|
| MLP over tree, smooth | 0.040 (16) | 0.045 (16) | 0.045 (16) |
| Tree over MLP, axis | 0.054 (16) | 0.049 (15) | 0.049 (15) |

**Useful learning.** Probe Brier improvement over the fixed prior. Threshold
0.01. Ranges cover both interaction worlds and all three regimes.

| Learner | Field evidence | Neighbor evidence |
|---|---:|---:|
| Linear | 0.014 to 0.029 (16 each) | -0.103 to -0.078 (0 each) |
| Tree | 0.081 to 0.151 (16 each) | -0.074 to -0.054 (0 each) |
| MLP | 0.092 to 0.136 (16 each) | -0.145 to -0.124 (0 each) |

**Entry behavior.** Productive share of entrants over linear, DiD. Threshold
0.02.

| Pair | Immediate | Lag 2 | Lag 4 |
|---|---:|---:|---:|
| Tree, smooth | 0.028 (15) | saturated | saturated |
| Tree, axis | 0.069 (16) | saturated | saturated |
| MLP, smooth | 0.014 (13) | saturated | saturated |
| MLP, axis | 0.013 (13) | saturated | saturated |

**Exit channel.** Reduction in active failed stock over linear, DiD.
Threshold 0.01.

| Pair | Immediate | Lag 2 | Lag 4 |
|---|---:|---:|---:|
| Tree, smooth | 0.017 (13) | 0.001 (6) | 0.001 (6) |
| Tree, axis | 0.053 (16) | 0.043 (16) | 0.043 (16) |
| MLP, smooth | -0.014 (3) | -0.017 (4) | -0.015 (3) |
| MLP, axis | -0.023 (2) | -0.036 (0) | -0.038 (1) |

## Raw field-arm means, immediate visibility

| World | Arm | Entry | Productive share of entrants | Active failed | Probe Brier |
|---|---|---:|---:|---:|---:|
| Linear | linear | 0.749 | 0.408 | 0.044 | 0.059 |
| Linear | tree | 0.754 | 0.402 | 0.051 | 0.073 |
| Linear | MLP | 0.704 | 0.436 | 0.011 | 0.013 |
| Smooth | linear | 0.667 | 0.368 | 0.059 | 0.176 |
| Smooth | tree | 0.711 | 0.390 | 0.049 | 0.124 |
| Smooth | MLP | 0.695 | 0.410 | 0.039 | 0.085 |
| Axis | linear | 0.662 | 0.355 | 0.063 | 0.187 |
| Axis | tree | 0.718 | 0.418 | 0.017 | 0.060 |
| Axis | MLP | 0.718 | 0.396 | 0.053 | 0.114 |

## Post-hoc checks

These checks were not registered. They reuse the frozen rows and run nothing new.

**Behavior follows the crossover in raw terms.** Under immediate visibility,
the MLP's entrant precision exceeds the tree's by 0.020 in the smooth world, in
15 of 16 replicates. The tree's exceeds the MLP's by 0.022 in the axis world,
in 15 of 16. Active failed stock follows the same ordering. Each family changes
behavior most where its inductive bias matches the success boundary.

**Why the MLP fails the entry and exit DiDs.** Even in the linear world, the
MLP beats the linear policy in all 16 replicates. Its margins are 0.046 to
0.047 in probe Brier and 0.023 to 0.033 in failed stock across regimes. Under
immediate visibility it also gains 0.028 in entrant precision. That edge is not interaction-specific, and the DiD removes it,
as registered. The tree is slightly worse than linear in the linear world, so
its DiD gains are not reduced. "Only the tree passes" therefore means that only
the tree's behavioral gain is specific to interaction structure. It does not
mean the MLP leaves behavior unchanged.

**Under delay, the MLP keeps more failed agents active in the axis world.**
With a two- or four-step delay, the axis-world MLP arm holds 0.059 to 0.060
active failed agents, against 0.046 for linear and 0.009 for the tree. Its
axis-world policy is less sharp for failed agents than the tree's, and after
saturated entry the exit score is the only place policy matters.

**Cross-fitting removed v1's memorization.** Field in-situ and probe Brier now
agree within 0.008, with one exception. In the axis world the tree's in-situ
Brier is 0.023 to 0.025 better than its probe Brier. In v1 the in-situ tree Brier was
as low as 0.0001. A possible cause, not tested here, is that rank matching sets
each population's axis thresholds from its own quantiles. Sharp tree splits
then transfer imperfectly to a new cohort.

**Saturation is complete from lag 2.** The linear, tree, and MLP field arms
reached full entry in 48 of 48 runs at both delays. The scalar arm did so in 16
of 48 at lag 2 and 36 of 48 at lag 4.

## NABM conclusion

For the Pivot flagship case, the policy-layer ladder now ends here:

- adaptation is needed, and scalar adaptation is not enough when feature
  structure matters;
- a nonlinear policy is needed when success depends on feature interactions,
  and that need survives out of sample inside the ABM;
- which nonlinear family is better depends on the geometry of the success
  boundary, in the policy layer and in ABM behavior alike;
- learned structure changes behavior only when outcome evidence is shared
  field-wide, and changes entry only when failure is visible early; and
- neighbor-scale evidence makes every feature learner worse than doing
  nothing.

Neural necessity is not established. The neural policy is the better learner
on smooth boundaries and the worse one on axis-aligned boundaries. No model is
selected. The remaining questions are empirical: which boundary geometry and
which evidence regime real pivot outcomes follow.

## Boundary

This scorecard uses deterministic synthetic ABM runs only. It reads zero
empirical rows, performs no empirical fit, changes no generic NABM API, selects
no model, establishes no neural necessity or scientific effect, and authorizes
no package promotion.
