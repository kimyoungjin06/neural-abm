# Pivot adaptive baseline ladder v1 result

Date: 2026-08-07

Status: `DESCRIPTIVE_SYNTHETIC_MODEL_COMPARISON_REPRODUCED`

## Result

The fixed-rule ABM cannot reproduce the Pivot visibility response, but the
core delayed-feedback surface does not require neural feature weights. A
single-state explicit delta rule reproduces the directions of overshoot,
delayed purge, persistent failure under selectively missing evidence, and the
cautionary exploration trade-off. Neural imitation changes several magnitudes
and immediate-regime composition, but neural necessity is not supported by
this synthetic scorecard.

The development comparison contains 320 paired rows across five models, four
visibility regimes, and 16 replicates. Fixed and neural rows exactly nest their
tracked predecessors. A second deterministic execution reproduced all four
artifacts byte for byte.

## Imitative comparison

| Visibility | Model | Entry | Active | Active productive | Active failed | Exited failed |
|---|---|---:|---:|---:|---:|---:|
| Immediate | Fixed | 0.807 | 0.304 | 0.242 | 0.062 | 0.503 |
| Immediate | Delta | 0.827 | 0.347 | 0.258 | 0.089 | 0.480 |
| Immediate | Neural | 0.798 | 0.378 | 0.252 | 0.126 | 0.420 |
| Failure lag 4 | Delta | 0.946 | 0.285 | 0.273 | 0.012 | 0.661 |
| Failure lag 4 | Neural | 0.960 | 0.285 | 0.276 | 0.009 | 0.675 |
| Selective failure lag 4 | Delta | 0.949 | 0.367 | 0.273 | 0.094 | 0.582 |
| Selective failure lag 4 | Neural | 0.961 | 0.379 | 0.276 | 0.103 | 0.582 |

Relative to each model's immediate regime, four-step failure delay raises
cumulative entry by `0.119` under delta imitation and `0.162` under neural
imitation. Failed exit rises by `0.180` and `0.255`; final active failed stock
falls by `0.077` and `0.116`. All directions hold in all 16 replicates for both
adaptive families.

When half of delayed failures are permanently unavailable, active failed stock
rises by `0.082` relative to full delay under delta learning and `0.094` under
neural learning. Both therefore produce the same central phase pattern. The
neural policy amplifies it, but does not uniquely create it.

## Where neural and delta differ

The largest delta-neural difference occurs under immediate imitation. Final
vector mean absolute distance is `0.040`, versus `0.006` under two-step failure
delay, `0.008` under four-step delay, and `0.014` under selective four-step
delay. Immediate neural imitation retains more active failed pivots (`0.126`
versus `0.089`) and fewer exits (`0.420` versus `0.480`).

Under four-step delay, neural imitation produces about `0.014` more cumulative
entry and failed exit than delta imitation. This shows feature-specific neural
weights affect amplification and composition. It does not show that those
differences are scientifically useful.

## Cautionary comparison

The nonneural and neural cautionary models are effectively the same on the
registered outputs. Their final vectors and trajectories are exactly identical
under complete two-step and four-step failure delay. The immediate final-vector
distance is `0.0004`, and the selective-delay distance is `0.0030`.

Thus the cautionary correction-versus-productive-exploration trade-off does
not require neural feature weights in this case. A scalar failure-driven delta
state is sufficient to reproduce it.

## NABM conclusion

This result narrows the claim appropriately:

- adaptation is necessary for the registered visibility-dependent surface;
- neural adaptation is not yet necessary for that surface;
- neural and delta policies are not behaviorally identical everywhere; and
- no model can be selected without disjoint predictive or empirical targets.

The next comparison must be precommitted and out of the current synthetic
surface: either a disjoint scenario-generalization scorecard or an empirical
predictive target with honest applicability. If neural differences do not
improve those targets, the simpler delta model should be preferred. No further
in-sample tuning of this ladder is warranted.

## Boundary

This scorecard uses deterministic synthetic model behavior only. It reads zero
empirical rows, performs no empirical fit or predictive validation, changes no
generic NABM API, selects no model, establishes no neural necessity or
scientific effect, and authorizes no package promotion.
