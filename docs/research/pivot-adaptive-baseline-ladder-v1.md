# Pivot adaptive baseline ladder v1

Date: 2026-08-07

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_adaptive_baseline_ladder_v1.yaml`](../../experiments/configs/pivot_adaptive_baseline_ladder_v1.yaml)

## Purpose

The persistence-and-exit prototype showed a nontrivial Pivot surface:
overshoot followed by purge, persistent failure under selectively missing
evidence, survivor-composition reversal, and similar final stocks produced by
different entry and exit paths. The next question is not whether another Pivot
state should be added. It is whether a neural policy is needed to produce this
surface.

This ladder compares the current fixed rule, an explicit one-state nonneural
delta rule, and the neural logistic policy under the same synthetic
environment. It is a generative mechanism comparison, not empirical fitting,
predictive validation, or model selection.

## Fair comparison

Every model uses the same sampled population, network, local shocks,
`hot_field_hype` scenario, readiness channel, entry threshold, private outcome,
public visibility queue, persistence score, exit rule, horizon, and paired
replicate identifiers. Only the policy update changes.

The five registered models are:

| Model | Adaptive state | Public evidence |
|---|---:|---|
| `fixed_rule` | 0 | None |
| `delta_imitative` | 1 scalar | Neighbor success and failure |
| `delta_cautionary` | 1 scalar | Neighbor failure only |
| `neural_imitative` | 9 feature weights plus bias | Neighbor success and failure |
| `neural_cautionary` | 9 feature weights plus bias | Neighbor failure only |

Fixed and neural rows must exactly nest their tracked predecessors. Any
difference in final target values or trajectories is an implementation failure.

## Explicit delta baseline

Each delta agent starts from its own frozen logistic policy output `b0`. When a
nonempty batch of eligible neighbor outcomes becomes public, the agent takes
the mean zero-one productivity label as its target and updates:

```text
b_next = clip(
    b + 0.5 * (target - b) + 0.25 * (b0 - b),
    0,
    1,
)
```

The cautionary version first filters the batch to failed events, making its
target zero. Self outcomes are excluded. No update occurs without evidence.
The scalar rule has no feature-specific adaptive weights, gradient, loss
function, backpropagation, or hidden state. Its local readiness and social
mixing lifecycle otherwise matches the neural agent.

The learning rate and prior anchor are fixed before results. They are not tuned
to match the neural output, and no post-result sweep is authorized in this
tranche.

## Common targets

The final target vector contains cumulative entry, active stock, active
productive and failed stocks, failed exit, and productive share among active
agents. Complete entry, productive, failed, and exit trajectories are retained.

The scorecard reports paired model-minus-fixed and delta-minus-neural
contrasts, final-vector mean absolute distance, and trajectory root mean
squared distance. These are descriptive distances with no acceptance threshold.
No model is selected because no empirical or held-out predictive target is
used.

Registered outcomes include scalar-delta reproduction of the neural phase
directions, matching of stocks but not flows, divergence by visibility,
nonneural reproduction of the cautionary trade-off, distinct feature-specific
neural behavior, sufficiency of the fixed rule, and no close family across all
targets.

## Boundary and next decision

If the delta rule reproduces the important surface, the scientific result is
that neural weights are not necessary for this generative case tranche. If it
does not, the difference remains a synthetic mechanism contrast until tested
on disjoint empirical or predictive targets. Either result can guide NABM
improvement without declaring a winner.

All policy meanings remain case-owned. The ladder changes no generic NABM API,
reads no empirical rows, performs no empirical fit or predictive validation,
selects no model, establishes no neural necessity or scientific effect, and
authorizes no package promotion.
