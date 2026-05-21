# Toy 2 Stag-Hunt Action Channel Gate Findings

Date: 2026-04-30

This gate tests the remaining action-channel hypothesis after the prior,
counterfactual-update, peer-context, and interaction-context diagnostics. It
motivates separating policy readout from realized action selection inside the
Toy 2 NABM unit.

## Setup

- Regime: `stag_hunt`
- Payoff: `R=4`, `T=3`, `P=2`, `S=0`
- Classical well-mixed unstable threshold: `p*=2/3`
- Neural update: `neural_policy`
- Mixer: `none`
- Policy prior: `match_p0`
- Local update rule: `counterfactual_advantage`
- Neural peer mode: `well_mixed`
- Interaction mode: `well_mixed_resampled`
- Revision rate: `0.25`
- Selection strength: `1.0`
- Policy readout temperature: `1.0`
- Decision mode: `sampled`
- Exploration epsilon: `0.0`
- Initial cooperation probabilities: `0.55`, `0.60`, `0.65`, `0.70`, `0.75`
- Seeds: `1`, `2`, `3`, `4`, `5`
- Epochs: `50`

## Gate A: Sampled Action Temperature

Final cooperation means across seeds:

| Action temperature | p0=0.55 | p0=0.60 | p0=0.65 | p0=0.70 | p0=0.75 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1.0` | 0.014 | 0.048 | 0.148 | 0.406 | 0.740 |
| `0.5` | 0.000 | 0.122 | 0.952 | 0.998 | 0.998 |
| `0.25` | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `0.1` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Acceptance readout:

| Action temperature | p0=0.60 | p0=0.70 | Passes `p0=0.60 < 0.25` and `p0=0.70 > 0.75` |
| ---: | ---: | ---: | --- |
| `1.0` | 0.048 | 0.406 | False |
| `0.5` | 0.122 | 0.998 | True |
| `0.25` | 1.000 | 1.000 | False |
| `0.1` | 1.000 | 1.000 | False |

Gate A restores the target threshold at `action_temperature=0.5`. The effect is
not monotone in a useful way: action temperatures that are too low collapse the
lower side upward because `match_p0` starts every policy above `0.5` for these
p0 values, and sharper sampling makes those policies effectively cooperate.

The original diagnostic overloaded `policy_temperature` for this action-channel
probe. The code now expresses the same calibration point as
`dynamics.decision.action_temperature`, leaving `policy_temperature` as the
policy readout/logging/social channel parameter.

## Gate B: Sampled Versus Argmax

Final cooperation means at `policy_temperature=1.0` and
`action_temperature=1.0`:

| Decision mode | p0=0.55 | p0=0.60 | p0=0.65 | p0=0.70 | p0=0.75 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sampled` | 0.014 | 0.048 | 0.148 | 0.406 | 0.740 |
| `argmax` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Acceptance readout:

| Decision mode | p0=0.60 | p0=0.70 | Passes `p0=0.60 < 0.25` and `p0=0.70 > 0.75` |
| --- | ---: | ---: | --- |
| `sampled` | 0.048 | 0.406 | False |
| `argmax` | 1.000 | 1.000 | False |

Argmax proves the upper-side issue is in realized action stochasticity: removing
sampling sends `p0=0.70` to full cooperation. But it overcorrects and also sends
`p0=0.60` to full cooperation, so pure argmax is not the RD-like diagnostic
answer under `policy_prior=match_p0`.

## Interpretation

The remaining mismatch is primarily decision-kernel calibration, not graph
locality or inability to learn the counterfactual advantage. This is the
structural reason for splitting the NABM unit into:

```text
observation -> policy head -> decision kernel -> realized action
```

- At `action_temperature=1.0`, sampled action noise is too soft: `p0=0.70`
  only reaches `0.406` final cooperation.
- At `action_temperature=0.5`, sampled actions recover the target basin split:
  `p0=0.60 -> 0.122`, `p0=0.70 -> 0.998`.
- At `action_temperature <= 0.25`, the decision kernel becomes too hard and
  converts all tested p0 values above `0.5` into cooperation.
- At `argmax`, the same over-hardening appears immediately: all tested p0 values
  go to full cooperation.

Gate C was not run because its trigger condition was not met. Argmax did satisfy
the upper-side criterion (`p0=0.70 > 0.75`), so this is not a finite-horizon
failure of the high-p0 basin.

## Artifacts

- `experiments/results/toy2_stag_hunt_action_channel_gate_a_temp_seed01_05_summary.csv`
- `experiments/results/toy2_stag_hunt_action_channel_gate_a_temp_seed01_05_grouped_summary.csv`
- `experiments/results/toy2_stag_hunt_action_channel_gate_a_temp_seed01_05_grouped_summary.md`
- `experiments/results/toy2_stag_hunt_action_channel_gate_b_argmax_seed01_05_summary.csv`
- `experiments/results/toy2_stag_hunt_action_channel_gate_b_argmax_seed01_05_grouped_summary.csv`
- `experiments/results/toy2_stag_hunt_action_channel_gate_b_argmax_seed01_05_grouped_summary.md`

Raw generated configs, logs, and run directories were kept under
`/tmp/neural_abm_toy2_action_channel`.
