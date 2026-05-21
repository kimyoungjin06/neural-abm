# Toy 2 Stag-Hunt Well-Mixed Peer Gate Findings

Date: 2026-04-30

This gate tests whether the remaining stag-hunt threshold gap comes from the neural learner seeing spatially local peer contexts. It keeps the actual game/payoff graph spatial, but changes the neural observation and counterfactual local-update peer actions from spatial neighbors to same-degree well-mixed peer samples.

## Setup

- Label: `toy2_stag_hunt_well_mixed_peer_gate_seed01_05`
- Regime: `stag_hunt`
- Payoff: `R=4`, `T=3`, `P=2`, `S=0`
- Classical well-mixed unstable threshold: `p*=2/3`
- Grid/payoff graph: `10 x 10` toroidal von Neumann neighborhood
- Update rule: `neural_policy` with `local_update_rule=counterfactual_advantage`
- Neural peer modes: `spatial`, `well_mixed`
- `well_mixed` samples the same number of peers as the spatial degree, excludes self, and resamples independently of the action-revision RNG stream.
- Policy prior: `match_p0`
- Mixer: `none`
- Revision rate: `0.25`
- Selection strength: `1.0`
- Initial cooperation probabilities: `0.55`, `0.60`, `0.65`, `0.70`, `0.75`
- Spatial seeds: `1`, `2`, `3`, `4`, `5`; RD uses the first seed because it is deterministic for these settings
- Epochs: `50`

## Final Cooperation

| Update Rule | Neural Peer | p0=0.55 | p0=0.60 | p0=0.65 | p0=0.70 | p0=0.75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fermi_imitation` | `spatial` | 0.442 | 0.464 | 0.502 | 0.632 | 0.708 |
| `neural_policy` | `spatial` | 0.096 | 0.146 | 0.326 | 0.576 | 0.780 |
| `neural_policy` | `well_mixed` | 0.012 | 0.054 | 0.146 | 0.412 | 0.746 |
| `rd_well_mixed` | `spatial` | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |

Standard deviations across spatial seeds:

| Update Rule | Neural Peer | p0=0.55 | p0=0.60 | p0=0.65 | p0=0.70 | p0=0.75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fermi_imitation` | `spatial` | 0.123 | 0.085 | 0.114 | 0.103 | 0.126 |
| `neural_policy` | `spatial` | 0.061 | 0.068 | 0.097 | 0.147 | 0.082 |
| `neural_policy` | `well_mixed` | 0.013 | 0.043 | 0.093 | 0.179 | 0.122 |

## Acceptance Readout

The target criteria remain: `p0=0.60 < 0.25` and `p0=0.70 > 0.75`.

| Neural Peer | p0=0.60 | p0=0.70 | p0=0.75 | p0=0.60 passes | p0=0.70 passes | p0=0.75 passes |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `spatial` | 0.146 | 0.576 | 0.780 | True | False | True |
| `well_mixed` | 0.054 | 0.412 | 0.746 | True | False | False |

## Final Slopes

| Update Rule | Neural Peer | Final Coop Slope | Final Policy Slope |
| --- | --- | ---: | ---: |
| `fermi_imitation` | `spatial` | 1.400 | 1.375 |
| `neural_policy` | `spatial` | 3.596 | 3.540 |
| `neural_policy` | `well_mixed` | 3.652 | 3.553 |
| `rd_well_mixed` | `spatial` | 6.000 | 6.000 |

## Readout

The well-mixed peer diagnostic does not restore the RD threshold. It sharpens the lower side further, but it also suppresses `p0=0.70` rather than lifting it.

- Spatial neural peer mode: `p0=0.60 -> 0.146`, `p0=0.70 -> 0.576`, `p0=0.75 -> 0.780`.
- Well-mixed neural peer mode: `p0=0.60 -> 0.054`, `p0=0.70 -> 0.412`, `p0=0.75 -> 0.746`.
- RD well-mixed reference remains hard: `p0=0.60 -> 0.000`, `p0=0.70 -> 1.000`, `p0=0.75 -> 1.000`.

This rules out the simplest hypothesis that local observation/update peer composition alone was hiding a well-mixed threshold. The actual spatial payoff/action channel still matters: even when the learner sees globally shuffled peer contexts, the realized spatial game can fail to nucleate cooperation at global `p0=0.70`.

## Interpretation

The diagnostic chain is now more specific:

1. Random neural initialization erased `p0`; `policy_prior=match_p0` fixed that signal path.
2. Sampled-action policy gradient preserved the prior; `counterfactual_advantage` exposed a stronger stag-hunt basin signal.
3. Increasing revision scope sharpened the lower side but could overcollapse the upper side.
4. Shuffling the neural peer context did not recover RD-like `p0=0.70` cooperation, so the remaining gap is not just local peer-context bias inside the learner.

The most likely remaining mechanism is spatial action/payoff nucleation under finite grids. RD assumes well-mixed frequencies, while Toy 2 still realizes actions and payoffs on a local graph. A global `p0=0.70` can contain enough local defection basins that the learned policy sees, samples, and reinforces collapse in parts of the grid.

## Recommended Next Step

The next clean diagnostic is to add an explicit well-mixed payoff/action channel and then isolate the decision kernel, not just a well-mixed learner context:

- Add a `payoff_graph_mode` or `interaction_mode` diagnostic with `spatial` versus `well_mixed_resampled` interactions.
- Keep `policy_prior=match_p0`, `local_update_rule=counterfactual_advantage`, `neural_peer_mode=well_mixed`, and `revision_rate=0.25`.
- If well-mixed interactions restore the RD jump near `p*=2/3`, the remaining mismatch is spatial nucleation rather than neural learning.
- If they still do not restore it, inspect action sampling temperature/exploration and the counterfactual loss target itself.

## Artifacts

- `experiments/results/toy2_stag_hunt_well_mixed_peer_gate_seed01_05_summary.csv`
- `experiments/results/toy2_stag_hunt_well_mixed_peer_gate_seed01_05_grouped_summary.csv`
- `experiments/results/toy2_stag_hunt_well_mixed_peer_gate_seed01_05_grouped_summary.md`
- `experiments/results/toy2_stag_hunt_well_mixed_peer_gate_seed01_05_epoch_means.csv`
- `experiments/results/toy2_stag_hunt_well_mixed_peer_gate_seed01_05_epoch_slopes.csv`

Raw run directories were kept under `/tmp/toy2_stag_hunt_well_mixed_peer_gate_seed01_05/runs`.
