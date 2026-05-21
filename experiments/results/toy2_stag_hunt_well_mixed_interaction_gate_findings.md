# Toy 2 Stag-Hunt Well-Mixed Interaction Gate Findings

Date: 2026-04-30

This gate tests whether the remaining neural stag-hunt threshold gap comes from the actual spatial payoff/action interaction channel. It keeps `policy_prior=match_p0`, `local_update_rule=counterfactual_advantage`, and `neural_peer_mode=well_mixed`, then compares spatial interactions with epoch-resampled well-mixed interactions.

## Setup

- Label: `toy2_stag_hunt_well_mixed_interaction_gate_seed01_05`
- Regime: `stag_hunt`
- Payoff: `R=4`, `T=3`, `P=2`, `S=0`
- Classical well-mixed unstable threshold: `p*=2/3`
- Grid size: `10 x 10`; spatial graph is still used for cluster metrics
- Neural update: `neural_policy`, `policy_prior=match_p0`, `local_update_rule=counterfactual_advantage`, `neural_peer_mode=well_mixed`
- Interaction modes: `spatial`, `well_mixed_resampled`
- `well_mixed_resampled` samples same-degree interaction peers without self and recomputes the interaction peer set each payoff/action step
- Revision rate: `0.25`
- Selection strength: `1.0`
- Initial cooperation probabilities: `0.55`, `0.60`, `0.65`, `0.70`, `0.75`
- Seeds: `1`, `2`, `3`, `4`, `5` for spatial stochastic runs; RD uses the first seed because it is deterministic
- Epochs: `50`

## Final Cooperation

| Condition | p0=0.55 | p0=0.60 | p0=0.65 | p0=0.70 | p0=0.75 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fermi_imitation`, interaction=`spatial` | 0.442 | 0.464 | 0.502 | 0.632 | 0.708 |
| `fermi_imitation`, interaction=`well_mixed_resampled` | 0.142 | 0.192 | 0.450 | 0.548 | 0.704 |
| `neural_policy`, peer=`well_mixed`, interaction=`spatial` | 0.012 | 0.054 | 0.146 | 0.412 | 0.746 |
| `neural_policy`, peer=`well_mixed`, interaction=`well_mixed_resampled` | 0.014 | 0.048 | 0.148 | 0.406 | 0.740 |
| `rd_well_mixed` | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |

Standard deviations across stochastic seeds:

| Condition | p0=0.55 | p0=0.60 | p0=0.65 | p0=0.70 | p0=0.75 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fermi_imitation`, interaction=`spatial` | 0.123 | 0.085 | 0.114 | 0.103 | 0.126 |
| `fermi_imitation`, interaction=`well_mixed_resampled` | 0.068 | 0.128 | 0.326 | 0.357 | 0.285 |
| `neural_policy`, peer=`well_mixed`, interaction=`spatial` | 0.013 | 0.043 | 0.093 | 0.179 | 0.122 |
| `neural_policy`, peer=`well_mixed`, interaction=`well_mixed_resampled` | 0.011 | 0.046 | 0.086 | 0.180 | 0.138 |

## Acceptance Readout

The target criteria remain: `p0=0.60 < 0.25` and `p0=0.70 > 0.75` for the neural policy.

| Interaction | p0=0.60 | p0=0.70 | p0=0.75 | p0=0.60 passes | p0=0.70 passes | p0=0.75 passes |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `spatial` | 0.054 | 0.412 | 0.746 | True | False | False |
| `well_mixed_resampled` | 0.048 | 0.406 | 0.740 | True | False | False |

## Final Slopes

| Condition | Final Coop Slope | Final Policy Slope |
| --- | ---: | ---: |
| `fermi_imitation`, interaction=`spatial` | 1.400 | 1.375 |
| `fermi_imitation`, interaction=`well_mixed_resampled` | 2.960 | 2.944 |
| `neural_policy`, peer=`well_mixed`, interaction=`spatial` | 3.652 | 3.553 |
| `neural_policy`, peer=`well_mixed`, interaction=`well_mixed_resampled` | 3.620 | 3.494 |
| `rd_well_mixed` | 6.000 | 6.000 |

## Readout

The explicit well-mixed interaction diagnostic still does not restore the RD jump for the neural policy.

- Neural with spatial interactions: `p0=0.60 -> 0.054`, `p0=0.70 -> 0.412`, `p0=0.75 -> 0.746`.
- Neural with well-mixed-resampled interactions: `p0=0.60 -> 0.048`, `p0=0.70 -> 0.406`, `p0=0.75 -> 0.740`.
- RD remains hard: `p0=0.60 -> 0.000`, `p0=0.70 -> 1.000`, `p0=0.75 -> 1.000`.

For the neural condition, changing the realized payoff/action interaction from spatial to well-mixed-resampled barely changes final cooperation. That rules out spatial payoff graph nucleation as the sole remaining cause under this learner setup.

Fermi reacts very differently: well-mixed-resampled interactions increase seed sensitivity around the threshold (`p0=0.65` and `p0=0.70`) rather than producing a deterministic RD-like jump in only `50` epochs with stochastic revision.

## Interpretation

The diagnostic chain now points away from graph locality alone:

1. `match_p0` fixes the erased initial policy signal.
2. `counterfactual_advantage` fixes the sampled-policy-gradient prior-preservation failure.
3. Higher revision sharpens the lower side but can overcollapse the upper side.
4. Well-mixed neural peer context does not restore `p0=0.70`.
5. Well-mixed-resampled payoff/action interactions also do not restore `p0=0.70` for the neural policy.

The remaining mismatch is now most likely in the neural action-sampling/update target loop: the counterfactual update learns a probability field, but action realization remains stochastic and finite-horizon, so `p0=0.70` does not reliably cross into the cooperative basin within `50` epochs. The policy probability slope is strong, but not hard enough to force the sampled action jump at the RD threshold.

## Recommended Next Step

Test the decision kernel directly instead of adding more graph variants:

- Add a diagnostic `dynamics.decision` block with `mode=sampled|argmax` and
  sampled-only `action_temperature` values.
- Keep `policy_prior=match_p0`, `local_update_rule=counterfactual_advantage`, `neural_peer_mode=well_mixed`, `interaction_mode=well_mixed_resampled`, and `revision_rate=0.25`.
- If deterministic/low-temperature action selection restores the threshold, the culprit is stochastic action sampling rather than payoff locality.
- If not, inspect the counterfactual loss target or update horizon/learning-rate schedule.

## Artifacts

- `experiments/results/toy2_stag_hunt_well_mixed_interaction_gate_seed01_05_summary.csv`
- `experiments/results/toy2_stag_hunt_well_mixed_interaction_gate_seed01_05_grouped_summary.csv`
- `experiments/results/toy2_stag_hunt_well_mixed_interaction_gate_seed01_05_grouped_summary.md`
- `experiments/results/toy2_stag_hunt_well_mixed_interaction_gate_seed01_05_epoch_means.csv`
- `experiments/results/toy2_stag_hunt_well_mixed_interaction_gate_seed01_05_epoch_slopes.csv`

Raw run directories were kept under `/tmp/toy2_stag_hunt_well_mixed_interaction_gate_seed01_05/runs`.
