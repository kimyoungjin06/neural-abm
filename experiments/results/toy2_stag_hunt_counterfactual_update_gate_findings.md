# Toy 2 Stag-Hunt Counterfactual Update Gate Findings

Date: 2026-04-30

This gate tests whether a local neural update that sees the counterfactual
payoff advantage `Q(C | neighborhood) - Q(D | neighborhood)` can recover a more
classical stag-hunt basin response once the policy prior already carries `p0`.

## Setup

- Label: `toy2_stag_hunt_counterfactual_update_gate_seed01_03`
- Regime: `stag_hunt`
- Payoff: `R=4`, `T=3`, `P=2`, `S=0`
- Classical well-mixed unstable threshold: `p*=2/3`
- Grid: `10 x 10` toroidal von Neumann neighborhood
- Revision rate: `0.1`
- Policy prior: `match_p0`
- Local neural updates:
  - `sampled_policy_gradient`: existing sampled-action REINFORCE-style update
  - `counterfactual_advantage`: trains every agent each epoch on local
    counterfactual payoff advantage
- Initial cooperation probabilities: `0.1`, `0.3`, `0.5`, `0.6`, `0.7`, `0.9`
- References: Fermi spatial imitation and RD well-mixed
- Seeds: `1`, `2`, `3` for spatial runs
- Epochs: `50`

## Final Cooperation

| Condition | p0=0.1 | p0=0.3 | p0=0.5 | p0=0.6 | p0=0.7 | p0=0.9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `neural_match_p0_sampled_policy_gradient` | 0.130 | 0.323 | 0.540 | 0.637 | 0.740 | 0.917 |
| `neural_match_p0_counterfactual_advantage` | 0.010 | 0.027 | 0.143 | 0.273 | 0.630 | 0.983 |
| `fermi_imitation` | 0.057 | 0.200 | 0.353 | 0.470 | 0.637 | 0.967 |
| `rd_well_mixed` | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |

Final cooperation slope versus `p0`:

| Condition | Final Coop Slope | Final Policy Slope |
| --- | ---: | ---: |
| `sampled_policy_gradient` | 0.996 | 1.072 |
| `counterfactual_advantage` | 1.232 | 1.229 |
| `fermi_imitation` | 1.111 | 1.118 |
| `rd_well_mixed` | 1.388 | 1.388 |

## Readout

The counterfactual local update substantially improves the stag-hunt basin
shape, but it still does not exactly reproduce the hard RD threshold.

Key comparison:

- Existing sampled update at `p0=0.6`: `0.637`
- Counterfactual update at `p0=0.6`: `0.273`
- Fermi reference at `p0=0.6`: `0.470`
- RD reference at `p0=0.6`: `0.000`

The counterfactual update now pushes the sub-threshold `p0=0.6` case materially
toward defection. This was the failure mode in the prior gate.

At the upper side of the threshold:

- Existing sampled update at `p0=0.7`: `0.740`
- Counterfactual update at `p0=0.7`: `0.630`
- Fermi reference at `p0=0.7`: `0.637`
- RD reference at `p0=0.7`: `1.000`

The counterfactual update is close to Fermi at `p0=0.7`, but remains much softer
than RD. Spatial stochasticity and seed sensitivity are visible: individual
counterfactual runs at `p0=0.7` ended at `0.48`, `0.80`, and `0.61`.

## Implementation Note

The first counterfactual attempt updated only revised agents, mirroring the
sampled-action learner. That was too weak: each agent trained only a few times
over `50` epochs at revision rate `0.1`. The final diagnostic variant trains
every neural agent each epoch because the counterfactual update does not depend
on a sampled action. Revision rate still controls action resampling, but the
policy learns from the full local payoff-gradient field.

## Interpretation

The Toy 2 failure is now decomposed into two concrete issues:

1. Random neural initialization erased `p0`; `match_p0` fixes that diagnostic
   channel.
2. Sampled-action policy gradient preserved the prior instead of expressing the
   stag-hunt payoff threshold; counterfactual local advantage restores a much
   stronger threshold-like response.

The remaining gap is the hardness and location of the threshold. RD jumps from
defection to cooperation between `p0=0.6` and `p0=0.7`; counterfactual neural
policy becomes Fermi-like and spatially seed-sensitive rather than RD-like.

## Recommended Next Step

Keep `counterfactual_advantage` as a diagnostic local update option. The next
gate should tune update strength and scope rather than social mixing:

- Sweep `selection_strength`: for example `1.0`, `2.0`, `4.0`
- Sweep `revision_rate`: `0.1`, `0.25`, `1.0`
- Keep `policy_prior=match_p0`
- Focus on p0 near the threshold: `0.55`, `0.6`, `0.65`, `0.7`, `0.75`

Acceptance for the next gate:

- Below-threshold cases near `0.6` should consistently fall below `0.25`.
- Above-threshold cases near `0.7` should consistently rise above `0.75`.
- The transition should become less seed-sensitive without requiring social
  mixing.

## Artifacts

- `experiments/results/toy2_stag_hunt_counterfactual_update_gate_seed01_03_summary.csv`
- `experiments/results/toy2_stag_hunt_counterfactual_update_gate_seed01_03_grouped_summary.csv`
- `experiments/results/toy2_stag_hunt_counterfactual_update_gate_seed01_03_grouped_summary.md`
- `experiments/results/toy2_stag_hunt_counterfactual_update_gate_seed01_03_epoch_means.csv`
- `experiments/results/toy2_stag_hunt_counterfactual_update_gate_seed01_03_epoch_slopes.csv`

Raw run directories were kept under `/tmp/toy2_stag_hunt_counterfactual_update_gate_seed01_03/runs`.
