# Toy 2 Stag-Hunt Prior Basin Gate Findings

Date: 2026-04-30

This gate extends the prior diagnostic from a narrow `p0=0.5,0.6,0.7`
comparison to a wider stag-hunt basin grid and adds Fermi/RD references.

## Setup

- Label: `toy2_stag_hunt_prior_basin_gate_seed01_03`
- Regime: `stag_hunt`
- Payoff: `R=4`, `T=3`, `P=2`, `S=0`
- Classical well-mixed unstable threshold: `p*=2/3`
- Grid: `10 x 10` toroidal von Neumann neighborhood
- Revision rate: `0.1`
- Initial cooperation probabilities: `0.1`, `0.3`, `0.5`, `0.6`, `0.7`, `0.9`
- Neural conditions: `default` prior and `match_p0` prior, with learning on/off
- References: Fermi spatial imitation and RD well-mixed
- Seeds: `1`, `2`, `3` for spatial runs
- Epochs: `50`

## Final Cooperation

| Condition | p0=0.1 | p0=0.3 | p0=0.5 | p0=0.6 | p0=0.7 | p0=0.9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `neural_default_no_learn` | 0.543 | 0.543 | 0.543 | 0.543 | 0.547 | 0.547 |
| `neural_default_learn` | 0.520 | 0.527 | 0.527 | 0.527 | 0.533 | 0.543 |
| `neural_match_p0_no_learn` | 0.153 | 0.360 | 0.553 | 0.640 | 0.730 | 0.903 |
| `neural_match_p0_learn` | 0.130 | 0.323 | 0.540 | 0.637 | 0.740 | 0.917 |
| `fermi_imitation` | 0.057 | 0.200 | 0.353 | 0.470 | 0.637 | 0.967 |
| `rd_well_mixed` | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |

Final cooperation slope versus `p0`:

| Condition | Final Coop Slope | Final Policy Slope |
| --- | ---: | ---: |
| `neural_default_no_learn` | 0.005 | 0.000 |
| `neural_default_learn` | 0.025 | 0.055 |
| `neural_match_p0_no_learn` | 0.936 | 1.000 |
| `neural_match_p0_learn` | 0.996 | 1.072 |
| `fermi_imitation` | 1.111 | 1.118 |
| `rd_well_mixed` | 1.388 | 1.388 |

## Readout

The matched policy prior fixes the original `p0` erasure, but it does not by
itself recover the classical stag-hunt basin threshold.

The distinction is important:

- `default` neural prior collapses all initial conditions into a near-`0.5`
  policy readout, which then feeds a similarly ambiguous realized action
  channel. This confirms the earlier diagnosis that random neural
  initialization erases the basin signal.
- `match_p0` preserves the basin signal almost linearly. Low `p0` remains low,
  high `p0` remains high, and learning does not collapse the slope.
- RD well-mixed implements the expected hard threshold for this payoff: `p0=0.6`
  goes to defection, while `p0=0.7` goes to cooperation.
- Fermi spatial imitation is threshold-like but softer and seed-sensitive:
  cooperation increases sharply from `p0=0.6` to `p0=0.7`, but the transition is
  not as hard as RD.
- Neural `match_p0` at `p0=0.6` ends near `0.637`, while RD goes to `0.0`.
  This means the neural local update is not yet applying a classical
  payoff-gradient pressure strong enough to push sub-threshold stag-hunt states
  toward defection.

## Implication

The root failure has two layers:

1. The initial neural policy prior was masking `p0`. The diagnostic prior fixes
   this observability/initialization problem.
2. After that fix, the current local policy-gradient update still behaves more
   like prior-preserving stochastic action revision than like classical
   stag-hunt selection dynamics.

So the next implementation target should not be more social mixing. It should
be a local-update diagnostic or redesign that makes the payoff-gradient channel
explicit enough to reproduce the RD/Fermi threshold behavior when the policy
prior already carries `p0`.

## Recommended Next Step

Add a controlled local-update variant for Toy 2 neural policy, gated behind a
config option, that trains against a counterfactual payoff advantage between
cooperation and defection under the current neighborhood state. The current
REINFORCE-style update only reinforces sampled actions, which can preserve a
sub-threshold prior instead of pushing it toward the payoff-dominant basin.

Minimal diagnostic options:

- Keep `policy_prior_cooperation_probability=match_p0`.
- Add a local advantage target using `Q(C | neighborhood) - Q(D | neighborhood)`.
- Compare the existing sampled-action policy gradient to the counterfactual
  advantage update on the same basin grid.

Acceptance for that next gate:

- `p0=0.6` should move materially toward defection under stag hunt.
- `p0=0.7` should move materially toward cooperation.
- The transition should approach the RD/Fermi reference without relying on
  social mixing.

## Artifacts

- `experiments/results/toy2_stag_hunt_prior_basin_gate_seed01_03_summary.csv`
- `experiments/results/toy2_stag_hunt_prior_basin_gate_seed01_03_grouped_summary.csv`
- `experiments/results/toy2_stag_hunt_prior_basin_gate_seed01_03_grouped_summary.md`
- `experiments/results/toy2_stag_hunt_prior_basin_gate_seed01_03_epoch_means.csv`
- `experiments/results/toy2_stag_hunt_prior_basin_gate_seed01_03_epoch_slopes.csv`

Raw run directories were kept under `/tmp/toy2_stag_hunt_prior_basin_gate_seed01_03/runs`.
