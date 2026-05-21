# Toy 2 Initial Findings

Toy model:

`Neural Spatial Prisoner's Dilemma`

This is the first executable Toy 2 result. Treat it as a runner and logging
validation, not as a robust game-dynamics claim.

## Setup

- Grid: `10 x 10` toroidal von Neumann neighborhood.
- Agent model: policy MLP `6 -> 16 -> 2`.
- Init mode: `independent_init`.
- Epochs: `50`.
- Seed: `1`.
- Payoff table: `R=3`, `S=0`, `T=5`, `P=1`.

## Runs

| Run | Mixer | Peer Rule | Final Cooperation | Final Mean Payoff | Final Policy Cooperation | Final Fragmentation |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `toy2_spatial_pd_baseline` | `none` | `none` | 0.120000 | 1.360000 | 0.128556 | 100 |
| `toy2_spatial_pd_output_average` | `output_average` | `none` | 0.060000 | 1.175000 | 0.065007 | 1 |

## Interpretation

- The Toy 2 runner writes config-driven aggregate and micro-state logs.
- No-social and output policy mixing both complete through the same simulation
  loop.
- In this single seed, output policy mixing pushes policy cooperation down
  faster than no-social and ends with lower realized cooperation and mean
  payoff.
- This is not yet a stable conclusion about policy mixing. It is the first
  sanity check showing that Toy 2 produces measurable game-dynamics changes.

Figure:

`paper/figures/toy2_initial_policy_mixing.png`

## Source Runs

| Run | Directory |
| --- | --- |
| `toy2_spatial_pd_baseline` | `experiments/runs/20260429_195814_toy2_spatial_pd_baseline_seed01` |
| `toy2_spatial_pd_output_average` | `experiments/runs/20260429_195934_toy2_spatial_pd_output_average_seed01` |

## Next Steps

1. Add a Toy 2 ablation script for `none` versus `output_average` across seeds.
2. Sweep `alpha` for output policy mixing.
3. Add a strategy-imitation baseline before learned edge functions.
4. Add cooperation-cluster plots once multi-seed behavior is known.
