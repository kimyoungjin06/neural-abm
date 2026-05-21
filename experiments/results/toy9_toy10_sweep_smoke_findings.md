# Toy9-Toy10 Sweep Smoke Findings

## Commands

```bash
uv run python scripts/run_toy9_sweep.py --label toy9_heterogeneous_sweep_smoke_seeds01_02 --seeds 1 2 --epochs 5 --mixers output_average --alphas 0 0.25 --threshold-group-fractions 0.25 0.75 --coordination-gate-modes gated --environment-thresholds 0.45 --benefits 1.2 --action-costs 0.35 --payoff-learning-rates 0.18
uv run python scripts/run_toy10_sweep.py --label toy10_market_sweep_smoke_seeds01_02 --seeds 1 2 --epochs 5 --mixers output_average --alphas 0 0.25 --recovery-rates 0.02 0.1 --extraction-costs 0.3 --dynamic-rewire-rates 0 0.1 --initial-price-expectation-means 0.5 --initial-conservation-norm-means 0.35
```

## Outputs

- `experiments/results/toy9_heterogeneous_sweep_smoke_seeds01_02_summary.csv`
- `experiments/results/toy9_heterogeneous_sweep_smoke_seeds01_02_grouped_summary.csv`
- `experiments/results/toy10_market_sweep_smoke_seeds01_02_summary.csv`
- `experiments/results/toy10_market_sweep_smoke_seeds01_02_grouped_summary.csv`

## Toy9 Smoke Readout

- Runs: 8 total, 2 seeds per grouped condition.
- The threshold-group fraction axis changes the social peer graph diagnostic:
  `threshold_group_fraction=0.25` produced mean final fragmentation `16.5`,
  while `0.75` produced `1.0` in this short smoke.
- Mean final action rates stayed close across this small grid: `0.300` to
  `0.310`.
- The grouped action-rate gap was larger with fewer threshold-rule agents:
  `0.04-0.06` at fraction `0.25` versus about `0.013` at fraction `0.75`.

## Toy10 Smoke Readout

- Runs: 16 total, 2 seeds per grouped condition.
- Higher recovery rate raised final resource fraction from about `0.722` to
  about `0.788`.
- Dynamic rewiring correctly exercised the topology churn path: `0.1` rewire
  rate produced mean cumulative rewired edge count `49.5`; `0.0` produced `0`.
- In this short grid, output-average alpha `0.25` moved final market/resource
  metrics only slightly relative to `0.0`, which is acceptable for a smoke
  check and leaves room for wider sensitivity sweeps.

## Next Sensitivity Axes

- Toy9: add `coordination_gate_mode in {gated, all_enabled, all_disabled}`,
  wider `environment_threshold`, and benefit/cost stress cases.
- Toy10: cross dynamic rewiring with extraction cost and initial conservation
  norm; this should better separate market pressure from ecology recovery.
