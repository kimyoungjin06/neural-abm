# Toy9-Toy10 Sensitivity Findings

## Commands

```bash
uv run python scripts/run_toy9_sweep.py --label toy9_heterogeneous_sensitivity_seeds01_05 --seeds 1 2 3 4 5 --epochs 25 --mixers output_average --alphas 0 0.25 --threshold-group-fractions 0.25 0.5 0.75 --coordination-gate-modes gated all_enabled all_disabled --environment-thresholds 0.35 0.55 --benefits 1.2 --action-costs 0.25 0.45 --payoff-learning-rates 0.18
uv run python scripts/run_toy10_sweep.py --label toy10_market_sensitivity_seeds01_05 --seeds 1 2 3 4 5 --epochs 25 --mixers output_average --alphas 0 0.25 --recovery-rates 0.02 0.1 --extraction-costs 0.25 0.45 --dynamic-rewire-rates 0 0.05 0.1 --initial-price-expectation-means 0.5 --initial-conservation-norm-means 0.2 0.5
```

## Outputs

- `experiments/results/toy9_heterogeneous_sensitivity_seeds01_05_summary.csv`
- `experiments/results/toy9_heterogeneous_sensitivity_seeds01_05_grouped_summary.csv`
- `experiments/results/toy10_market_sensitivity_seeds01_05_summary.csv`
- `experiments/results/toy10_market_sensitivity_seeds01_05_grouped_summary.csv`

## Coverage

- Toy9: 360 runs, 72 grouped conditions, 5 seeds per condition.
- Toy10: 240 runs, 48 grouped conditions, 5 seeds per condition.

## Toy9 Readout

The strongest Toy9 drivers are threshold/cost and group composition.

| Axis | Mean Action | Mean Payoff | Mean Group Gap | Mean Fragmentation |
| --- | ---: | ---: | ---: | ---: |
| gate `all_disabled` | 0.2973 | 0.3189 | 0.2973 | 100.0 |
| gate `all_enabled` | 0.2951 | 0.3179 | 0.2951 | 1.0 |
| gate `gated` | 0.2877 | 0.3078 | 0.1845 | 5.8667 |
| threshold fraction 0.25 | 0.2088 | 0.2318 | 0.1847 | 38.5333 |
| threshold fraction 0.50 | 0.2780 | 0.3050 | 0.2572 | 34.3333 |
| threshold fraction 0.75 | 0.3933 | 0.4079 | 0.3350 | 34.0000 |

Threshold/cost stress shows the clearest regime shift:

| Threshold | Cost | Mean Action | Mean Payoff | Mean Group Gap |
| ---: | ---: | ---: | ---: | ---: |
| 0.35 | 0.25 | 0.6351 | 0.4618 | 0.5199 |
| 0.35 | 0.45 | 0.3926 | 0.2884 | 0.3561 |
| 0.55 | 0.25 | 0.1156 | 0.2383 | 0.1256 |
| 0.55 | 0.45 | 0.0302 | 0.2710 | 0.0344 |

Output-average alpha `0.25` is modest in this grid: average delta versus
alpha `0.0` is `-0.0079` for final action rate, `-0.0080` for final payoff,
and `-0.0095` for group action-rate gap.

Interpretation: Toy9 is doing its intended job as a heterogeneous-agent stress
case. It exposes group-level rule differences, coordination gating, and
fragmentation diagnostics without changing the public result contract.

## Toy10 Readout

Toy10 is dominated by ecology recovery in this sensitivity range.

| Axis | Mean Resource Fraction | Mean Price | Mean Harvest | Mean Payoff | Mean Rewired Edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| recovery 0.02 | 0.3197 | 0.6910 | 0.3619 | -0.0115 | 129.1500 |
| recovery 0.10 | 0.6667 | 0.6495 | 0.4914 | 0.0960 | 128.9167 |
| rewiring 0.00 | 0.4933 | 0.6702 | 0.4268 | 0.0423 | 0.0000 |
| rewiring 0.05 | 0.4932 | 0.6700 | 0.4278 | 0.0421 | 133.0750 |
| rewiring 0.10 | 0.4932 | 0.6705 | 0.4254 | 0.0424 | 254.0250 |

Extraction cost mostly affects payoff, not resource state, in this grid:

| Extraction Cost | Mean Resource Fraction | Mean Price | Mean Harvest | Mean Payoff |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.4913 | 0.6708 | 0.4267 | 0.0604 |
| 0.45 | 0.4952 | 0.6696 | 0.4266 | 0.0241 |

Initial conservation norm has only a small aggregate effect over 25 epochs:
mean resource fraction rises from `0.4897` at norm `0.2` to `0.4968` at norm
`0.5`. Output-average alpha is also nearly neutral here; all mean deltas are
within about `0.0003`.

Interpretation: Toy10 is exercising multi-channel social messages and dynamic
network churn. The current market/ecology equations make recovery and cost
more important than social averaging, so a future calibration pass should raise
peer-message leverage if Toy10 is meant to be a stronger social-mixing test.

## Follow-Up

- Toy9: use the sensitivity results as the first representative sweep; no
  immediate calibration issue.
- Toy10: social leverage calibration has been completed in
  `experiments/results/toy10_social_calibration_findings.md`; the baseline now
  uses `social_disagreement_penalty=3.0` while preserving legacy defaults.
- Validation: the calibrated representative Toy1-10 suite passed all `78` runs
  under label
  `toy_validation_representative_toy1_10_social_calibrated_seeds01_03`.
