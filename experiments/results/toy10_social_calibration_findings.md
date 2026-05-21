# Toy10 Social Calibration Findings

## Commands

```bash
uv run python scripts/run_toy10_sweep.py --label toy10_market_social_calibration_v2_seeds01_03 --seeds 1 2 3 --epochs 25 --mixers output_average --peer-rules output_similarity --alphas 0 0.25 0.5 --recovery-rates 0.02 0.1 --extraction-costs 0.3 --dynamic-rewire-rates 0.05 --initial-price-expectation-means 0.5 --initial-conservation-norm-means 0.35 --social-harvest-gains 1 --social-disagreement-penalties 0 1.5 --conservation-harvest-weights 0.75
uv run python scripts/run_toy10_sweep.py --label toy10_market_social_calibration_v3_seeds01_03 --seeds 1 2 3 --epochs 25 --mixers output_average --peer-rules output_similarity --alphas 0 0.25 0.5 --recovery-rates 0.02 0.1 --extraction-costs 0.3 --dynamic-rewire-rates 0.05 --initial-price-expectation-means 0.5 --initial-conservation-norm-means 0.35 --social-harvest-gains 1 --social-disagreement-penalties 3 --conservation-harvest-weights 0.75
uv run python scripts/run_toy_validation.py --preset representative --label toy_validation_representative_toy1_10_social_calibrated_seeds01_03 --seeds 1 2 3
```

## Outputs

- `experiments/results/toy10_market_social_calibration_v2_seeds01_03_summary.csv`
- `experiments/results/toy10_market_social_calibration_v2_seeds01_03_grouped_summary.csv`
- `experiments/results/toy10_market_social_calibration_v3_seeds01_03_summary.csv`
- `experiments/results/toy10_market_social_calibration_v3_seeds01_03_grouped_summary.csv`
- `experiments/results/toy_validation_representative_toy1_10_social_calibrated_seeds01_03_runs.csv`
- `experiments/results/toy_validation_representative_toy1_10_social_calibrated_seeds01_03_metrics.csv`
- `experiments/results/toy_validation_representative_toy1_10_social_calibrated_seeds01_03_report.md`

## Calibration Readout

Toy10 now separates three harvest controls:

- `conservation_harvest_weight`: conservation norm pressure in the raw harvest equation.
- `social_harvest_gain`: gain on the mixed-channel harvest shift; default `1.0` preserves the prior output-average behavior.
- `social_disagreement_penalty`: harvest discount from peer/local signal disagreement; default `0.0` preserves legacy configs.

With `social_disagreement_penalty=0.0`, output-average alpha remains nearly neutral in aggregate metrics, matching the earlier sensitivity result. With `social_disagreement_penalty=3.0`, the low-recovery regime becomes a clearer social-coupling diagnostic:

| Recovery | Penalty | Alpha Delta | Resource Fraction Delta | Harvest Delta | Payoff Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 3.0 | 0.5 - 0.0 | +0.00218 | +0.00144 | +0.00051 |
| 0.10 | 3.0 | 0.5 - 0.0 | -0.00007 | +0.00010 | -0.00002 |

Interpretation: Toy10 remains dominated by ecology recovery, but the social path is no longer purely a topology/multi-channel contract check. Low-recovery regimes now show a measurable alpha response while high-recovery regimes stay ecology-dominated.

## Representative Validation

The representative Toy1-10 validation with the calibrated Toy10 baseline passed all `78` runs:

- Toy10 representative mean: resource fraction `0.4475`, market price `0.6930`, harvest intensity `0.4028`, rewired edges `271`.
- The report is available at `experiments/results/toy_validation_representative_toy1_10_social_calibrated_seeds01_03_report.md`.

## Decision

Use `social_disagreement_penalty=3.0` in `experiments/configs/toy10_market_ecology_baseline.yaml`. Existing configs remain source-compatible because the new fields have defaults that preserve previous behavior unless explicitly set.
