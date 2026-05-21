# Toy 6-8 Sensitivity Findings

Generated on 2026-05-11.

## Runs

Commands:

```bash
uv run python scripts/run_toy6_sweep.py \
  --label toy6_categorical_sensitivity_seeds01_03 \
  --seeds 1 2 3 \
  --mixers output_average \
  --peer-rules output_similarity \
  --alphas 0 0.5 \
  --thresholds 0 0.8 \
  --payoff-profiles baseline win_bonus loss_heavy \
  --initial-distribution-labels balanced biased

uv run python scripts/run_toy7_sweep.py \
  --label toy7_resource_sensitivity_seeds01_05 \
  --seeds 1 2 3 4 5 \
  --epochs 50 \
  --mixers output_average \
  --peer-rules output_similarity \
  --alphas 0 0.5 \
  --thresholds 0 0.8 \
  --recovery-rates 0.05 \
  --extraction-costs 0.35 \
  --initial-intensity-stds 0.05 0.2 \
  --exploration-stds 0.02 0.08

uv run python scripts/run_toy8_sweep.py \
  --label toy8_async_sensitivity_seeds01_05 \
  --seeds 1 2 3 4 5 \
  --epochs 25 \
  --mixers output_average \
  --peer-rules output_similarity \
  --alphas 0 0.25 0.5 \
  --initial-active-fractions 0.1 0.25 \
  --initial-failed-fractions 0.0 \
  --base-activation-rates 0.02 \
  --peer-activation-rates 0.15 0.3 \
  --failure-rates 0.03 \
  --overload-failure-rates 0.04 0.08 \
  --recovery-rates 0.01 0.05
```

Outputs:

- `experiments/results/toy6_categorical_sensitivity_seeds01_03_summary.csv`
- `experiments/results/toy6_categorical_sensitivity_seeds01_03_grouped_summary.csv`
- `experiments/results/toy7_resource_sensitivity_seeds01_05_summary.csv`
- `experiments/results/toy7_resource_sensitivity_seeds01_05_grouped_summary.csv`
- `experiments/results/toy8_async_sensitivity_seeds01_05_summary.csv`
- `experiments/results/toy8_async_sensitivity_seeds01_05_grouped_summary.csv`

Coverage:

- Toy 6: 72 runs, 3 seeds, 2 initial distributions, 3 payoff profiles, alpha values 0 and 0.5, thresholds 0 and 0.8.
- Toy 7: 80 runs, 5 seeds, 50 epochs, alpha values 0 and 0.5, thresholds 0 and 0.8, initial intensity std values 0.05 and 0.2, exploration std values 0.02 and 0.08.
- Toy 8: 240 runs, 5 seeds, 25 event steps, initial active fractions 0.1 and 0.25, peer activation rates 0.15 and 0.3, overload failure rates 0.04 and 0.08, recovery rates 0.01 and 0.05, alpha values 0, 0.25, and 0.5.

## Toy 6 Sensitivity

Payoff profile is the first meaningful sensitivity lever. Aggregated over initial distribution, alpha, and threshold:

| Payoff Profile | Payoff Mean | Entropy Mean | Dominant Fraction Mean |
| --- | ---: | ---: | ---: |
| baseline | 0.000000 | 0.942493 | 0.461667 |
| loss_heavy | -0.155104 | 0.947143 | 0.447500 |
| win_bonus | 0.156458 | 0.954348 | 0.431667 |

Alpha and threshold remain weak at this sweep scale:

| Alpha | Threshold | Payoff Mean | Entropy Mean | Dominant Fraction Mean | Components Mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.0 | 0.000764 | 0.946337 | 0.447778 | 1.0 |
| 0.0 | 0.8 | 0.000764 | 0.946337 | 0.447778 | 1.0 |
| 0.5 | 0.0 | 0.000139 | 0.949651 | 0.446111 | 1.0 |
| 0.5 | 0.8 | 0.000139 | 0.949651 | 0.446111 | 1.0 |

Interpretation:

- Asymmetric payoff profiles break the payoff-neutral baseline and produce the expected payoff direction.
- The `win_bonus` profile also slightly increases entropy and lowers dominant-strategy concentration, especially under biased initial conditions.
- Threshold 0 versus 0.8 does not change the aggregate result here. Current agents remain similar enough in policy-output space that the threshold does not produce a different peer graph.
- Toy 6 is now better as categorical-action coverage with payoff sensitivity, but still does not yet provide strong social-threshold dynamics.

## Toy 7 Sensitivity

The social and heterogeneity settings remain flat in the current adaptive-intensity formulation. Aggregating over heterogeneity and exploration:

| Alpha | Threshold | Resource Mean | Intensity Mean | Payoff Mean | Components Mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.0 | 0.305338 | 0.434852 | 0.065434 | 1.0 |
| 0.0 | 0.8 | 0.305338 | 0.434852 | 0.065434 | 1.0 |
| 0.5 | 0.0 | 0.305338 | 0.434852 | 0.065434 | 1.0 |
| 0.5 | 0.8 | 0.305338 | 0.434851 | 0.065434 | 1.0 |

Heterogeneity and exploration are also only weakly visible:

| Initial Std | Exploration Std | Resource Mean | Intensity Mean | Payoff Mean |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.02 | 0.305157 | 0.435091 | 0.066405 |
| 0.05 | 0.08 | 0.305515 | 0.434622 | 0.064461 |
| 0.20 | 0.02 | 0.305161 | 0.435082 | 0.066406 |
| 0.20 | 0.08 | 0.305519 | 0.434613 | 0.064462 |

Interpretation:

- Even with 50 epochs and stronger initial heterogeneity, Toy 7 remains dominated by its adaptive resource target and environment parameters.
- Higher exploration slightly lowers payoff, but the effect is small.
- Social output mixing and output-similarity threshold are not currently active levers for Toy 7.

## Toy 8 Sensitivity

Toy 8 adds scheduler/event-queue coverage. In this 25-event sweep, all grouped
conditions consumed the full event budget, so the useful signals are final
state fractions and event type counts rather than absorption.

Initial active fraction is the strongest state-composition lever:

| Initial Active Fraction | Active Fraction Mean | Failed Fraction Mean | Events Mean |
| ---: | ---: | ---: | ---: |
| 0.10 | 0.2507 | 0.0482 | 25.0 |
| 0.25 | 0.3810 | 0.0540 | 25.0 |

Hazard parameters move outcomes in the expected direction:

| Axis | Value | Active Fraction Mean | Failed Fraction Mean | Event Count Mean |
| --- | ---: | ---: | ---: | ---: |
| peer activation | 0.15 | 0.3037 | 0.0567 | 25.0 |
| peer activation | 0.30 | 0.3280 | 0.0455 | 25.0 |
| overload failure | 0.04 | 0.3206 | 0.0491 | 5.1167 failures |
| overload failure | 0.08 | 0.3111 | 0.0531 | 5.5667 failures |
| recovery | 0.01 | 0.3173 | 0.0526 | 0.0833 recoveries |
| recovery | 0.05 | 0.3143 | 0.0496 | 0.3833 recoveries |

Output-average alpha is modest at this short horizon:

| Alpha | Active Fraction Mean | Failed Fraction Mean | Events Mean |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.3195 | 0.0498 | 25.0 |
| 0.25 | 0.3124 | 0.0524 | 25.0 |
| 0.50 | 0.3156 | 0.0511 | 25.0 |

Interpretation:

- Toy 8 is doing the intended job as asynchronous event coverage. It exercises
  event scheduling, stale-event invalidation, hazard recomputation, and
  aggregate event counters.
- Initial state and hazard rates dominate the short-run outcomes. Social
  output mixing is present, but should not be used as the main evidence source
  for coordination effects without a longer horizon or stronger peer-hazard
  coupling.

## Decision

- Keep Toy 6 as categorical-action coverage and use asymmetric payoff profiles in future categorical experiments.
- Keep Toy 7 as continuous-scalar/resource coverage, but do not rely on it for social-mixing evidence without changing the update formulation.
- Keep Toy 8 as async/event-driven coverage. It exposes event scheduling, hazard rates, and event-count diagnostics; its primary sensitivity levers are initial active fraction, peer activation, overload failure, and recovery.
