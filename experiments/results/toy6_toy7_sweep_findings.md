# Toy 6 and Toy 7 Sweep Findings

Generated on 2026-05-11.

## Runs

Commands:

```bash
uv run python scripts/run_toy6_sweep.py
uv run python scripts/run_toy7_sweep.py
```

Outputs:

- `experiments/results/toy6_categorical_sweep_seeds01_05_summary.csv`
- `experiments/results/toy6_categorical_sweep_seeds01_05_grouped_summary.csv`
- `experiments/results/toy7_resource_sweep_seeds01_05_summary.csv`
- `experiments/results/toy7_resource_sweep_seeds01_05_grouped_summary.csv`

Coverage:

- Toy 6: 40 runs, 5 seeds, 2 initial distributions, no-social baseline, and output-average alpha values 0, 0.25, 0.5.
- Toy 7: 180 runs, 5 seeds, 3 recovery rates, 3 extraction costs, no-social baseline, and output-average alpha values 0, 0.25, 0.5.

## Toy 6 Readout

The categorical spatial game is deterministic enough for a compact sweep signal, but the current cyclic payoff is payoff-neutral at the aggregate level:

| Initial Distribution | Mixer | Alpha | Payoff Mean | Entropy Mean | Dominant Fraction Mean | Components Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| balanced | none | 0 | 0.000000 | 0.995721 | 0.370 | 100.0 |
| balanced | output_average | 0 | 0.000000 | 0.995721 | 0.370 | 1.0 |
| balanced | output_average | 0.25 | 0.000000 | 0.992925 | 0.382 | 1.0 |
| balanced | output_average | 0.50 | 0.000000 | 0.993910 | 0.376 | 1.0 |
| biased | none | 0 | 0.000000 | 0.895853 | 0.542 | 100.0 |
| biased | output_average | 0 | 0.000000 | 0.895853 | 0.542 | 1.0 |
| biased | output_average | 0.25 | 0.000000 | 0.896391 | 0.534 | 1.0 |
| biased | output_average | 0.50 | 0.000000 | 0.896082 | 0.538 | 1.0 |

Interpretation:

- The initial strategy distribution is the visible driver: balanced starts stay high-entropy, while biased starts keep a larger dominant-strategy fraction.
- Alpha has only a small effect on entropy and dominant fraction under the current cyclic payoff and output-similarity settings.
- Fragmentation differs sharply by mixer because output-average creates one active peer component, while no-social leaves the 100 agents isolated in the active-peer graph.
- Toy 6 is useful as categorical-action coverage, but it needs an asymmetric payoff or stronger peer-threshold sweep before it can serve as a strong social-influence result.

## Toy 7 Readout

The resource model produces a clear environment response. Aggregating over mixer and alpha, final resource fraction rises from 0.067780 at low recovery and low extraction cost to 0.551623 at high recovery and high extraction cost.

| Recovery | Extraction Cost | Resource Fraction Mean | Intensity Mean | Payoff Mean |
| ---: | ---: | ---: | ---: | ---: |
| 0.02 | 0.25 | 0.067780 | 0.176315 | 0.003253 |
| 0.02 | 0.35 | 0.118620 | 0.220312 | 0.008992 |
| 0.02 | 0.50 | 0.206148 | 0.256605 | 0.020976 |
| 0.05 | 0.25 | 0.211747 | 0.451355 | 0.042964 |
| 0.05 | 0.35 | 0.284892 | 0.432970 | 0.057076 |
| 0.05 | 0.50 | 0.379475 | 0.404999 | 0.072043 |
| 0.10 | 0.25 | 0.372078 | 0.769969 | 0.137145 |
| 0.10 | 0.35 | 0.458282 | 0.665134 | 0.149115 |
| 0.10 | 0.50 | 0.551623 | 0.560868 | 0.151903 |

The alpha response is effectively flat at this sweep scale:

| Mixer | Alpha | Resource Mean | Intensity Mean | Payoff Mean |
| --- | ---: | ---: | ---: | ---: |
| none | 0 | 0.294516 | 0.437613 | 0.071496 |
| output_average | 0 | 0.294516 | 0.437613 | 0.071496 |
| output_average | 0.25 | 0.294516 | 0.437615 | 0.071496 |
| output_average | 0.50 | 0.294516 | 0.437616 | 0.071496 |

Interpretation:

- Recovery rate and extraction cost are the dominant levers in the current Toy 7 setup.
- Higher recovery supports higher intensity and payoff. Higher extraction cost preserves more resource and, in this payoff formulation, often improves final payoff through resource availability.
- Output-average alpha does not materially alter final resource, intensity, or payoff in the current baseline. Toy 7 is therefore good continuous-scalar/resource coverage, but it needs a stronger social-sensitivity setting before being used as evidence for social mixing effects.

## Next Steps

- Add a small Toy 6 sensitivity sweep with asymmetric payoff weights or nonzero output-similarity thresholds.
- Add a Toy 7 social-sensitivity sweep with stronger initial heterogeneity, threshold variation, and possibly longer horizons.
- After those narrow checks, proceed to Toy 8 as an async/event-driven ABM to cover scheduler and event-queue dynamics that Toy1-7 do not exercise.
