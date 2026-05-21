# Toy 1 First Ablation: Seeds 1-5 Grouped Summary

Source per-run CSV:

`experiments/results/toy1_first_ablation_seeds01_05_summary.csv`

## Grouped Metrics

| Case | Seeds | Accuracy Mean | Accuracy SD | Consensus Mean | Consensus SD | Fragmentation Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `latent_average_state_similarity_same_init` | 5 | 0.889104 | 0.005147 | 0.950259 | 0.011022 | 1.00 |
| `none_none_same_init` | 5 | 0.887050 | 0.005978 | 0.945778 | 0.013116 | 50.00 |
| `output_average_output_similarity_same_init` | 5 | 0.891126 | 0.004016 | 0.959370 | 0.004486 | 1.00 |
| `parameter_average_state_similarity_independent_init` | 5 | 0.887794 | 0.003294 | 0.947597 | 0.005357 | 50.00 |
| `parameter_average_state_similarity_same_init` | 5 | 0.893134 | 0.005661 | 0.976601 | 0.006068 | 1.00 |

## Initial Readout

- `output_average` improves mean accuracy over no-social in this 5-seed batch.
- `latent_average` is slightly above no-social on mean accuracy, but weaker than output mixing here.
- `parameter_average` with `same_init` has the highest mean accuracy and consensus in this batch.
- `parameter_average` with `independent_init` is worse than same-init parameter averaging and remains fragmented under the state-similarity threshold.
- Fragmentation is 50 for no-social because peer sets are intentionally empty for that baseline.

These results validate the ablation workflow and support the next step: threshold and alpha sweeps before stronger claims.

Figure:

`paper/figures/toy1_mixer_comparison.png`
