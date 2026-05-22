# Adapter Holdout Evidence: adapter_only_threshold_holdout_quick

Status: `pass`

Runs: `experiments/results/nabm_effect_matrix/adapter_only_threshold_holdout_quick_runs.csv`

| Case | Variant | Group | Final full hits | Safety hits | Mean final adoption | Mean TtF |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| no_seed_safety | `exposure_baseline` | baseline | 0 | 3 | 0.000 |  |
| no_seed_safety | `thresholdless_global_pressure` | negative_control | 3 | 0 | 20.000 | 1.000 |
| no_seed_safety | `adapter_threshold_readiness` | main | 0 | 3 | 0.000 |  |
| sparse_seed_spread | `exposure_baseline` | baseline | 0 | 0 | 1.000 |  |
| sparse_seed_spread | `thresholdless_global_pressure` | negative_control | 3 | 0 | 20.000 | 1.000 |
| sparse_seed_spread | `adapter_threshold_readiness` | main | 3 | 0 | 20.000 | 5.000 |

Claim boundary:

> This is a real adapter-only holdout with baseline, negative control, main variant, and result artifacts. It strengthens the v1 extensibility claim but is still a tiny binary holdout, not a general-purpose ABM framework proof.

