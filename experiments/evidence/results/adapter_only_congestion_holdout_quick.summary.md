# Adapter Congestion Holdout Evidence: adapter_only_congestion_holdout_quick

Status: `pass`

Runs: `experiments/results/nabm_effect_matrix/adapter_only_congestion_holdout_quick_runs.csv`

| Case | Variant | Group | Capacity hits | Max error | Max overcrowding | Mean welfare | Mean TtC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| symmetric_capacity | `imitation_baseline` | baseline | 0 | 10 | 10 | 0.500 |  |
| symmetric_capacity | `global_pressure_negative_control` | negative_control | 0 | 10 | 10 | 0.500 |  |
| symmetric_capacity | `adapter_capacity_policy_main` | main | 3 | 0 | 0 | 1.000 | 1.000 |
| asymmetric_capacity | `imitation_baseline` | baseline | 0 | 14 | 14 | 0.300 |  |
| asymmetric_capacity | `global_pressure_negative_control` | negative_control | 0 | 14 | 14 | 0.300 |  |
| asymmetric_capacity | `adapter_capacity_policy_main` | main | 3 | 0 | 0 | 1.000 | 1.000 |
| noisy_preference_capacity | `imitation_baseline` | baseline | 0 | 12 | 12 | 0.400 |  |
| noisy_preference_capacity | `global_pressure_negative_control` | negative_control | 0 | 12 | 12 | 0.400 |  |
| noisy_preference_capacity | `adapter_capacity_policy_main` | main | 3 | 0 | 0 | 1.000 | 1.000 |

Claim boundary:

> This is a non-cascade adapter-only holdout with baseline, negative control, main variant, and result artifacts. It strengthens extensibility beyond the threshold holdout but remains a tiny scripted binary domain.

