# Adapter Stochastic Commons Holdout Evidence: adapter_only_stochastic_commons_quick

Status: `pass`

Runs: `experiments/results/nabm_effect_matrix/adapter_only_stochastic_commons_quick_runs.csv`

| Case | Variant | Group | Min resource | Max collapse epochs | Mean welfare | Mean harvest | Recovery hits | Max recovery |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| steady_regen_commons | `greedy_harvest_baseline` | baseline | 0.040 | 17 | -0.136 | 1.000 | 0 |  |
| steady_regen_commons | `global_pressure_negative_control` | negative_control | 0.316 | 6 | 0.076 | 0.385 | 0 |  |
| steady_regen_commons | `adapter_local_resource_main` | main | 0.458 | 0 | 0.067 | 0.300 | 0 |  |
| localized_resource_shock | `greedy_harvest_baseline` | baseline | 0.027 | 19 | -0.143 | 1.000 | 0 |  |
| localized_resource_shock | `global_pressure_negative_control` | negative_control | 0.316 | 5 | 0.071 | 0.353 | 3 | 2 |
| localized_resource_shock | `adapter_local_resource_main` | main | 0.439 | 0 | 0.069 | 0.312 | 3 | 2 |
| heterogeneous_need_commons | `greedy_harvest_baseline` | baseline | 0.028 | 19 | -0.141 | 1.000 | 0 |  |
| heterogeneous_need_commons | `global_pressure_negative_control` | negative_control | 0.282 | 7 | 0.076 | 0.395 | 3 | 5 |
| heterogeneous_need_commons | `adapter_local_resource_main` | main | 0.485 | 0 | 0.066 | 0.288 | 3 | 1 |

Claim boundary:

> This is a closed-loop adapter-only holdout with baseline, negative control, main variant, and result artifacts. It strengthens extensibility beyond threshold and capacity scripts but remains a compact scripted binary commons, not a general-purpose ABM framework proof.

