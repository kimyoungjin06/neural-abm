# Adapter-Only Stochastic Commons Holdout Findings

Manifest: `adapter_only_stochastic_commons_quick`

## Purpose

- Test an adapter-only binary ABM with endogenous state transitions.
- Let actions deplete local resources, conservation regenerate them, and
  stochastic shocks perturb local resource stock.
- Keep the holdout outside `src/neural_abm` and use public binary policy
  lifecycle callbacks only.

## Result

Gate status: `pass`.

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

## Interpretation

This moves the adapter-only evidence beyond a fixed threshold or
capacity target. The relevant state is endogenous: harvest decisions
change future resource stock, shocks perturb the environment, and the
main adapter uses local state rather than a direct target mask.

The claim remains bounded. This is still a compact scripted commons
holdout. It is not a full general-purpose ABM framework proof.

## Artifacts

- Runs: `experiments/results/nabm_effect_matrix/adapter_only_stochastic_commons_quick_runs.csv`
- Summary JSON: `experiments/evidence/results/adapter_only_stochastic_commons_quick.summary.json`
- Summary Markdown: `experiments/evidence/results/adapter_only_stochastic_commons_quick.summary.md`
