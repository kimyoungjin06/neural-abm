# Adapter-Only Congestion Holdout Findings

Manifest: `adapter_only_congestion_holdout_quick`

## Purpose

- Test a non-cascade binary holdout where the target is capacity-matched
  allocation rather than full adoption.
- Keep the domain outside `src/neural_abm` and use public binary policy
  lifecycle callbacks only.
- Compare imitation baseline, global-pressure negative control, and an
  adapter-owned capacity policy.

## Result

Gate status: `pass`.

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

## Interpretation

This strengthens the adapter-only claim in a direction that is not
threshold-cascade isomorphic. The target is a capacity-matched binary
allocation, and success is measured by capacity error and overcrowding
rather than full adoption.

The claim remains bounded. This is still a tiny scripted holdout, not
a full general-purpose ABM framework demonstration.

## Artifacts

- Runs: `experiments/results/nabm_effect_matrix/adapter_only_congestion_holdout_quick_runs.csv`
- Summary JSON: `experiments/evidence/results/adapter_only_congestion_holdout_quick.summary.json`
- Summary Markdown: `experiments/evidence/results/adapter_only_congestion_holdout_quick.summary.md`
