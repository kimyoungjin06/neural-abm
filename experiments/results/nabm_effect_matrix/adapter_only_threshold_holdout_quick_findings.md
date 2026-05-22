# Adapter-Only Threshold Holdout Findings

Manifest: `adapter_only_threshold_holdout_quick`

## Purpose

- Test whether a new binary threshold-like domain can use the NABM Unit v1
  policy lifecycle and readiness propagation through adapter callbacks only.
- Keep the holdout outside `src/neural_abm` so the generic unit does not
  absorb domain semantics.
- Include a baseline, a negative control, a main adapter path, and result
  artifacts.

## Result

Gate status: `pass`.

| Case | Variant | Group | Final full hits | Safety hits | Mean final adoption | Mean TtF |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| no_seed_safety | `exposure_baseline` | baseline | 0 | 3 | 0.000 |  |
| no_seed_safety | `thresholdless_global_pressure` | negative_control | 3 | 0 | 20.000 | 1.000 |
| no_seed_safety | `adapter_threshold_readiness` | main | 0 | 3 | 0.000 |  |
| sparse_seed_spread | `exposure_baseline` | baseline | 0 | 0 | 1.000 |  |
| sparse_seed_spread | `thresholdless_global_pressure` | negative_control | 3 | 0 | 20.000 | 1.000 |
| sparse_seed_spread | `adapter_threshold_readiness` | main | 3 | 0 | 20.000 | 5.000 |

## Interpretation

This strengthens the extensibility claim beyond a pure unit smoke test:
a separate holdout manifest now runs a small domain with baseline,
negative-control, and main variants while using only public unit APIs.

The claim remains bounded. This is still a tiny binary holdout, not a
full general-purpose ABM framework demonstration.

## Artifacts

- Runs: `experiments/results/nabm_effect_matrix/adapter_only_threshold_holdout_quick_runs.csv`
- Summary JSON: `experiments/evidence/results/adapter_only_threshold_holdout_quick.summary.json`
- Summary Markdown: `experiments/evidence/results/adapter_only_threshold_holdout_quick.summary.md`
