# Toy4 Heterogeneous Local Observation Stress Findings

Date: 2026-05-21

## Question

Does the Toy4 local resource-threshold path remain stable when three stress
axes are combined?

- resource-coupled Toy4 with initial contribution probability `0.35`
- checkerboard heterogeneous extraction with heterogeneity `1.0`
- noisy reputation `2.0` for the reputation diagnostic and neural comparison
  variants
- local-threshold neural variants with `global`, `hidden`, and
  `local_sustain` resource observation modes

This extends the earlier local-observation stress by making local sustain
thresholds differ across space. The stress remains Toy4-specific; no generic
NABM unit semantics are changed.

## Artifacts

- Manifest:
  `experiments/evidence/toy4_resource_threshold_heterogeneous_local_observation_stress_quick.yaml`
- Run rows:
  `experiments/results/nabm_effect_matrix/toy4_hetero_local_obs_stress_quick_runs.csv`
- Gate summary:
  `experiments/evidence/results/toy4_hetero_local_obs_stress_quick.summary.json`
- Profile:
  `experiments/results/nabm_effect_matrix/toy4_hetero_local_obs_stress_quick_profile.json`

## Result

The gate passed. Clean reputation imitation remains faster, but the noisy
reputation diagnostic again drops to `3/5` final ceiling hits. The local
resource-threshold neural variants remain stable across all five seeds.

| Variant | Group | Final ceiling hits | Ever ceiling hits | Mean TtC | Mean final payoff | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `rep_clean_hetero` | baseline | 5/5 | 5/5 | 15.0 | 0.600000 | 1.000 |
| `rep_noisy_s2p0_hetero` | diagnostic | 3/5 | 3/5 | 47.333 | 0.494039 | 0.600 |
| `rev_pop_global_obs_noisy_s2p0_hetero` | diagnostic | 0/5 | 0/5 | n/a | -0.302000 | 0.000 |
| `rev_local_global_obs_noisy_s2p0_hetero` | diagnostic | 5/5 | 5/5 | 33.0 | 0.600000 | 1.000 |
| `rev_local_hidden_obs_noisy_s2p0_hetero` | diagnostic | 5/5 | 5/5 | 32.2 | 0.600000 | 1.000 |
| `rev_local_sustain_obs_noisy_s2p0_hetero` | main | 5/5 | 5/5 | 31.8 | 0.600000 | 1.000 |

## Interpretation

This is useful but still bounded evidence.

What strengthened:

1. The local-threshold neural path survives spatially heterogeneous resource
   extraction and noisy reputation ranking together.
2. The noisy reputation diagnostic remains seed-fragile at `3/5`, while the
   local-threshold variants remain `5/5`.
3. The population-threshold negative control remains `0/5`, preserving the
   distinction between local resource thresholding and generic environment
   weighting.

What did not strengthen:

1. Clean reputation imitation is still `5/5` and much faster at mean TtC
   `15.0`.
2. `local_sustain` is only slightly faster than hidden/global observation
   (`31.8` versus `32.2` and `33.0`). This supports it as a domain-native main
   candidate, but not as a necessary observation feature.
3. The baseline-breaking contrast still depends on reputation noise. The
   resource heterogeneity itself is a robustness stress, not a clean baseline
   breaker.

## Decision

Keep this as the current strongest Toy4 resource-local robustness artifact:
local thresholding plus precommitment/peer evidence remains stable when local
resource damage differs across space and reputation ordering is noisy.

Do not overclaim general superiority. The clean hand-coded reputation rule is
still faster in clean ranking conditions.

The next useful step is not another observation-mode variant. It should either:

- move this result into the manuscript claim matrix as a bounded Toy4
  robustness claim; or
- design a non-reputation baseline-fragility stress where clean reputation
  imitation fails because the environment objective changes, not because its
  ranking signal is externally noised.

## Verification

- `uv run pytest tests/test_evidence_gate.py::test_toy4_resource_threshold_heterogeneous_local_observation_contract -q`
- `uv run python scripts/run_basin_credit_evidence_workflow.py --manifest experiments/evidence/toy4_resource_threshold_heterogeneous_local_observation_stress_quick.yaml`
