# Toy4 Resource Threshold Sparse Resource Observation Stress Findings

Date: 2026-05-21

## Question

Does the local resource-threshold neural path depend on directly observing the
global resource stock, or can it recover from the resource-coupled Toy4 stress
with hidden or local resource information?

This stress changes only the Toy4 neural observation feature that previously
held global resource fraction:

- `global`: original global resource fraction
- `hidden`: constant `1.0`, hiding the direct stock signal
- `local_sustain`: local contribution rate divided by the local sustain rate

The observation dimension remains unchanged. The environment dynamics and
local-threshold credit are still Toy4-specific; no shared NABM unit semantics
are changed.

## Artifacts

- Manifest:
  `experiments/evidence/toy4_resource_threshold_sparse_resource_observation_stress_quick.yaml`
- Run rows:
  `experiments/results/nabm_effect_matrix/toy4_resource_threshold_sparse_resource_observation_stress_quick_runs.csv`
- Gate summary:
  `experiments/evidence/results/toy4_resource_threshold_sparse_resource_observation_stress_quick.summary.json`
- Profile:
  `experiments/results/nabm_effect_matrix/toy4_resource_threshold_sparse_resource_observation_stress_quick_profile.json`

## Result

The gate passed. All local-threshold neural variants reached and held the
ceiling, including the hidden-resource observation condition.

| Variant | Group | Final ceiling hits | Ever ceiling hits | Mean TtC | Mean final payoff | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `reputation_imitation_resource_p0p35_clean` | baseline | 5/5 | 5/5 | 15.0 | 0.600000 | 1.000 |
| `revision_precommitment_peer_evidence_resource_threshold_population_envw2p0_global_observation` | diagnostic | 0/5 | 0/5 |  | -0.298000 | 0.000 |
| `revision_precommitment_peer_evidence_resource_threshold_local_envw2p0_global_observation` | diagnostic | 5/5 | 5/5 | 30.8 | 0.600000 | 1.000 |
| `revision_precommitment_peer_evidence_resource_threshold_local_envw2p0_hidden_observation` | diagnostic | 5/5 | 5/5 | 30.4 | 0.600000 | 1.000 |
| `revision_precommitment_peer_evidence_resource_threshold_local_envw2p0_local_sustain_observation` | main | 5/5 | 5/5 | 30.0 | 0.600000 | 1.000 |

## Mechanism

This result narrows the explanation. The successful local-threshold path is not
mainly exploiting the global resource fraction input. Hiding that scalar does
not remove the recovery. Replacing it with a local sustain observation slightly
improves mean TtC in this five-seed slice.

Seed-1 contrast:

| Observation | Epoch | Action rate | Resource fraction | Env advantage | Effective advantage | Policy prob | High-policy rate | Ready component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| global | 19 | 0.800 | 0.678 | 0.133 | -0.068 | 0.637 | 0.100 | 0.340 |
| global | 24 | 0.980 | 0.790 | 0.001 | -0.085 | 0.568 | 0.020 | 0.910 |
| global | 34 | 1.000 | 1.000 | 0.000 | 0.375 | 0.724 | 0.360 | 1.000 |
| hidden | 19 | 0.800 | 0.685 | 0.122 | -0.074 | 0.643 | 0.090 | 0.450 |
| hidden | 24 | 0.990 | 0.806 | 0.000 | -0.052 | 0.582 | 0.000 | 0.970 |
| hidden | 34 | 1.000 | 1.000 | 0.000 | 0.375 | 0.746 | 0.430 | 1.000 |
| local_sustain | 19 | 0.900 | 0.681 | 0.038 | -0.251 | 0.634 | 0.080 | 0.670 |
| local_sustain | 24 | 1.000 | 0.821 | 0.000 | -0.019 | 0.536 | 0.000 | 0.980 |
| local_sustain | 34 | 1.000 | 1.000 | 0.000 | 0.375 | 0.719 | 0.230 | 1.000 |

The important distinction remains population versus local thresholding.
Population thresholding still fails at `0/5`; local thresholding succeeds across
all three observation modes.

## Decision

Keep `resource_observation_mode` as a Toy4 stress/control:

- `global` preserves the original observation semantics.
- `hidden` tests whether the path depends on direct resource-stock visibility.
- `local_sustain` tests a local-information alternative aligned with the local
  threshold mechanism.

Do not overclaim:

1. Clean reputation imitation is still faster at mean TtC `15.0`.
2. Hidden observation passing means direct global stock observation is not
   necessary for this path, not that observation quality is irrelevant.
3. The structural result is still local-threshold specific; the population
   threshold negative control remains important.

The next useful step is to combine local-sustain observation with the earlier
baseline-fragility setting: noisy reputation plus resource stress. That asks
whether the more local observation gives a real advantage when the clean
hand-coded baseline no longer has perfect reputation ordering.

## Verification

- `uv run ruff check src/neural_abm/config.py src/neural_abm/toy_public_goods.py tests/test_toy4_runner.py tests/test_evidence_gate.py`
- `uv run pytest tests/test_toy4_runner.py tests/test_evidence_gate.py -q`
- `uv run python scripts/run_basin_credit_evidence_workflow.py --manifest experiments/evidence/toy4_resource_threshold_sparse_resource_observation_stress_quick.yaml`
