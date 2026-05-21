# Toy4 Resource Threshold Heterogeneous Extraction Stress Findings

Date: 2026-05-21

## Question

Does the local resource-threshold neural path remain stable when Toy4 resource
depletion is spatially heterogeneous?

This stress makes defector extraction rates checkerboard-heterogeneous while
keeping the mean extraction scale fixed:

- base extraction per defector: `0.05`
- extraction heterogeneity: `1.0`
- high-extraction cells: `0.10`
- low-extraction cells: `0.00`
- initial contribution probability: `0.35`
- resource initial/capacity: `60/100`

The stress is implemented in Toy4 environment dynamics, not in the shared NABM
unit. Local threshold values use the focal group's local extraction mix when
computing sustain-rate pressure.

## Artifacts

- Manifest:
  `experiments/evidence/toy4_resource_threshold_heterogeneous_extraction_stress_quick.yaml`
- Run rows:
  `experiments/results/nabm_effect_matrix/toy4_resource_threshold_heterogeneous_extraction_stress_quick_runs.csv`
- Gate summary:
  `experiments/evidence/results/toy4_resource_threshold_heterogeneous_extraction_stress_quick.summary.json`
- Profile:
  `experiments/results/nabm_effect_matrix/toy4_resource_threshold_heterogeneous_extraction_stress_quick_profile.json`

## Result

The gate passed. The local-threshold neural path kept five-seed ceiling
stability, but clean reputation imitation also remained perfect and faster.

| Variant | Group | Final ceiling hits | Ever ceiling hits | Mean TtC | Mean final payoff | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `reputation_imitation_resource_p0p35_heterogeneous_extraction_h1p0` | baseline | 5/5 | 5/5 | 15.0 | 0.600000 | 1.000 |
| `reputation_imitation_resource_p0p35_noisy_s2p0_heterogeneous_extraction_h1p0` | diagnostic | 3/5 | 3/5 | 47.333 | 0.494039 | 0.600 |
| `revision_precommitment_peer_evidence_resource_threshold_population_envw2p0_heterogeneous_extraction_h1p0` | diagnostic | 0/5 | 0/5 |  | -0.302000 | 0.000 |
| `revision_precommitment_peer_evidence_resource_threshold_local_envw2p0_heterogeneous_extraction_h1p0` | main | 5/5 | 5/5 | 33.0 | 0.600000 | 1.000 |

## Mechanism

This result is positive but weaker than the noisy-reputation stress. Heterogeneous
extraction does not by itself make the clean reputation baseline fragile. Once
reputation imitation moves quickly to all contribution, the checkerboard
resource heterogeneity disappears as a practical obstacle.

The useful signal is still local rather than population-level. In seed 1:

| Variant | Epoch | Action rate | Resource fraction | Env advantage | Effective advantage | Policy/action prob | High-policy rate | Ready component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean reputation | 4 | 1.000 | 0.694 |  |  | 1.000 | 0.000 | 0.000 |
| clean reputation | 14 | 1.000 | 0.994 |  |  | 1.000 | 0.000 | 0.000 |
| population threshold | 19 | 0.500 | 0.383 | 0.333 | -0.318 | 0.504 | 0.000 | 0.000 |
| population threshold | 39 | 0.290 | 0.000 | 0.865 | -0.095 | 0.312 | 0.000 | 0.000 |
| local threshold | 19 | 0.700 | 0.628 | 0.219 | -0.006 | 0.594 | 0.030 | 0.170 |
| local threshold | 24 | 0.850 | 0.720 | 0.041 | -0.158 | 0.547 | 0.020 | 0.720 |
| local threshold | 34 | 1.000 | 1.000 | 0.000 | 0.375 | 0.578 | 0.000 | 1.000 |
| local threshold | 59 | 1.000 | 1.000 | 0.000 | 0.375 | 0.975 | 0.990 | 1.000 |

Population thresholding again fails to form a ready component. Local thresholding
forms the bridge, but it reaches the ceiling later than in the homogeneous
hardening run: mean TtC `33.0` versus `30.8`.

## Decision

Keep heterogeneous extraction support as a Toy4 stress/control. It is useful for
checking whether the local-threshold signal is robust to spatially uneven
resource damage, but it is not yet a baseline-breaking environment by itself.

Do not overclaim:

1. Clean reputation imitation is unchanged at `5/5` and remains faster.
2. Heterogeneous extraction plus noisy reputation reproduces the noisy baseline
   fragility, but does not create a qualitatively new failure mode.
3. The main positive result is robustness of the local-threshold neural path,
   not dominance over the clean hand-coded rule.

The next stronger test should make the resource information itself sparse or
local. That is more likely to distinguish a local threshold mechanism from a
global reputation ranking rule than extraction heterogeneity alone.

## Verification

- `uv run ruff check src/neural_abm/config.py src/neural_abm/toy_public_goods.py tests/test_toy4_runner.py tests/test_evidence_gate.py`
- `uv run pytest tests/test_toy4_runner.py tests/test_evidence_gate.py -q`
- `uv run python scripts/run_basin_credit_evidence_workflow.py --manifest experiments/evidence/toy4_resource_threshold_heterogeneous_extraction_stress_quick.yaml`
