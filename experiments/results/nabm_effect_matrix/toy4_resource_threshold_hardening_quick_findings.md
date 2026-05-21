# Toy4 Resource Threshold Hardening Findings

Date: 2026-05-21

## Question

Does the passing Toy4 local resource-threshold candidate remain stable beyond
the initial 3-seed probe, and is the local threshold scope doing real work
relative to a population-level threshold?

This hardening run keeps the revision-operator, objective+basin,
precommitment, and peer-evidence path fixed. It expands the passing candidate
to five seeds and compares three roles:

- baseline: `reputation_imitation_resource_p0p35`
- diagnostic: `threshold_scope=population`, `environment_weight=2.0`
- hardening candidate: `threshold_scope=local`, `environment_weight=2.0`

## Artifacts

- Manifest:
  `experiments/evidence/toy4_resource_threshold_hardening_quick.yaml`
- Run rows:
  `experiments/results/nabm_effect_matrix/toy4_resource_threshold_hardening_quick_runs.csv`
- Gate summary:
  `experiments/evidence/results/toy4_resource_threshold_hardening_quick.summary.json`
- Profile:
  `experiments/results/nabm_effect_matrix/toy4_resource_threshold_hardening_quick_profile.json`

## Result

The gate passed. The local-threshold candidate reached and held the ceiling in
all five seeds. The population-threshold diagnostic reached the ceiling in no
seeds.

| Variant | Group | Final ceiling hits | Ever ceiling hits | Mean TtC | Mean final payoff | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `reputation_imitation_resource_p0p35` | baseline | 5/5 | 5/5 | 15.0 | 0.600000 | 1.000 |
| `revision_precommitment_peer_evidence_resource_threshold_population_envw2p0` | diagnostic | 0/5 | 0/5 |  | -0.298000 | 0.000 |
| `revision_precommitment_peer_evidence_resource_threshold_local_envw2p0` | resource_threshold_hardening | 5/5 | 5/5 | 30.8 | 0.600000 | 1.000 |

The baseline remains faster on this toy, but the local neural path now matches
its final ceiling stability in the resource-stress condition.

## Mechanism

The local threshold signal changes the failure shape. Population thresholding
does not create a stable local propagation front: resource stock declines,
effective advantage oscillates around low-confidence decisions, and
precommitment never forms. Local thresholding gives nearby agents a coordinated
recovery signal before the global population has crossed the sustain rate.

Seed-1 contrast:

| Scope | Epoch | Action rate | Resource fraction | Threshold env advantage | Effective advantage | Post-social policy prob | High-policy rate | Ready component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| population | 1 | 0.490 | 0.589 | 0.358 | 0.188 | 0.520 | 0.000 | 0.000 |
| population | 9 | 0.560 | 0.532 | 0.181 | -0.292 | 0.527 | 0.000 | 0.000 |
| population | 19 | 0.520 | 0.412 | 0.282 | -0.355 | 0.506 | 0.000 | 0.000 |
| population | 29 | 0.520 | 0.265 | 0.282 | -0.677 | 0.432 | 0.000 | 0.000 |
| population | 44 | 0.340 | 0.000 | 0.738 | -0.348 | 0.283 | 0.000 | 0.000 |
| population | 59 | 0.310 | 0.000 | 0.814 | -0.196 | 0.227 | 0.000 | 0.000 |
| local | 1 | 0.490 | 0.589 | 0.603 | 0.677 | 0.517 | 0.000 | 0.000 |
| local | 9 | 0.720 | 0.570 | 0.221 | -0.129 | 0.702 | 0.330 | 0.030 |
| local | 19 | 0.800 | 0.678 | 0.133 | -0.068 | 0.637 | 0.100 | 0.340 |
| local | 29 | 1.000 | 0.940 | 0.000 | 0.244 | 0.570 | 0.010 | 1.000 |
| local | 44 | 1.000 | 1.000 | 0.000 | 0.375 | 0.947 | 0.990 | 1.000 |
| local | 59 | 1.000 | 1.000 | 0.000 | 0.375 | 0.989 | 1.000 | 1.000 |

This supports the interpretation from the 3-seed probe: the useful signal is
not generic resource pressure. It is a local sustain-threshold continuation
signal that creates a neighborhood-scale bridge from partial contribution to
full contribution.

## Decision

Keep the Toy4 resource-threshold hook as the current resource-coupled structural
candidate:

- `resource_environment_threshold_weight`
- `resource_environment_threshold_scope`

The non-resource regression should remain part of the contract: threshold
weighting must be inert when resource dynamics are disabled.

Do not overclaim:

1. Reputation imitation is still faster in the current Toy4 environment.
2. The passing neural path depends on local scope and `environment_weight=2.0`.
3. This shows a plausible structural hook for resource-coupled fragility, not a
   general proof that the neural path dominates the hand-coded baseline.

## Next Step

The next comparison should test whether the local neural path becomes more
valuable when reputation imitation is made fragile: noisier reputation,
heterogeneous sustain thresholds, or sparse/local resource observations. That
is the right place to distinguish a learned local threshold mechanism from a
baseline that is directly matched to the current toy.

## Verification

- `uv run ruff check tests/test_toy4_runner.py tests/test_evidence_gate.py`
- `uv run pytest tests/test_toy4_runner.py tests/test_evidence_gate.py -q`
- `uv run python scripts/run_basin_credit_evidence_workflow.py --manifest experiments/evidence/toy4_resource_threshold_hardening_quick.yaml`
