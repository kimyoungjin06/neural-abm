# Toy4 Resource Environment Weight Probe Findings

Date: 2026-05-21

## Question

Can the new Toy4 resource-maintenance pressure become a useful structural signal
when it is routed into the state-continuation objective via
`environment_weight`?

This probe intentionally keeps the existing revision-operator,
objective+basin, precommitment, and peer-evidence settings fixed, then sweeps
only `environment_weight in {0.0, 0.5, 1.0, 2.0}` on the resource-coupled Toy4
case.

## Artifacts

- Manifest:
  `experiments/evidence/toy4_resource_environment_weight_probe_quick.yaml`
- Run rows:
  `experiments/results/nabm_effect_matrix/toy4_resource_environment_weight_probe_quick_runs.csv`
- Gate summary:
  `experiments/evidence/results/toy4_resource_environment_weight_probe_quick.summary.json`
- Profile:
  `experiments/results/nabm_effect_matrix/toy4_resource_environment_weight_probe_quick_profile.json`

## Result

The gate failed. The environment component is wired through the objective, but
the current scalar pressure is not sufficient to recover Toy4 ceiling behavior.

| Variant | Group | Final ceiling hits | Ever ceiling hits | Mean TtC | Mean final payoff | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `reputation_imitation_resource_p0p35` | baseline | 3/3 | 3/3 | 15.0 | 0.600000 | 1.000 |
| `revision_precommitment_peer_evidence_resource_envw0p0` | diagnostic | 0/3 | 0/3 |  | -0.003333 | 0.000 |
| `revision_precommitment_peer_evidence_resource_envw0p5` | main | 0/3 | 0/3 |  | -0.003333 | 0.000 |
| `revision_precommitment_peer_evidence_resource_envw1p0` | main | 0/3 | 0/3 |  | -0.140000 | 0.000 |
| `revision_precommitment_peer_evidence_resource_envw2p0` | main | 0/3 | 0/3 |  | -0.319948 | 0.000 |

## Interpretation

The hook is active, but it does not create a cooperative cascade.

- At `envw0p5`, the positive resource pressure weakens the negative objective,
  but the effective advantage remains negative after the resource begins to
  erode. The trajectory is nearly identical to `envw0p0`.
- At `envw1p0`, late effective advantage can become slightly positive in some
  rows, but policy probability remains too low for precommitment to activate.
- At `envw2p0`, action rates rise into the 0.5-0.6 range, but the resource still
  declines. Once the resource is depleted or low, partial contribution becomes
  costly and final payoff worsens.

Seed-1 trajectory illustrates the failure mode:

| Variant | Epoch | Action rate | Resource fraction | Env advantage | Effective advantage | Post-social policy prob |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `envw0p0` | 10 | 0.20 | 0.403 | 1.036 | -0.939 | 0.183 |
| `envw0p0` | 60 | 0.01 | 0.000 | 1.984 | -1.825 | 0.002 |
| `envw0p5` | 10 | 0.20 | 0.405 | 1.032 | -0.418 | 0.192 |
| `envw0p5` | 60 | 0.01 | 0.000 | 1.984 | -0.833 | 0.002 |
| `envw1p0` | 10 | 0.27 | 0.421 | 0.894 | -0.004 | 0.246 |
| `envw1p0` | 60 | 0.09 | 0.000 | 1.856 | 0.031 | 0.091 |
| `envw2p0` | 10 | 0.57 | 0.528 | 0.243 | -0.177 | 0.525 |
| `envw2p0` | 60 | 0.64 | 0.285 | 0.544 | -0.110 | 0.589 |

The key structural issue is that this first environment component is a global
pressure signal. It tells the population that the resource is under-maintained,
but it does not produce a sharp action-conditioned path to the basin. It is also
reactive: pressure grows after action rate/resource fraction are already below
their maintenance thresholds. That makes it a useful diagnostic hook, not yet a
successful mechanism.

## Decision

Keep the resource environment component and the manifest as negative evidence.
Do not escalate this to a larger weight sweep yet. The failure says the next
Toy4 improvement should be more structural than scalar objective weighting:

1. Add an action-conditioned resource lookahead signal that estimates the
   one-step resource/payoff consequence of choosing action 1 versus action 0.
2. Keep the environment component explicit in Toy4 rather than hiding it inside
   a generic NABM unit.
3. Use this probe as the control for any future resource-aware critic or
   threshold-aware policy path.

## Verification

- `uv run ruff check tests/test_evidence_gate.py`
- `uv run pytest tests/test_evidence_gate.py -q`
- `uv run python scripts/run_basin_credit_evidence_workflow.py --manifest experiments/evidence/toy4_resource_environment_weight_probe_quick.yaml`
- `uv run python -m neural_abm.diagnostics.profile_index --manifest ... --index-label evidence_profile_index_calibration`
