# Toy4 Resource Lookahead Probe Findings

Date: 2026-05-21

## Question

Does an action-conditioned one-step resource lookahead improve the Toy4
resource-coupled failure mode better than the previous scalar resource pressure?

This probe preserves the existing revision-operator, objective+basin,
precommitment, and peer-evidence path. It only changes the Toy4 environment
component:

- scalar pressure diagnostic: `pressure_weight=1.0`, `lookahead_weight=0.0`
- lookahead candidates: `pressure_weight=0.0`, `lookahead_weight=1.0`
- objective sweep: `environment_weight in {2.0, 5.0, 10.0}`

## Artifacts

- Manifest:
  `experiments/evidence/toy4_resource_lookahead_probe_quick.yaml`
- Run rows:
  `experiments/results/nabm_effect_matrix/toy4_resource_lookahead_probe_quick_runs.csv`
- Gate summary:
  `experiments/evidence/results/toy4_resource_lookahead_probe_quick.summary.json`
- Profile:
  `experiments/results/nabm_effect_matrix/toy4_resource_lookahead_probe_quick_profile.json`

## Result

The gate failed. Lookahead-only candidates did not reach the ceiling in any
seed.

| Variant | Group | Final ceiling hits | Ever ceiling hits | Mean TtC | Mean final payoff | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `reputation_imitation_resource_p0p35` | baseline | 3/3 | 3/3 | 15.0 | 0.600000 | 1.000 |
| `revision_precommitment_peer_evidence_resource_pressure_envw2p0` | diagnostic | 0/3 | 0/3 |  | -0.319948 | 0.000 |
| `revision_precommitment_peer_evidence_resource_lookahead_envw2p0` | main | 0/3 | 0/3 |  | -0.003333 | 0.000 |
| `revision_precommitment_peer_evidence_resource_lookahead_envw5p0` | main | 0/3 | 0/3 |  | -0.003333 | 0.000 |
| `revision_precommitment_peer_evidence_resource_lookahead_envw10p0` | main | 0/3 | 0/3 |  | -0.003333 | 0.000 |

## Interpretation

The lookahead hook is structurally cleaner than the scalar pressure because it
is computed from a focal action-1 versus action-0 counterfactual. However, the
one-step stock delta is too local and too late to create the cooperative
cascade.

Seed-1 trajectory:

| Variant | Epoch | Action rate | Resource fraction | Env advantage | Effective advantage | Post-social policy prob |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lookahead_envw2p0` | 2 | 0.43 | 0.574 | 0.050 | -0.463 | 0.445 |
| `lookahead_envw2p0` | 10 | 0.20 | 0.403 | 0.050 | -0.839 | 0.183 |
| `lookahead_envw2p0` | 20 | 0.02 | 0.000 | 0.000 | -1.825 | 0.030 |
| `lookahead_envw10p0` | 2 | 0.43 | 0.574 | 0.050 | -0.063 | 0.466 |
| `lookahead_envw10p0` | 10 | 0.29 | 0.432 | 0.050 | -0.375 | 0.213 |
| `lookahead_envw10p0` | 20 | 0.01 | 0.008 | 0.000 | -1.807 | 0.029 |

The core issue is clipping. Once resource depletion is severe, changing one
agent from action 0 to action 1 no longer changes the next resource stock:
both counterfactuals remain clipped to zero. The lookahead signal then vanishes
exactly when recovery would need a coordinated multi-agent shift.

The scalar pressure does the opposite: it remains large after depletion, but it
is not action-path specific and can push partial contribution into a low-resource
payoff trap. The one-step lookahead avoids that trap but is too weak to escape
collapse.

## Decision

Keep the action-conditioned lookahead hook as a domain-specific Toy4 component,
but treat this evidence as negative for one-step individual lookahead as the
main solution.

The next structural candidate should represent coordinated threshold recovery,
not another scalar weight increase:

1. Add a Toy4 resource-threshold continuation signal that evaluates whether a
   local or population action shift can cross the sustain rate.
2. Make the signal explicitly multi-agent or neighborhood-scoped, because the
   single-agent one-step counterfactual is clipped to zero after collapse.
3. Keep `resource_environment_pressure_weight` and
   `resource_environment_lookahead_weight` as controls for future ablations.

## Verification

- `uv run ruff check src/neural_abm/config.py src/neural_abm/toy_public_goods.py tests/test_toy4_runner.py`
- `uv run pytest tests/test_toy4_runner.py -q`
- `uv run ruff check src tests scripts`
- `uv run pytest tests/test_toy4_runner.py tests/test_evidence_gate.py -q`
- `uv run python scripts/run_basin_credit_evidence_workflow.py --manifest experiments/evidence/toy4_resource_lookahead_probe_quick.yaml`
