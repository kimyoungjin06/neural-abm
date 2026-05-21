# Toy4 Resource Threshold Probe Findings

Date: 2026-05-21

## Question

Can Toy4 recover from the resource-coupled failure when the environment signal
represents coordinated sustain-threshold recovery instead of scalar pressure or
single-agent one-step lookahead?

This probe preserves the existing revision-operator, objective+basin,
precommitment, and peer-evidence path. It compares:

- scalar pressure diagnostic: `pressure_weight=1.0`, `environment_weight=2.0`
- one-step lookahead diagnostic: `lookahead_weight=1.0`, `environment_weight=10.0`
- local threshold candidates: `threshold_weight=1.0`,
  `threshold_scope=local`, `environment_weight in {0.5, 1.0, 2.0}`

## Artifacts

- Manifest:
  `experiments/evidence/toy4_resource_threshold_probe_quick.yaml`
- Run rows:
  `experiments/results/nabm_effect_matrix/toy4_resource_threshold_probe_quick_runs.csv`
- Gate summary:
  `experiments/evidence/results/toy4_resource_threshold_probe_quick.summary.json`
- Profile:
  `experiments/results/nabm_effect_matrix/toy4_resource_threshold_probe_quick_profile.json`

## Result

The gate passed. Only the local-threshold candidate at
`environment_weight=2.0` reached the ceiling.

| Variant | Group | Final ceiling hits | Ever ceiling hits | Mean TtC | Mean final payoff | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `reputation_imitation_resource_p0p35` | baseline | 3/3 | 3/3 | 15.0 | 0.600000 | 1.000 |
| `revision_precommitment_peer_evidence_resource_pressure_envw2p0` | diagnostic | 0/3 | 0/3 |  | -0.319948 | 0.000 |
| `revision_precommitment_peer_evidence_resource_lookahead_envw10p0` | diagnostic | 0/3 | 0/3 |  | -0.003333 | 0.000 |
| `revision_precommitment_peer_evidence_resource_threshold_local_envw0p5` | main | 0/3 | 0/3 |  | -0.003333 | 0.000 |
| `revision_precommitment_peer_evidence_resource_threshold_local_envw1p0` | main | 0/3 | 0/3 |  | -0.016667 | 0.000 |
| `revision_precommitment_peer_evidence_resource_threshold_local_envw2p0` | main | 3/3 | 3/3 | 30.667 | 0.600000 | 1.000 |

The best threshold candidate ties the reputation baseline on final ceiling and
terminal stability, but remains slower: mean TtC 30.667 versus baseline 15.0.

## Mechanism

The result supports the structural diagnosis from the previous two probes:

- Scalar pressure remains active after resource decline, but is not tied to a
  coordinated action path and can stabilize a low-resource partial-contribution
  trap.
- One-step individual lookahead is action-conditioned, but vanishes when both
  counterfactual next-resource states are clipped to zero.
- Local threshold continuation stays active after collapse risk and values each
  agent's role in moving its neighborhood toward the sustain contribution rate.

Seed-1 trajectory shows the successful transition:

| Epoch | Action rate | Resource fraction | Threshold env advantage | Effective advantage | Post-social policy prob | High-policy rate | Ready component |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.44 | 0.574 | 0.705 | 0.848 | 0.546 | 0.01 | 0.00 |
| 5 | 0.54 | 0.558 | 0.508 | 0.419 | 0.617 | 0.04 | 0.00 |
| 10 | 0.72 | 0.570 | 0.221 | -0.129 | 0.702 | 0.33 | 0.03 |
| 20 | 0.80 | 0.678 | 0.133 | -0.068 | 0.637 | 0.10 | 0.34 |
| 25 | 0.98 | 0.790 | 0.001 | -0.085 | 0.568 | 0.02 | 0.91 |
| 30 | 1.00 | 0.940 | 0.000 | 0.244 | 0.570 | 0.01 | 1.00 |
| 35 | 1.00 | 1.000 | 0.000 | 0.375 | 0.724 | 0.36 | 1.00 |
| 45 | 1.00 | 1.000 | 0.000 | 0.375 | 0.947 | 0.99 | 1.00 |
| 60 | 1.00 | 1.000 | 0.000 | 0.375 | 0.989 | 1.00 | 1.00 |

The threshold signal is strongest before the sustain rate is crossed, then
naturally shuts off near full contribution. That is the desired shape: it
pushes coordinated recovery without remaining as a permanent resource-pressure
bias after the basin is reached.

## Decision

Keep `resource_environment_threshold_weight` and
`resource_environment_threshold_scope` as Toy4 domain-specific structural hooks.
This is the first resource-coupled Toy4 neural path in this sequence that
recovers final ceiling behavior without teacher bootstrap replay.

Do not overclaim yet:

1. The result is still slower than reputation imitation on this toy.
2. The pass depends on `environment_weight=2.0`; lower threshold gains fail.
3. The grouped main average is still poor because the main group intentionally
   includes failed 0.5 and 1.0 threshold probes.

Next validation should harden only the passing candidate:

1. Rerun `threshold_local_envw2p0` with more seeds.
2. Add a non-resource Toy4 regression/control to confirm the threshold hook is
   inert unless explicitly enabled.
3. Compare `threshold_scope=population` versus `local` to separate global
   sustain-rate recovery from neighborhood threshold propagation.

## Verification

- `uv run ruff check src/neural_abm/config.py src/neural_abm/toy_public_goods.py tests/test_toy4_runner.py`
- `uv run pytest tests/test_toy4_runner.py -q`
- `uv run ruff check src/neural_abm/config.py src/neural_abm/toy_public_goods.py tests/test_toy4_runner.py tests/test_evidence_gate.py`
- `uv run pytest tests/test_toy4_runner.py tests/test_evidence_gate.py -q`
- `uv run python scripts/run_basin_credit_evidence_workflow.py --manifest experiments/evidence/toy4_resource_threshold_probe_quick.yaml`
