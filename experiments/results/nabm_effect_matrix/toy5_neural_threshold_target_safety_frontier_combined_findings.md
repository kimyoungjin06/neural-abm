# Toy5 Neural Threshold-Target Safety Frontier Combined Findings

## Scope

This run binds two Toy5 stress regimes into one gate:

- no-seed safety: no real adopter is present, so readiness propagation must not
  self-excite into adoption;
- seeded frontier spread: one real adopter is present in a high-threshold
  frontier, so the neural policy must still propagate adoption.

Artifacts:

- Manifest: `experiments/evidence/toy5_neural_threshold_target_safety_frontier_combined.yaml`
- Gate summary: `experiments/evidence/results/toy5_neural_threshold_target_safety_frontier_combined.summary.md`
- Run rows: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_safety_frontier_combined_runs.csv`
- Effect report: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_safety_frontier_combined_effects.md`

## Gate Result

| Case | Main variant | Final ceiling hits | Mean TtC | Metric mean | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `toy5_threshold_target_no_seed_safety` | `neural_threshold_target_no_seed_exposure_anchored_prior0p49` | 10/10 | 0.0 | 1.0 non-adoption | 1.0 |
| `toy5_threshold_target_seeded_frontier_spread` | `neural_threshold_target_frontier_exposure_anchored_w2p0` | 10/10 | 31.4 | 100 cascade size | 1.0 |

Overall gate status: pass.

## Interpretation

The useful distinction is exposure-gated direction, not generic readiness
amplification. The non-directional readiness diagnostic still fails no-seed
safety: it reaches 0/10 safety hits, forces all agents by epoch 2, and ends with
mean non-adoption 0.0. That is the self-excitation failure mode.

The strict threshold direction source was too conservative for the seeded
frontier. It required agents to already satisfy the adoption threshold before
they could use readiness propagation, so a single real seed could not start the
frontier wave.

The exposure-anchored direction source separates these two cases. With no seed,
direction scores remain negative and no precommitment forcing occurs. With a
real seed, current adoption becomes a local exposure/readiness anchor, and the
frontier spreads through peer evidence without opening the no-seed path.

## Claim Boundary

This supports a narrow Toy5 claim: threshold-target neural learning plus
exposure-anchored readiness propagation can satisfy no-seed safety and seeded
frontier spread in the calibrated stress setting.

It is not yet a broad robustness claim. The candidate still uses
`policy_prior_action_probability=0.49`, argmax decisions, homogeneous no-seed
threshold 0.25, homogeneous frontier threshold 0.75, and
`precommitment_readiness_direction_weight=2.0`.

## Next Work

The next useful test is not another one-point tuning pass. It should vary the
structural conditions that matter to the claim: graph topology, threshold
heterogeneity, and number or placement of initial seeds.
