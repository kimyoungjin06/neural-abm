# Toy5 Neural Threshold-Target Frontier Stability Findings

## Scope

This run tests whether the Toy5 neural policy can spread from a single real
seed under the threshold-target local update, while preserving the no-seed
safety separation established by the direction-control stress run.

Artifacts:

- Manifest: `experiments/evidence/toy5_neural_threshold_target_frontier_stability.yaml`
- Gate summary: `experiments/evidence/results/toy5_neural_threshold_target_frontier_stability.summary.md`
- Run rows: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_frontier_stability_runs.csv`
- Effect report: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_frontier_stability_effects.md`

## Gate Result

| Variant | Group | Final ceiling hits | Mean TtC | Final cascade mean | Terminal ceiling rate | Direction ok mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `neural_threshold_target_output_average` | baseline | 0/10 | n/a | 1 | 0.0 | 0.0 |
| `neural_threshold_target_readiness_propagation_w1p0` | diagnostic | 10/10 | 31.4 | 100 | 1.0 | 1.0 |
| `neural_threshold_target_local_direction_w1p0` | diagnostic | 0/10 | n/a | 1 | 0.0 | 0.0 |
| `neural_threshold_target_readiness_augmented_w1p0` | diagnostic | 0/10 | n/a | 1 | 0.0 | 0.0 |
| `neural_threshold_target_exposure_anchored_w2p0` | threshold_frontier | 10/10 | 31.4 | 100 | 1.0 | 1.0 |

The evidence gate passed for the main `threshold_frontier` group:
`neural_threshold_target_exposure_anchored_w2p0` reached final ceiling in
10/10 seeds, with mean time-to-ceiling 31.4 epochs.

## Interpretation

The frontier failure was not solved by threshold-target learning alone. The
baseline stayed at the single initial adopter in every seed. Local-threshold
direction and unanchored readiness-augmented direction also stayed at one
adopter because the real seed was not treated as a valid exposure direction
anchor.

The new exposure-anchored direction source changes that bootstrap condition:
current adopters can act as local exposure/readiness anchors, and readiness
propagation can then move through the frontier. This is still not a claim that
raw readiness propagation is sufficient, because the non-directional diagnostic
also spreads in the seeded frontier case.

The safety distinction comes from the earlier no-seed threshold-target stress
run: `neural_threshold_target_non_directional_readiness_self_excitation` failed
the no-seed safety case with 0/10 final ceiling hits and mean non-adoption 0.0,
while the direction-gated threshold-target candidates preserved 10/10 no-seed
non-adoption. Together, the two stress cases indicate that the useful mechanism
is not generic self-excitation, but direction-gated propagation that can still
recognize a real seed.

## Limits

This is a calibrated frontier candidate, not a final broad claim. It uses
`policy_prior_action_probability=0.49`, 50 epochs, argmax decisions, and
homogeneous threshold 0.75. The paper claim should cite both stress cases:
no-seed safety and seeded frontier spread.

The combined safety/frontier manifest now enforces both stress regimes under one
label, so this frontier-only result should be treated as supporting detail rather
than the main claim artifact.
