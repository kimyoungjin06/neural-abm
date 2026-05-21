# Toy5 Neural Threshold-Target Direction-Control Stress Findings

## Scope

- Manifest: `experiments/evidence/toy5_neural_threshold_target_direction_control_stress.yaml`
- Run rows: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_direction_control_stress_runs.csv`
- Gate summary: `experiments/evidence/results/toy5_neural_threshold_target_direction_control_stress.summary.md`
- Seeds: 1-10
- Epochs: 20

This run keeps the neural policy path active and re-enables local learning with
`model.policy.domain.local_update_rule: threshold_target`. The rule assigns a
positive policy-gradient advantage when the sampled action matches the local
threshold target and a negative advantage when it does not. In no-seed states,
action 0 is therefore reinforced instead of penalized.

## Gate Result

The main direction-control claim passed.

| Variant | Group | Final control hits | Mean TtC | Mean non-adoption rate | Final action rate | Ever-final misses |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `neural_threshold_target_no_seed_control` | baseline | 10/10 | 0.0 | 1.0 | 0.0 | 0 |
| `neural_threshold_target_non_directional_readiness_self_excitation` | diagnostic | 0/10 | 0.0 | 0.0 | 1.0 | 10 |
| `neural_threshold_target_local_direction_control` | diagnostic | 10/10 | 0.0 | 1.0 | 0.0 | 0 |
| `neural_threshold_target_augmented_direction_control_w1p0` | direction_control | 10/10 | 0.0 | 1.0 | 0.0 | 0 |

## Interpretation

The threshold-target local update fixes the no-seed local-learning problem:
the neural prior baseline remains at action rate 0.0 for all ten seeds with
learning enabled.

The non-directional readiness diagnostic still creates a false full adoption
cascade in every seed. All agents become ready and forced at epoch 2, and the
terminal non-adoption rate is 0.0. This confirms that threshold-target local
learning stabilizes the local policy but does not by itself solve the
coordination self-excitation problem.

Both directional controls block the false cascade. Their final
`precommitment_direction_ok_rate` is 0.0, so readiness evidence cannot spread
when no local threshold signal supports adoption.

## Relation To Previous Stress Runs

This strengthens the earlier neural no-seed stress:

- Previous neural stress disabled local learning to isolate the readiness rule.
- This run turns local learning back on with a threshold-target advantage.
- The same control pattern survives: non-directional readiness fails, while
  local and augmented direction gates block self-excitation.

## Next Step

The remaining open question is whether threshold-target local learning can also
support the frontier spread case, where a real seed should propagate. A focused
follow-up should reuse the frontier setup and replace the default
`adoption_utility` local rule with `threshold_target`, then compare local
direction, non-directional readiness, and readiness-augmented direction again.
