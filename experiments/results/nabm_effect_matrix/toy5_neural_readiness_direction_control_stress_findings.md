# Toy5 Neural Readiness Direction-Control Stress Findings

## Scope

- Manifest: `experiments/evidence/toy5_neural_readiness_direction_control_stress.yaml`
- Run rows: `experiments/results/nabm_effect_matrix/toy5_neural_readiness_direction_control_stress_runs.csv`
- Gate summary: `experiments/evidence/results/toy5_neural_readiness_direction_control_stress.summary.md`
- Seeds: 1-10
- Epochs: 20

This is the neural version of the no-seed control stress. The initial adoption
fraction is 0.0, and the neural policy is initialized with an explicit action-1
prior of 0.0. Local learning is disabled so the run isolates whether the
readiness propagation mechanism can create a false adoption cascade from a
stable neural no-adoption policy.

## Gate Result

The main direction-control claim passed.

| Variant | Group | Final control hits | Mean TtC | Mean non-adoption rate | Final action rate | Ever-final misses |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `neural_prior_no_seed_control` | baseline | 10/10 | 0.0 | 1.0 | 0.0 | 0 |
| `neural_non_directional_readiness_self_excitation` | diagnostic | 0/10 | 0.0 | 0.0 | 1.0 | 10 |
| `neural_local_threshold_direction_control` | diagnostic | 10/10 | 0.0 | 1.0 | 0.0 | 0 |
| `neural_readiness_augmented_direction_control_w1p0` | direction_control | 10/10 | 0.0 | 1.0 | 0.0 | 0 |

## Interpretation

The neural prior baseline preserves no-adoption in every seed. That removes the
earlier random-initialization confound where a no-seed neural policy could adopt
without any domain signal.

The non-directional readiness diagnostic still creates a false full adoption
cascade in every seed. Its agents all become ready and forced at epoch 2, and
the final action rate is 1.0 for all ten seeds. This confirms that the failure
comes from the readiness propagation rule, not from random neural initialization.

Both directional controls block the self-excitation. Their final direction-ok
rate is 0.0 in the no-seed state, so readiness cannot propagate into adoption
evidence when the local domain state provides no adoption support.

## Relation To Frontier Evidence

Together with the frontier stability run, this supports the directional-control
framing:

- Non-directional readiness propagation is fast but over-permissive under
  no-seed control.
- Local-threshold direction is safe under no-seed control but too conservative
  in frontier spread.
- Readiness-augmented direction is the candidate that preserves the no-seed
  safety check while recovering the frontier spread case.

## Limitations

This run is neural in the policy-output path, but local policy learning is
disabled to isolate the coordination mechanism. It is therefore not yet a full
end-to-end learned-control claim. The next stronger version should keep the
action-1 prior but re-enable a calibrated local learning signal that does not
turn no-adoption into negative reinforcement for action 0.
