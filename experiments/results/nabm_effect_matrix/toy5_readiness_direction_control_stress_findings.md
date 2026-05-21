# Toy5 Readiness Direction-Control Stress Findings

## Scope

- Manifest: `experiments/evidence/toy5_readiness_direction_control_stress.yaml`
- Run rows: `experiments/results/nabm_effect_matrix/toy5_readiness_direction_control_stress_runs.csv`
- Gate summary: `experiments/evidence/results/toy5_readiness_direction_control_stress.summary.md`
- Seeds: 1-10
- Epochs: 20

This stress case is a no-seed control: the initial adoption fraction is 0.0, so
the correct domain behavior is to preserve non-adoption. The primary metric is
`domain_non_adoption_rate`; a wrong self-excited adoption cascade drives it to
0.0.

## Gate Result

The main direction-control claim passed.

| Variant | Group | Final control hits | Mean TtC | Mean non-adoption rate | Final action rate | Ever-final misses |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `complex_threshold_no_seed_control` | baseline | 10/10 | 0.0 | 1.0 | 0.0 | 0 |
| `non_directional_readiness_self_excitation` | diagnostic | 0/10 | 0.0 | 0.0 | 1.0 | 10 |
| `local_threshold_direction_control` | diagnostic | 10/10 | 0.0 | 1.0 | 0.0 | 0 |
| `readiness_augmented_direction_control_w1p0` | direction_control | 10/10 | 0.0 | 1.0 | 0.0 | 0 |

## Interpretation

The non-directional readiness diagnostic creates a full false adoption cascade
in every seed. It starts inside the ceiling at epoch 0 because no agents are
adopted, then leaves that correct state and ends with `domain_non_adoption_rate`
0.0. The gate records this as 10/10 ever-final misses and terminal ceiling rate
0.0.

Both directional variants block the self-excited cascade. In the no-seed state,
the final `precommitment_direction_ok_rate` is 0.0 for both local-threshold and
readiness-augmented direction, so peer readiness does not add adoption evidence
without domain support.

This result should be read together with
`toy5_readiness_augmented_direction_frontier_stability_findings.md`:

- Frontier stability showed that local-threshold direction is too conservative
  when a real seed should spread: local direction reached 3/10, while
  readiness-augmented direction reached 10/10.
- This no-seed stress shows that non-directional readiness is too permissive:
  it reached 0/10 control hits and forced a full false cascade.
- Readiness-augmented direction is therefore the directional-control candidate:
  it preserves the no-seed control behavior while recovering the frontier seeds
  that local-only direction misses.

## Limitations

This is a mechanism-control stress, not a full neural-policy superiority claim.
The stress intentionally uses the deterministic `complex_threshold` policy to
isolate the readiness propagation operator. The baseline also passes, so
`Baseline Improved` is false by design. The evidence value is the contrast
against the non-directional readiness diagnostic, not an improvement over the
correct no-seed baseline.

The next stronger test should keep the same control framing but move back to a
neural policy with an explicit prior or calibration that prevents random
no-seed adoption. That would make the same self-excitation test fully neural
instead of operator-isolated.
