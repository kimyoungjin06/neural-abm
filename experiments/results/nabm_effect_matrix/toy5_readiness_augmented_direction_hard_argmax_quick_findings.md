# Toy5 Readiness-Augmented Direction Hard-Argmax Findings

Manifest:
`experiments/evidence/toy5_readiness_augmented_direction_hard_argmax_quick.yaml`

Purpose:

- Re-check the Toy5 hard-argmax holdout after adding a Toy5-local
  precommitment direction source.
- Separate three mechanisms that were previously conflated:
  non-directional readiness propagation, local threshold direction, and
  readiness-augmented threshold direction.
- Keep the shared readiness unit domain-neutral: Toy5 computes threshold
  direction scores, and the shared binary runner only consumes the resulting
  score.

Run artifacts:

- `experiments/results/nabm_effect_matrix/toy5_readiness_augmented_direction_hard_argmax_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy5_readiness_augmented_direction_hard_argmax_quick_effects.md`
- `experiments/evidence/results/toy5_readiness_augmented_direction_hard_argmax_quick.summary.md`

Gate result: **pass**. The main readiness-augmented direction candidate reaches
the 0.95 action-rate ceiling in all three seeds with mean time-to-ceiling 5.67.

| Variant | Group | Final hits | Mean TtC | Final cascade size | Direction ok | Direction score | Direction positive | All-ready epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neural_argmax_output_average` | baseline | 0/3 |  | 74.67 | 0.0000 | 0.0000 | 0.0000 |  |
| `neural_argmax_precommitment_evidence` | diagnostic | 0/3 |  | 75.00 | 1.0000 | 0.0000 | 0.0000 |  |
| `neural_argmax_readiness_propagation_w1p0` | diagnostic | 3/3 | 4.33 | 100.00 | 1.0000 | 0.0000 | 0.0000 | 5.00 |
| `neural_argmax_local_threshold_direction_w1p0` | diagnostic | 3/3 | 5.67 | 100.00 | 1.0000 | 0.7500 | 1.0000 | 6.67 |
| `neural_argmax_readiness_augmented_direction_w1p0` | readiness_direction | 3/3 | 5.67 | 100.00 | 1.0000 | 1.7500 | 1.0000 | 6.67 |

Interpretation:

- The earlier direction-gated failure was partly a missing-signal problem. Toy5
  did not provide a domain-local precommitment direction score, so a
  `requires_direction=true` variant could be blocked even when threshold
  structure made direction meaningful.
- Adding `local_threshold` direction is already sufficient in this hard-argmax
  regime. It reaches the 0.95 ceiling in all seeds.
- Readiness-augmented direction also passes, but it does not outperform the
  simpler local-threshold direction here. It raises the final direction score
  mean from 0.75 to 1.75 without changing endpoint hits or time-to-ceiling.
- Non-directional readiness remains faster in this regime, so the direction
  constraint is interpretable but not free. It trades speed for a stronger
  domain-grounded justification.

Conclusion:

- This is a real structural improvement over the previous direction-gated
  diagnostic: Toy5 now has an explicit precommitment direction hook, and
  direction-aware readiness can recover the hard-argmax failure.
- The conservative claim is not yet "readiness augmentation is necessary." The
  current evidence says "domain-local threshold direction is sufficient, and
  readiness-augmented direction is compatible but not uniquely beneficial."
- The next hard holdout should target a frontier regime where local threshold
  direction remains sparse, so readiness-augmented direction can be tested
  against a simpler local-direction control.
