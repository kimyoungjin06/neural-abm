# Toy5 Readiness-Augmented Direction Frontier Findings

Manifest:
`experiments/evidence/toy5_readiness_augmented_direction_frontier_quick.yaml`

Purpose:

- Find a Toy5 frontier regime where local threshold direction is not enough,
  but readiness-augmented direction closes the cascade.
- Keep the comparison conservative by retaining the non-directional readiness
  diagnostic and the local-threshold direction control.
- Use the same hard-argmax setup as the prior Toy5 direction runs, but raise
  the homogeneous threshold to 0.75 and reduce precommitment evidence decay to
  0.5.

Run artifacts:

- `experiments/results/nabm_effect_matrix/toy5_readiness_augmented_direction_frontier_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy5_readiness_augmented_direction_frontier_quick_effects.md`
- `experiments/evidence/results/toy5_readiness_augmented_direction_frontier_quick.summary.md`

Gate result: **pass**. The main readiness-augmented direction candidate reaches
the 0.95 action-rate ceiling in all three seeds with mean time-to-ceiling 9.00.

| Variant | Group | Final hits | Final action rates | Mean cascade size | Mean TtC | Direction ok | Direction score | All-ready epoch |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `neural_argmax_output_average` | baseline | 0/3 | 0.89, 0.85, 0.91 | 88.33 |  | 0.0000 | 0.0000 |  |
| `neural_argmax_precommitment_evidence` | diagnostic | 0/3 | 0.89, 0.86, 0.92 | 89.00 |  | 1.0000 | 0.0000 |  |
| `neural_argmax_readiness_propagation_w1p0` | diagnostic | 3/3 | 1.00, 1.00, 1.00 | 100.00 | 5.67 | 1.0000 | 0.0000 | 6.67 |
| `neural_argmax_local_threshold_direction_w1p0` | diagnostic | 1/3 | 0.95, 0.89, 0.94 | 92.67 | 11.00 | 0.8900 | 0.1749 |  |
| `neural_argmax_readiness_augmented_direction_w1p0` | readiness_frontier | 3/3 | 1.00, 1.00, 1.00 | 100.00 | 9.00 | 1.0000 | 1.2500 | 12.00 |

Interpretation:

- This is the first Toy5 setting in this sequence where the local direction
  control fails while readiness-augmented direction passes.
- The endpoint gap against local threshold direction is meaningful but not
  huge: mean cascade size improves from 92.67 to 100.00, and final ceiling hits
  improve from 1/3 to 3/3.
- The augmented direction path works by turning ready-neighbor mass into
  threshold-relevant direction. Its mean direction score is 1.25, compared with
  0.1749 for local threshold direction.
- Non-directional readiness remains faster, with mean time-to-ceiling 5.67.
  The augmented direction mechanism therefore buys interpretability and a
  stricter domain-grounded gate, not speed.
- Plain precommitment still does not explain the recovery; it remains below the
  0.95 ceiling in all three seeds.

Conclusion:

- The frontier regime supports a narrower but real mechanism claim:
  readiness-augmented direction can recover a partial-cascade failure that
  local threshold direction does not recover.
- It does not support a stronger claim that direction-aware readiness dominates
  non-directional readiness. The non-directional path is still the faster
  diagnostic upper bound in this regime.
- This is enough to keep the mechanism as a research candidate, but the claim
  should be framed as an interpretable, domain-grounded readiness gate rather
  than a pure performance improvement.
