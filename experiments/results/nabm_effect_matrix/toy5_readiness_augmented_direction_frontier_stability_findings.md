# Toy5 Readiness-Augmented Direction Frontier Stability Findings

## Scope

- Manifest: `experiments/evidence/toy5_readiness_augmented_direction_frontier_stability.yaml`
- Run rows: `experiments/results/nabm_effect_matrix/toy5_readiness_augmented_direction_frontier_stability_runs.csv`
- Gate summary: `experiments/evidence/results/toy5_readiness_augmented_direction_frontier_stability.summary.md`
- Seeds: 1-10
- Epochs: 30

This run tests whether readiness-augmented direction evidence fixes the Toy5
frontier failure mode where local threshold direction is too conservative after
the initial seed has already made the correct action discoverable.

## Gate Result

The main readiness-frontier claim passed.

| Variant | Group | Final ceiling hits | Mean time to ceiling | Mean cascade size | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `neural_argmax_output_average` | baseline | 0/10 | n/a | 89.2 | Output averaging alone remains below the 0.95 ceiling threshold. |
| `neural_argmax_precommitment_evidence` | diagnostic | 1/10 | 4.0 | 89.3 | Evidence without propagation does not solve cascade completion. |
| `neural_argmax_readiness_propagation_w1p0` | diagnostic | 10/10 | 5.4 | 100.0 | Strong fastest diagnostic, but it has no directional gate. |
| `neural_argmax_local_threshold_direction_w1p0` | diagnostic | 3/10 | 11.0 | 93.0 | Local threshold direction is too conservative at this frontier. |
| `neural_argmax_readiness_augmented_direction_w1p0` | readiness_frontier | 10/10 | 9.4 | 100.0 | Passes the stability claim with directional gating enabled. |

## Seed-Level Comparison

Readiness-augmented direction rescued 7 seeds that local threshold direction
missed: 2, 3, 4, 6, 7, 8, and 9. There were no regressions from local-pass to
augmented-miss.

| Seed | Local direction final rate | Local TtC | Augmented direction final rate | Augmented TtC |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.95 | 11 | 1.00 | 9 |
| 2 | 0.89 | n/a | 1.00 | 10 |
| 3 | 0.94 | n/a | 1.00 | 8 |
| 4 | 0.94 | n/a | 1.00 | 10 |
| 5 | 0.98 | 7 | 1.00 | 7 |
| 6 | 0.93 | n/a | 1.00 | 12 |
| 7 | 0.92 | n/a | 1.00 | 9 |
| 8 | 0.94 | n/a | 1.00 | 9 |
| 9 | 0.85 | n/a | 1.00 | 12 |
| 10 | 0.96 | 15 | 1.00 | 8 |

## Direction Diagnostics

The local-threshold diagnostic had a mean direction score of 0.1806 and a
direction-ok rate of 0.891. That is enough to occasionally pass but not enough
to sustain broad cascade completion.

The readiness-augmented direction variant had a mean direction score of 1.25,
a direction-ok rate of 1.0, and a positive direction-score rate of 1.0. All
ten seeds reached all-ready states with observed all-ready epochs from 9 to 16.

## Interpretation

The stability evidence supports the frontier mechanism: adding readiness-aware
direction evidence closes the gap created by a purely local direction predicate.
This is not just a random seed artifact under the 10-seed check; the augmented
variant improved final ceiling hits from 3/10 to 10/10 and mean cascade size
from 93.0 to 100.0.

The result does not show that readiness-augmented direction dominates the
non-directional readiness propagation diagnostic. The non-directional diagnostic
also reached 10/10 and did so faster, with mean TtC 5.4 versus 9.4. The
stronger claim is therefore not "augmented direction is the fastest Toy5
mechanism." The defensible claim is that it preserves an explicit directional
gate while recovering the local-threshold failure seeds.

## Next Decision

Keep readiness-augmented direction as the directional-control candidate, not as
the unconstrained throughput winner. The next evidence step should compare the
directional candidate against non-directional readiness propagation on cases
where incorrect or premature propagation is possible. That is the setting where
directional control can justify the added complexity.
