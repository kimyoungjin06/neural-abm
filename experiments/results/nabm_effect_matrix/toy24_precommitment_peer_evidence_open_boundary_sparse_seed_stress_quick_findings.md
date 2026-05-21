# Toy2/Toy4 Open-Boundary Sparse-Seed Stress Findings

Manifest:
`experiments/evidence/toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick.yaml`

Purpose:

- Extend the sparse-seed stress with open/non-periodic boundaries.
- Use `domain.environment.initial_action_probability: 0.1` for both toys.
- Use `domain.environment.periodic: false` for Toy2 and
  `domain.graph.periodic: false` for Toy4.
- Check whether topology edge effects plus sparse initial action seeds make the
  reputation baseline fragile, while keeping the readiness-propagation candidate
  stable.
- Treat this as an exploratory topology stress, not as a final
  baseline-superiority test.

Run artifacts:

- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick_effects.md`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick_profile.md`
- `experiments/evidence/results/toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick.summary.md`

Gate result: **pass** overall. The main claim group is
`peer_evidence_open_boundary_sparse_seed_stress`, and the best main variant in
both Toy2 and Toy4 is
`revision_precommitment_peer_evidence_open_sparse_p0p1`.

| Case | Variant | Role | Final hits | Ever hits | Ever-final misses | Mean TtC | Terminal ceiling rate | Late flip rate | Metric mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | `reputation_imitation_open_sparse_p0p1` | baseline | 5/5 | 5/5 | 0 | 9.2 | 1.00 | 0.000048 | 3.0000 |
| Toy2 | `objective_basin_open_sparse_p0p1` | diagnostic | 4/5 | 5/5 | 1 | 19.0 | 0.84 | 0.011186 | 2.9985 |
| Toy2 | `revision_objective_basin_open_sparse_p0p1` | diagnostic | 4/5 | 5/5 | 1 | 20.2 | 0.76 | 0.009850 | 2.9975 |
| Toy2 | `revision_precommitment_evidence_open_sparse_p0p1` | diagnostic | 5/5 | 5/5 | 0 | 10.8 | 1.00 | 0.000103 | 3.0000 |
| Toy2 | `revision_precommitment_peer_evidence_open_sparse_p0p1` | main | 5/5 | 5/5 | 0 | 9.4 | 1.00 | 0.000000 | 3.0000 |
| Toy4 | `reputation_imitation_open_sparse_p0p1` | baseline | 5/5 | 5/5 | 0 | 8.8 | 1.00 | 0.000144 | 0.6000 |
| Toy4 | `objective_basin_open_sparse_p0p1` | diagnostic | 4/5 | 5/5 | 1 | 18.2 | 0.80 | 0.007786 | 0.5986 |
| Toy4 | `revision_objective_basin_open_sparse_p0p1` | diagnostic | 4/5 | 5/5 | 1 | 19.0 | 0.80 | 0.007267 | 0.5986 |
| Toy4 | `revision_precommitment_evidence_open_sparse_p0p1` | diagnostic | 5/5 | 5/5 | 0 | 9.8 | 1.00 | 0.000540 | 0.6000 |
| Toy4 | `revision_precommitment_peer_evidence_open_sparse_p0p1` | main | 5/5 | 5/5 | 0 | 9.0 | 1.00 | 0.000048 | 0.6000 |

Interpretation:

- The readiness-propagation candidate remains stable under open-boundary sparse
  seeds: Toy2 passes 5/5 with mean TtC 9.4, and Toy4 passes 5/5 with mean TtC
  9.0.
- Open boundary plus sparse initial seeds still does not make reputation
  imitation fragile in this quick slice. The baseline remains 5/5 in both toys
  and is slightly faster, with mean TtC 9.2 in Toy2 and 8.8 in Toy4.
- The stress is nevertheless useful diagnostically. The objective+basin and raw
  revision variants again reach ceiling at least once but leave final misses and
  late-flip hazard.
- Plain precommitment evidence removes the final-miss pattern, and peer
  evidence keeps that stability while recovering most of the transition speed:
  Toy2 improves from precommitment TtC 10.8 to peer-evidence TtC 9.4; Toy4
  improves from 9.8 to 9.0.

Conclusion:

- This is a robustness pass for the readiness candidate, not a superiority
  result against reputation imitation.
- The result reinforces the earlier caveat: Toy2/Toy4 are still friendly to
  clean reputation imitation even when initial action seeds are sparse and
  boundaries are open.
- For baseline-fragility evidence, the noisy-reputation stress remains the
  stronger current artifact. The next non-reputation stress should change the
  information or reward structure more directly, for example heterogeneous
  thresholds, sparse/delayed rewards, or topology bottlenecks that reduce
  observable frontier quality rather than only removing periodic wraparound.
