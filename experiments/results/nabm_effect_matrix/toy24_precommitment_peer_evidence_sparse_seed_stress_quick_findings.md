# Toy2/Toy4 Sparse-Seed Stress Findings

Manifest:
`experiments/evidence/toy24_precommitment_peer_evidence_sparse_seed_stress_quick.yaml`

Purpose:

- Move beyond the noisy-reputation stress by testing a stress that does not
  directly target the reputation baseline's observation channel.
- Reduce the initial cooperative/action-1 seed rate to
  `domain.environment.initial_action_probability: 0.1`.
- Check whether the readiness-propagation candidate remains stable when the
  adoption signal starts from a sparse initial action frontier.
- Treat this as an exploratory propagation stress, not as a guaranteed
  baseline-fragility test.

Run artifacts:

- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_sparse_seed_stress_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_sparse_seed_stress_quick_effects.md`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_sparse_seed_stress_quick_profile.md`
- `experiments/evidence/results/toy24_precommitment_peer_evidence_sparse_seed_stress_quick.summary.md`

Gate result: **pass** overall. The main claim group is
`peer_evidence_sparse_seed_stress`, and the best main variant in both Toy2 and
Toy4 is `revision_precommitment_peer_evidence_sparse_p0p1`.

| Case | Variant | Role | Final hits | Ever hits | Ever-final misses | Mean TtC | Terminal ceiling rate | Late flip rate | Metric mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | `reputation_imitation_sparse_p0p1` | baseline | 5/5 | 5/5 | 0 | 7.6 | 1.00 | 0.000000 | 3.0000 |
| Toy2 | `objective_basin_sparse_p0p1` | diagnostic | 4/5 | 5/5 | 1 | 20.8 | 0.84 | 0.009223 | 2.9980 |
| Toy2 | `revision_objective_basin_sparse_p0p1` | diagnostic | 4/5 | 5/5 | 1 | 20.2 | 0.72 | 0.009322 | 2.9980 |
| Toy2 | `revision_precommitment_evidence_sparse_p0p1` | diagnostic | 5/5 | 5/5 | 0 | 10.8 | 1.00 | 0.000103 | 3.0000 |
| Toy2 | `revision_precommitment_peer_evidence_sparse_p0p1` | main | 5/5 | 5/5 | 0 | 9.4 | 1.00 | 0.000000 | 3.0000 |
| Toy4 | `reputation_imitation_sparse_p0p1` | baseline | 5/5 | 5/5 | 0 | 7.6 | 1.00 | 0.000000 | 0.6000 |
| Toy4 | `objective_basin_sparse_p0p1` | diagnostic | 4/5 | 5/5 | 1 | 19.2 | 0.80 | 0.007387 | 0.5988 |
| Toy4 | `revision_objective_basin_sparse_p0p1` | diagnostic | 4/5 | 5/5 | 1 | 19.8 | 0.80 | 0.006718 | 0.5988 |
| Toy4 | `revision_precommitment_evidence_sparse_p0p1` | diagnostic | 5/5 | 5/5 | 0 | 10.2 | 1.00 | 0.000195 | 0.6000 |
| Toy4 | `revision_precommitment_peer_evidence_sparse_p0p1` | main | 5/5 | 5/5 | 0 | 8.8 | 1.00 | 0.000000 | 0.6000 |

Interpretation:

- The readiness-propagation candidate remains stable under sparse initial
  action seeds: Toy2 passes 5/5 with mean TtC 9.4, and Toy4 passes 5/5 with
  mean TtC 8.8.
- Sparse seed stress does not make reputation imitation fragile in this quick
  slice. The baseline remains 5/5 in both toys and is still faster, with mean
  TtC 7.6.
- The older diagnostic contrast remains visible: objective+basin and raw
  revision still reach ceiling at least once but leave final misses and late
  flip rates.
- Peer-readiness evidence again improves transition speed over plain
  precommitment while preserving final stability: Toy2 improves from 10.8 to
  9.4, and Toy4 improves from 10.2 to 8.8.

Conclusion:

- This is a robustness pass for the readiness candidate, but not a superiority
  result against reputation imitation.
- Sparse initial action probability alone is not a strong enough
  non-reputation stress for Toy2/Toy4 because reputation imitation still finds
  and amplifies the small cooperative frontier.
- The next non-reputation stress should be stronger than p=0.1 sparse seeds:
  topology bottlenecks, non-periodic/open boundaries plus sparse seeds, or a
  Toy4 resource shock where payoff/resource dynamics make direct imitation
  less reliable.
