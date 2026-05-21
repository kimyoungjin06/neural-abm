# Toy2/Toy4 Reputation-Fragility Stress Findings

Manifest:
`experiments/evidence/toy24_precommitment_peer_evidence_reputation_fragility_stress_quick.yaml`

Purpose:

- Combine the two earlier Toy2/Toy4 stress knobs that probe the
  reputation-imitation caveat:
  `domain.environment.initial_action_probability: 0.1`,
  open/non-periodic boundaries, and `model.state.reputation.noise: 1.0`.
- Check whether the clean Toy2/Toy4 reputation baseline is fragile once its
  peer-ranking information and spatial wraparound support are both weakened.
- Keep the NABM candidate unchanged: objective+basin, revision operator,
  evidence precommitment, and peer evidence with weight `1.0`.
- Treat this as a targeted baseline-fragility artifact, not as a general
  superiority claim.

Run artifacts:

- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_reputation_fragility_stress_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_reputation_fragility_stress_quick_effects.md`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_reputation_fragility_stress_quick_profile.md`
- `experiments/evidence/results/toy24_precommitment_peer_evidence_reputation_fragility_stress_quick.summary.md`

Gate result: **pass** overall. The main claim group is
`peer_evidence_reputation_fragility_stress`, and the best main variant in both
Toy2 and Toy4 is
`revision_precommitment_peer_evidence_open_sparse_noisy_p0p1_s1p0`.

| Case | Variant | Role | Final hits | Ever hits | Ever-final misses | Mean TtC | Terminal ceiling rate | Late flip rate | Metric mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | `reputation_imitation_open_sparse_noisy_p0p1_s1p0` | baseline | 0/5 | 0/5 | 0 | n/a | 0.00 | n/a | 2.4973 |
| Toy2 | `objective_basin_open_sparse_noisy_p0p1_s1p0` | diagnostic | 4/5 | 5/5 | 1 | 19.0 | 0.84 | 0.011186 | 2.9985 |
| Toy2 | `revision_objective_basin_open_sparse_noisy_p0p1_s1p0` | diagnostic | 4/5 | 5/5 | 1 | 20.2 | 0.76 | 0.009850 | 2.9975 |
| Toy2 | `revision_precommitment_evidence_open_sparse_noisy_p0p1_s1p0` | diagnostic | 5/5 | 5/5 | 0 | 10.8 | 1.00 | 0.000103 | 3.0000 |
| Toy2 | `revision_precommitment_peer_evidence_open_sparse_noisy_p0p1_s1p0` | main | 5/5 | 5/5 | 0 | 9.4 | 1.00 | 0.000000 | 3.0000 |
| Toy4 | `reputation_imitation_open_sparse_noisy_p0p1_s1p0` | baseline | 0/5 | 0/5 | 0 | n/a | 0.00 | n/a | 0.4083 |
| Toy4 | `objective_basin_open_sparse_noisy_p0p1_s1p0` | diagnostic | 4/5 | 5/5 | 1 | 18.2 | 0.80 | 0.007786 | 0.5986 |
| Toy4 | `revision_objective_basin_open_sparse_noisy_p0p1_s1p0` | diagnostic | 4/5 | 5/5 | 1 | 19.0 | 0.80 | 0.007267 | 0.5986 |
| Toy4 | `revision_precommitment_evidence_open_sparse_noisy_p0p1_s1p0` | diagnostic | 5/5 | 5/5 | 0 | 9.8 | 1.00 | 0.000540 | 0.6000 |
| Toy4 | `revision_precommitment_peer_evidence_open_sparse_noisy_p0p1_s1p0` | main | 5/5 | 5/5 | 0 | 9.0 | 1.00 | 0.000048 | 0.6000 |

Interpretation:

- This stress makes the reputation baseline genuinely fragile. It does not
  merely slow the baseline or expose a final-epoch miss: the baseline reaches
  ceiling in `0/5` seeds for both Toy2 and Toy4.
- The peer-evidence candidate remains stable: Toy2 reaches `5/5` final hits
  with mean TtC `9.4`, and Toy4 reaches `5/5` final hits with mean TtC `9.0`.
- The raw objective+basin and raw revision diagnostics still show the known
  final-miss pattern. They reach ceiling at least once in all seeds but finish
  at `4/5`, so evidence precommitment remains the part that removes late
  hazard.
- Peer evidence improves transition time over plain precommitment evidence:
  Toy2 improves from `10.8` to `9.4`, and Toy4 improves from `9.8` to `9.0`.

Conclusion:

- This is now the strongest Toy2/Toy4 artifact for the baseline-fit caveat:
  the NABM candidate survives a condition that directly breaks clean
  reputation imitation.
- The claim must remain bounded. Because the stress includes reputation noise,
  it is targeted at the reputation baseline's information channel rather than
  proving general NABM superiority.
- The next Toy2/Toy4 evidence step should move away from reputation-targeted
  perturbations and test non-reputation fragility, such as delayed/sparse
  reward, heterogeneous observation quality, or topology bottlenecks.
