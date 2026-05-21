# Toy2/Toy4 Precommitment Peer-Evidence Stability Findings

Manifest:
`experiments/evidence/toy24_revision_operator_precommitment_peer_evidence_stability_quick.yaml`

Purpose:

- Preserve a focused 10-seed stability check for the peer-readiness evidence
  mechanism.
- Compare the candidate against three references: reputation imitation, the
  mixed objective+basin revision operator, and plain precommitment evidence.
- Keep this as a quick-plus stability artifact, not a broad generalized claim.
- The candidate mechanism is now named `BinaryReadinessPropagationUnit` in
  `src/neural_abm/readiness.py`: prior ready-state scores are averaged over
  peers and converted into precommitment evidence increments for not-yet-active
  agents.

Run artifacts:

- `experiments/results/nabm_effect_matrix/toy24_revision_operator_precommitment_peer_evidence_stability_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy24_revision_operator_precommitment_peer_evidence_stability_quick_effects.md`
- `experiments/evidence/results/toy24_revision_operator_precommitment_peer_evidence_stability_quick.summary.md`

Gate result: **pass** overall.

| Case | Variant | Final hits | Ever-final misses | Mean TtC | Terminal ceiling rate | Late flip rate | Metric mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | `reputation_imitation` | 10/10 | 0 | 2.70 | 1.00 | 0.00000 | 3.0000 |
| Toy2 | `revision_operator_mixed_objective_basin_w0p5_0p5_h1` | 9/10 | 1 | 20.40 | 0.76 | 0.00963 | 2.9990 |
| Toy2 | `revision_operator_precommitment_evidence` | 10/10 | 0 | 10.90 | 1.00 | 0.00015 | 3.0000 |
| Toy2 | `revision_operator_precommitment_peer_evidence_w1p0` | 10/10 | 0 | 9.50 | 1.00 | 0.00000 | 3.0000 |
| Toy4 | `reputation_imitation` | 10/10 | 0 | 2.70 | 1.00 | 0.00000 | 0.6000 |
| Toy4 | `revision_operator_mixed_objective_basin_w0p5_0p5_h1` | 8/10 | 2 | 19.50 | 0.74 | 0.00797 | 0.5988 |
| Toy4 | `revision_operator_precommitment_evidence` | 10/10 | 0 | 10.40 | 1.00 | 0.00036 | 0.6000 |
| Toy4 | `revision_operator_precommitment_peer_evidence_w1p0` | 10/10 | 0 | 8.90 | 1.00 | 0.00010 | 0.6000 |

Candidate seed detail:

| Case | TtC values | Final hits | Ever-final misses | All-ready epoch mean |
| --- | --- | ---: | ---: | ---: |
| Toy2 | 9, 9, 11, 9, 9, 10, 10, 10, 9, 9 | 10/10 | 0 | 9.80 |
| Toy4 | 9, 9, 10, 8, 9, 9, 8, 9, 9, 9 | 10/10 | 0 | 9.60 |

Interpretation:

- The 10-seed check preserves the quick-sweep conclusion. The candidate
  `revision_operator_precommitment_peer_evidence_w1p0` passes both cases under
  the focused stability criteria.
- The candidate improves TtC over plain precommitment while preserving final
  ceiling stability:
  Toy2 improves from 10.90 to 9.50, and Toy4 improves from 10.40 to 8.90.
- The all-ready tail remains the measured path of improvement:
  Toy2 all-ready epoch is 9.80 for the candidate, and Toy4 is 9.60.
- The mixed objective+basin revision operator remains the diagnostic failure
  contrast: Toy2 has 1 ever-final miss and Toy4 has 2 ever-final misses across
  10 seeds.
- Reputation imitation remains faster, so the claim should not be framed as
  beating classic ABM baselines on speed. The claim is narrower: the neural
  revision path needs an explicit readiness propagation process to avoid the
  pre-ceiling bottleneck.

Conclusion:

- Peer-readiness evidence is now the strongest candidate mechanism in this
  branch.
- It is a structural ABM-native intervention: ready peers contribute adoption
  evidence before hard commitment, instead of post-hoc terminal argmax or
  sampler stay-bias.
- The implementation should keep this as a named readiness-propagation unit,
  while treating the 10-seed quick-plus result as a paper-stage candidate rather
  than a final generalized claim.
