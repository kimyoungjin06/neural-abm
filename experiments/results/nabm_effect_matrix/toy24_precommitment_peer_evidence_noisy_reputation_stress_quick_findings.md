# Toy2/Toy4 Noisy-Reputation Stress Findings

Manifest:
`experiments/evidence/toy24_precommitment_peer_evidence_noisy_reputation_stress_quick.yaml`

Purpose:

- Turn the Toy2/Toy4 baseline-friendly caveat into a first stress artifact.
- Stress reputation imitation by adding Gaussian noise to reputation-ranked
  peer choice: `model.state.reputation.noise: 1.0`.
- Check whether the current readiness-propagation candidate remains stable
  when the reputation baseline no longer has clean peer-ranking information.
- Keep this as a noisy-reputation probe only; it does not yet cover
  heterogeneous thresholds, sparse rewards, or topology bottlenecks.

Run artifacts:

- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_noisy_reputation_stress_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_noisy_reputation_stress_quick_effects.md`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_noisy_reputation_stress_quick_profile.md`
- `experiments/evidence/results/toy24_precommitment_peer_evidence_noisy_reputation_stress_quick.summary.md`

Gate result: **pass** overall. The main claim group is
`peer_evidence_noisy_reputation_stress`, and the best main variant in both
Toy2 and Toy4 is `revision_precommitment_peer_evidence_noisy_s1p0`.

| Case | Variant | Role | Final hits | Ever hits | Ever-final misses | Mean TtC | Terminal ceiling rate | Late flip rate | Metric mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | `reputation_imitation_noisy_s1p0` | baseline | 4/5 | 4/5 | 0 | 33.25 | 0.80 | 0.000000 | 2.9320 |
| Toy2 | `objective_basin_noisy_s1p0` | diagnostic | 4/5 | 5/5 | 1 | 20.8 | 0.84 | 0.008807 | 2.9980 |
| Toy2 | `revision_objective_basin_noisy_s1p0` | diagnostic | 4/5 | 5/5 | 1 | 21.0 | 0.76 | 0.009282 | 2.9980 |
| Toy2 | `revision_precommitment_evidence_noisy_s1p0` | diagnostic | 5/5 | 5/5 | 0 | 10.8 | 1.00 | 0.000103 | 3.0000 |
| Toy2 | `revision_precommitment_peer_evidence_noisy_s1p0` | main | 5/5 | 5/5 | 0 | 9.4 | 1.00 | 0.000000 | 3.0000 |
| Toy4 | `reputation_imitation_noisy_s1p0` | baseline | 4/5 | 4/5 | 0 | 33.25 | 0.80 | 0.000000 | 0.5700 |
| Toy4 | `objective_basin_noisy_s1p0` | diagnostic | 4/5 | 5/5 | 1 | 19.0 | 0.76 | 0.008454 | 0.5988 |
| Toy4 | `revision_objective_basin_noisy_s1p0` | diagnostic | 3/5 | 5/5 | 2 | 20.6 | 0.72 | 0.007957 | 0.5976 |
| Toy4 | `revision_precommitment_evidence_noisy_s1p0` | diagnostic | 5/5 | 5/5 | 0 | 10.6 | 1.00 | 0.000513 | 0.6000 |
| Toy4 | `revision_precommitment_peer_evidence_noisy_s1p0` | main | 5/5 | 5/5 | 0 | 9.0 | 1.00 | 0.000095 | 0.6000 |

Interpretation:

- The caveat is real and now testable. Clean Toy2/Toy4 are favorable to
  reputation imitation, but noisy reputation makes that baseline slower and
  seed-fragile: both Toy2 and Toy4 drop to 4/5 final hits with mean TtC 33.25.
- The readiness-propagation candidate is stable under this stress slice:
  Toy2 remains 5/5 with mean TtC 9.4, and Toy4 remains 5/5 with mean TtC 9.0.
- The raw objective+basin and raw revision diagnostics still show the older
  failure pattern. They reach ceiling at least once but leave final misses and
  late flip rates, so the result is not merely a reputation-baseline artifact.
- Plain precommitment evidence is already sufficient for final stability in
  this slice, while peer-readiness evidence shortens the transition tail.
- This is the first result where the readiness candidate is better than the
  reputation baseline on both final hits and TtC. The claim should still be
  conservative because the stress is targeted at reputation ranking and does
  not yet cover other ways classic ABM rules can fail.

Implementation note:

- This run exposed a floating-roundoff issue in shared noisy-reputation
  probabilities: weighted peer-action averages can produce values such as
  `1.0000000000000002`. `src/neural_abm/reputation.py` now clips returned
  cooperation probabilities to the probability contract `[0, 1]`, with a
  regression test in `tests/test_reputation_mobility.py`.

Conclusion:

- The current best claim is now stronger than the clean closure claim:
  `BinaryReadinessPropagationUnit` remains stable on Toy2/Toy4 when the
  reputation baseline is made noisy enough to become slow and seed-fragile.
- This supports continuing with stress generalization rather than more clean
  Toy2/Toy4 tuning.
- The next slices should test whether the same unit survives non-reputation
  fragility: heterogeneous thresholds, sparse or delayed reward, and topology
  bottlenecks.
