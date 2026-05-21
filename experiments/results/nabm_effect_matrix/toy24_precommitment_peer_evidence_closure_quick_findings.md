# Toy2/Toy4 Precommitment Peer-Evidence Closure Findings

Manifest:
`experiments/evidence/toy24_precommitment_peer_evidence_closure_quick.yaml`

Purpose:

- Preserve a compact closure artifact for the current Toy2/Toy4 mechanism
  chain.
- Test the narrow structural claim that objective+basin gives the neural path
  direction, while precommitment plus peer-readiness evidence removes the
  remaining revision-path fragility.
- Keep the candidate scoped as a NABM unit mechanism, not as evidence that the
  neural path beats classic ABM baselines on transition speed.

Run artifacts:

- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_closure_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_closure_quick_effects.md`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_closure_quick_profile.md`
- `experiments/evidence/results/toy24_precommitment_peer_evidence_closure_quick.summary.md`

Gate result: **pass** overall. The main claim group is
`peer_evidence_closure`, and the best main variant in both Toy2 and Toy4 is
`revision_precommitment_peer_evidence_w1p0`.

| Case | Variant | Role | Final hits | Ever hits | Ever-final misses | Mean TtC | Terminal ceiling rate | Late flip rate | Metric mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | `reputation_imitation` | baseline | 5/5 | 5/5 | 0 | 2.6 | 1.00 | 0.000000 | 3.0000 |
| Toy2 | `objective_basin_w0p5_0p5_h1` | diagnostic | 4/5 | 5/5 | 1 | 20.8 | 0.84 | 0.008807 | 2.9980 |
| Toy2 | `revision_objective_basin_w0p5_0p5_h1` | diagnostic | 4/5 | 5/5 | 1 | 21.0 | 0.76 | 0.009282 | 2.9980 |
| Toy2 | `revision_precommitment_evidence` | diagnostic | 5/5 | 5/5 | 0 | 10.8 | 1.00 | 0.000103 | 3.0000 |
| Toy2 | `revision_precommitment_peer_evidence_w1p0` | main | 5/5 | 5/5 | 0 | 9.4 | 1.00 | 0.000000 | 3.0000 |
| Toy4 | `reputation_imitation` | baseline | 5/5 | 5/5 | 0 | 2.6 | 1.00 | 0.000000 | 0.6000 |
| Toy4 | `objective_basin_w0p5_0p5_h1` | diagnostic | 4/5 | 5/5 | 1 | 19.0 | 0.76 | 0.008454 | 0.5988 |
| Toy4 | `revision_objective_basin_w0p5_0p5_h1` | diagnostic | 3/5 | 5/5 | 2 | 20.6 | 0.72 | 0.007957 | 0.5976 |
| Toy4 | `revision_precommitment_evidence` | diagnostic | 5/5 | 5/5 | 0 | 10.6 | 1.00 | 0.000513 | 0.6000 |
| Toy4 | `revision_precommitment_peer_evidence_w1p0` | main | 5/5 | 5/5 | 0 | 9.0 | 1.00 | 0.000095 | 0.6000 |

Interpretation:

- The diagnostic contrast remains intact. Objective+basin reaches the ceiling
  at least once in every seed, but it leaves final misses and a non-trivial
  late flip rate. This supports the earlier diagnosis that the issue is no
  longer direction alone; the revision path still has a stability problem.
- The raw revision operator does not solve that problem by itself. In Toy4 it
  is worse than objective+basin without revision on final hits, which keeps the
  revision path as the target of the mechanism change.
- Plain precommitment evidence removes the final miss in this closure slice,
  but the peer-readiness evidence variant also shortens the transition tail:
  Toy2 TtC improves from 10.8 to 9.4, and Toy4 improves from 10.6 to 9.0.
- The candidate is therefore best framed as readiness propagation before hard
  commitment. Ready peers provide additional adoption evidence to not-yet-ready
  agents; this is different from terminal argmax, final-window metric changes,
  or sampler stay-bias.
- Reputation imitation remains much faster. This artifact should not be used
  to claim speed superiority over classic ABM dynamics. The useful claim is
  narrower: a neural revision path needs an explicit precommitment readiness
  propagation unit to close the Toy2/Toy4 fragility shown by objective+basin
  and raw revision diagnostics.
- Toy2/Toy4 are also baseline-friendly ceiling tasks for reputation imitation.
  The closure result should therefore be read as a sanity-stage mechanism
  closure under a favorable classic baseline, not as the main robustness claim.
  The next value test is whether the readiness unit holds when reputation
  imitation becomes brittle under noise, heterogeneity, sparse reward, or
  topology constraints.

Conclusion:

- `revision_precommitment_peer_evidence_w1p0` is the current closure candidate
  for Toy2/Toy4.
- The evidence supports promoting `BinaryReadinessPropagationUnit` as the
  structural unit under test, while retaining `revision_precommitment_evidence`
  as the ablation that separates local precommitment from peer propagation.
- The claim is still quick-evidence scope. It should be broadened with holdout
  seeds or a consolidated NABM-unit manifest before being treated as a general
  paper claim.

Recommended next slice:

- Update the core profile index with this closure artifact.
- Run the same candidate against a broader seed holdout before changing paper
  language from "candidate mechanism" to "validated mechanism".
- Keep future changes attached to the named readiness-propagation unit instead
  of adding another sampler-side or endpoint correction.
