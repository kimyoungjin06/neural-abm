# Toy2/Toy4 Revision Operator Precommitment Control Findings

Manifest:
`experiments/evidence/toy24_revision_operator_precommitment_controls_quick.yaml`

Purpose:

- Check whether evidence-accumulation precommitment addresses the pre-ceiling
  transition bottleneck.
- Test the structural hypothesis that the remaining miss is readiness
  propagation, not terminal sampling noise or local ready-to-action conversion.
- Add a small peer-readiness evidence sweep:
  `precommitment_peer_evidence_weight in {0.25, 0.5, 1.0}`.
- The mechanism is implemented as `BinaryReadinessPropagationUnit` in
  `src/neural_abm/readiness.py`, which converts prior ready-state scores into
  peer evidence increments.

Run artifacts:

- `experiments/results/nabm_effect_matrix/toy24_revision_operator_precommitment_controls_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy24_revision_operator_precommitment_controls_quick_effects.md`
- `experiments/evidence/results/toy24_revision_operator_precommitment_controls_quick.summary.md`

Gate result: **pass** overall. The best main variant in both Toy2 and Toy4 is
`revision_operator_precommitment_peer_evidence_w1p0`.

| Case | Variant | Final hits | Ever hits | Ever-final misses | Mean TtC | Terminal ceiling rate | Late flip rate | Metric mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | `revision_operator_mixed_objective_basin_w0p5_0p5_h1` | 2/3 | 3/3 | 1 | 19.33 | 0.867 | 0.00978 | 2.99667 |
| Toy2 | `revision_operator_commitment_hysteresis` | 3/3 | 3/3 | 0 | 15.00 | 1.000 | 0.00056 | 3.00000 |
| Toy2 | `revision_operator_precommitment_evidence` | 3/3 | 3/3 | 0 | 11.00 | 1.000 | 0.00017 | 3.00000 |
| Toy2 | `revision_operator_precommitment_peer_evidence_w0p25` | 3/3 | 3/3 | 0 | 11.00 | 1.000 | 0.00000 | 3.00000 |
| Toy2 | `revision_operator_precommitment_peer_evidence_w0p5` | 3/3 | 3/3 | 0 | 10.33 | 1.000 | 0.00033 | 3.00000 |
| Toy2 | `revision_operator_precommitment_peer_evidence_w1p0` | 3/3 | 3/3 | 0 | 9.67 | 1.000 | 0.00000 | 3.00000 |
| Toy2 | `revision_operator_precommitment_commitment_hysteresis` | 3/3 | 3/3 | 0 | 11.00 | 1.000 | 0.00017 | 3.00000 |
| Toy4 | `revision_operator_mixed_objective_basin_w0p5_0p5_h1` | 1/3 | 3/3 | 2 | 19.00 | 0.800 | 0.00976 | 0.59600 |
| Toy4 | `revision_operator_commitment_hysteresis` | 3/3 | 3/3 | 0 | 13.67 | 0.933 | 0.00148 | 0.60000 |
| Toy4 | `revision_operator_precommitment_evidence` | 3/3 | 3/3 | 0 | 11.00 | 1.000 | 0.00085 | 0.60000 |
| Toy4 | `revision_operator_precommitment_peer_evidence_w0p25` | 3/3 | 3/3 | 0 | 11.00 | 1.000 | 0.00085 | 0.60000 |
| Toy4 | `revision_operator_precommitment_peer_evidence_w0p5` | 3/3 | 3/3 | 0 | 10.67 | 1.000 | 0.00000 | 0.60000 |
| Toy4 | `revision_operator_precommitment_peer_evidence_w1p0` | 3/3 | 3/3 | 0 | 9.33 | 1.000 | 0.00000 | 0.60000 |
| Toy4 | `revision_operator_precommitment_commitment_hysteresis` | 3/3 | 3/3 | 0 | 11.00 | 1.000 | 0.00085 | 0.60000 |

Precommitment trajectory diagnostics:

| Case | Variant | First ready epoch | All ready epoch | First forced epoch | Ready-to-forced delay | Premature exits | Peer weight | Peer increment | Ready component fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | `revision_operator_precommitment_evidence` | 4.00 | 13.67 | 4.67 | 3.50 | 0.00 | 0.00 | 0.00 | 1.00 |
| Toy2 | `revision_operator_precommitment_peer_evidence_w0p25` | 4.00 | 12.67 | 4.67 | 3.31 | 0.00 | 0.25 | 0.25 | 1.00 |
| Toy2 | `revision_operator_precommitment_peer_evidence_w0p5` | 4.00 | 12.00 | 4.67 | 3.27 | 0.00 | 0.50 | 0.50 | 1.00 |
| Toy2 | `revision_operator_precommitment_peer_evidence_w1p0` | 4.00 | 10.00 | 4.67 | 3.01 | 0.00 | 1.00 | 1.00 | 1.00 |
| Toy4 | `revision_operator_precommitment_evidence` | 4.33 | 19.00 | 4.33 | 3.35 | 0.00 | 0.00 | 0.00 | 1.00 |
| Toy4 | `revision_operator_precommitment_peer_evidence_w0p25` | 4.33 | 18.00 | 4.33 | 3.35 | 0.00 | 0.25 | 0.25 | 1.00 |
| Toy4 | `revision_operator_precommitment_peer_evidence_w0p5` | 4.33 | 11.67 | 4.33 | 3.30 | 0.00 | 0.50 | 0.50 | 1.00 |
| Toy4 | `revision_operator_precommitment_peer_evidence_w1p0` | 4.33 | 9.67 | 4.33 | 3.47 | 0.00 | 1.00 | 1.00 | 1.00 |

Interpretation:

- Plain precommitment materially attacks the pre-ceiling transition bottleneck:
  TtC drops from 19.33 to 11.00 in Toy2 and from 19.00 to 11.00 in Toy4.
- Peer-readiness evidence attacks the remaining tail. At weight 1.0, TtC drops
  further to 9.67 in Toy2 and 9.33 in Toy4, crossing the current gate in both
  cases.
- The improvement is monotonic in the sweep for the key propagation metric:
  all-ready epoch falls from 13.67 -> 12.67 -> 12.00 -> 10.00 in Toy2 and
  19.00 -> 18.00 -> 11.67 -> 9.67 in Toy4.
- Terminal stability is preserved: final hits are 3/3, ever-final misses are 0,
  terminal ceiling rate is 1.0, and late flip rate is 0.0 for the weight-1.0
  peer-evidence variant in both toys.
- Commitment hysteresis is not the main explanation. It shortens
  ready-to-forced delay, but does not improve TtC beyond plain precommitment.
  Peer evidence improves TtC by reducing the readiness-propagation tail.

Conclusion:

- This supports the structural mechanism: a pre-ceiling readiness state with
  `BinaryReadinessPropagationUnit` peer-evidence propagation is a better
  intervention than endpoint argmax, stay bias, or pure commitment hysteresis.
- The pass should still be claimed conservatively. The sweep validates the
  mechanism direction on Toy2/Toy4 quick evidence, but it is not yet a final
  generalized NABM unit claim.

Recommended next slice:

- Promote `revision_operator_precommitment_peer_evidence_w1p0` to the next
  candidate reference, while keeping plain precommitment as the ablation.
- Run a broader stability check before expanding claims: more seeds or a
  retained quick-plus manifest with the same fields.
- Keep the candidate path routed through `BinaryReadinessPropagationUnit`
  rather than re-expanding it as another coordination flag cluster.
