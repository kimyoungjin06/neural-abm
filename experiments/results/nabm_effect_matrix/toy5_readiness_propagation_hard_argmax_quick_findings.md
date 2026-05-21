# Toy5 Readiness Propagation Hard-Argmax Findings

Manifest:
`experiments/evidence/toy5_readiness_propagation_hard_argmax_quick.yaml`

Purpose:

- Replace the saturated Toy5 holdout with a harder binary-domain check.
- Use `argmax` decisions to remove sampled adoption accumulation as the main
  path to full cascade.
- Require a 0.95 final action-rate ceiling so partial cascades no longer count
  as successful endpoint recovery.
- Compare baseline output averaging, plain precommitment, peer-readiness
  propagation, and a direction-gated diagnostic.

Run artifacts:

- `experiments/results/nabm_effect_matrix/toy5_readiness_propagation_hard_argmax_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy5_readiness_propagation_hard_argmax_quick_effects.md`
- `experiments/evidence/results/toy5_readiness_propagation_hard_argmax_quick.summary.md`

Gate result: **pass**. The main readiness-propagation candidate reaches the
0.95 action-rate ceiling in all three seeds with mean time-to-ceiling 4.33.

| Variant | Group | Final hits | Mean TtC | Final cascade size | Terminal ceiling rate | Late flip rate | All-ready epoch | Peer readiness | Peer increment |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neural_argmax_output_average` | baseline | 0/3 |  | 74.67 | 0.0000 |  |  | 0.0000 | 0.0000 |
| `neural_argmax_precommitment_evidence` | diagnostic | 0/3 |  | 75.00 | 0.0000 |  |  | 0.0000 | 0.0000 |
| `neural_argmax_readiness_propagation_w1p0` | readiness_hard | 3/3 | 4.33 | 100.00 | 1.0000 | 0.0009 | 5.00 | 1.0000 | 1.0000 |
| `neural_argmax_readiness_direction_gated_w1p0` | diagnostic | 0/3 |  | 74.67 | 0.0000 |  |  | 0.0000 | 0.0000 |

Interpretation:

- This is a stronger holdout than the first Toy5 check. The baseline no longer
  saturates: `neural_argmax_output_average` stays around 73-76 adopted agents.
- Plain precommitment does not explain the improvement. It also stays below the
  0.95 ceiling, with final cascade size essentially unchanged.
- The active difference is peer-readiness propagation. The candidate reaches
  full cascade in all seeds and records peer-readiness and peer-increment means
  of 1.0.
- The direction-gated diagnostic is important: when readiness evidence requires
  local positive threshold direction, the effect disappears. The current Toy5
  hard result therefore supports a narrower claim about non-directional
  readiness-state propagation, not an objective-direction-gated contagion
  mechanism.

Conclusion:

- The hard Toy5 holdout provides endpoint evidence that the shared readiness
  unit can recover a failure mode that output averaging and plain
  precommitment do not recover.
- The claim should remain conservative: this is a structural readiness
  propagation result under argmax binary decisions, with a clear caveat that
  direction-gated readiness does not yet work in this Toy5 regime.
- The next research step should be a direction-aware version of readiness
  propagation, or a domain condition where positive local direction exists
  before full adoption and peer readiness can still percolate.
