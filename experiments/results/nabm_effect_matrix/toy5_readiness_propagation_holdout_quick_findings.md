# Toy5 Readiness Propagation Holdout Findings

Manifest:
`experiments/evidence/toy5_readiness_propagation_holdout_quick.yaml`

Purpose:

- Check whether `BinaryReadinessPropagationUnit` is wired cleanly outside the
  Toy2/Toy4 basin-credit setting.
- Use Toy5 contagion as a binary-domain holdout before expanding the mechanism
  into a broader NABM unit claim.
- Treat this as diagnostic evidence, not as a new performance claim.

Run artifacts:

- `experiments/results/nabm_effect_matrix/toy5_readiness_propagation_holdout_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy5_readiness_propagation_holdout_quick_effects.md`
- `experiments/evidence/results/toy5_readiness_propagation_holdout_quick.summary.md`

Gate result: **pass**, but with no performance improvement over the neural
baseline. The candidate reaches the 50% adoption ceiling in all three seeds,
as does `neural_output_average`.

| Variant | Group | Final hits | Mean TtC | Final cascade size | Terminal ceiling rate | Late flip rate | All-ready epoch | Peer readiness | Peer increment |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex_threshold` | diagnostic | 1/3 | 15.00 | 34.00 | 0.3333 | 0.0134 |  | 0.0000 | 0.0000 |
| `neural_output_average` | baseline | 3/3 | 1.33 | 100.00 | 1.0000 | 0.0079 |  | 0.0000 | 0.0000 |
| `neural_precommitment_evidence` | diagnostic | 3/3 | 1.33 | 100.00 | 1.0000 | 0.0074 | 21.00 | 0.0000 | 0.0000 |
| `neural_readiness_propagation_w1p0` | readiness_holdout | 3/3 | 1.33 | 100.00 | 1.0000 | 0.0074 | 4.67 | 1.0000 | 1.0000 |

Interpretation:

- Toy5 does not provide a new endpoint-performance win. The neural
  output-average baseline already reaches full cascade in all three seeds, so
  the candidate's grouped effect is exactly 0 on final cascade size.
- The holdout does confirm the structural wiring: the readiness propagation
  path is active in Toy5, with final peer-readiness and peer-increment means of
  1.0.
- The main observable difference is internal timing. Plain precommitment
  reaches all-ready at mean epoch 21.00, while peer readiness propagation
  reaches all-ready at mean epoch 4.67.
- No premature ready exits were observed in either precommitment variant.
- The classic complex-threshold diagnostic remains seed-sensitive under this
  first-agent, 1% initial-action setting: only one of three seeds cascades.

Conclusion:

- This supports a narrow structural claim: `BinaryReadinessPropagationUnit` is
  not Toy2/Toy4-specific and can be routed through another binary domain.
- It does **not** support a stronger Toy5 performance claim, because the neural
  baseline is already at the endpoint ceiling.
- The next step should be either a harder Toy5 holdout where the neural
  baseline is not saturated, or documentation that keeps the current claim at
  the level of readiness timing and cross-domain wiring.
