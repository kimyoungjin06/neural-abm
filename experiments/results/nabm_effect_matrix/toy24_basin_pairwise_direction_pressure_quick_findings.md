# Toy2/Toy4 Pairwise Direction-Pressure Findings

## Question

Can the direction module move from candidate-context ceiling labels to an
explicit candidate-action label while keeping pressure as a separate replay
weight module?

This slice tests a pseudo-pairwise direction critic:

```text
direction = pairwise candidate-action basin critic
pressure = learned intervention-pressure replay scorer
```

The direction labels are still distilled from the prototype one-step basin
advantage sign and gated by future relevance. They are not true rollout
counterfactual labels.

## Artifacts

- Direction-critic manifest:
  `experiments/evidence/toy24_basin_phase_critic_pairwise_direction_quick.yaml`
- Direction-critic summary:
  `experiments/results/basin_critic/toy24_basin_phase_critic_pairwise_direction_quick_summary.md`
- Evidence manifest:
  `experiments/evidence/toy24_basin_pairwise_direction_pressure_quick.yaml`
- Evidence gate:
  `experiments/evidence/results/toy24_basin_pairwise_direction_pressure_quick.summary.md`
- Effect matrix:
  `experiments/results/nabm_effect_matrix/toy24_basin_pairwise_direction_pressure_quick_effects.md`

## Offline Direction-Critic Quality

| Toy | Label mode | Eval AUC | Pairwise rank | Abstain |
| --- | --- | ---: | ---: | ---: |
| Toy2 | `prototype_direction` | 0.9908 | 0.9908 | 0.0002 |
| Toy4 | `prototype_direction` | 0.9576 | 0.9576 | 0.0126 |

The prototype observed-score AUC is `0.5` in both toys because this label mode
does not ask the critic to recover absolute ceiling proximity. It asks the
critic to recover the candidate-action direction induced by the prototype
basin advantage.

## Runtime Evidence

Gate status: `pass`.

| Case | Main variant | Final ceiling hits | Mean TtC | Baseline TtC |
| --- | --- | ---: | ---: | ---: |
| Toy2 | `learned_pairwise_direction_pressure_scorer_replay` | 3/3 | 9.33 | 23.67 |
| Toy4 | `learned_pairwise_direction_pressure_scorer_replay` | 3/3 | 11.33 | 21.00 |

Relative to the previous candidate-context direction-pressure diagnostic:

| Toy | Candidate-context TtC | Pairwise TtC | Interpretation |
| --- | ---: | ---: | --- |
| Toy2 | 9.33 | 9.33 | Same ceiling arrival in this quick gate. |
| Toy4 | 11.67 | 11.33 | Small improvement in the strict quick gate. |

Final action rate is `1.0` for all successful basin variants in both toys, so
the distinction is in arrival speed and training signal shape, not final policy
state.

## Runtime Diagnostics

| Toy | Variant | Replay weight | Learned credit rate | Learned action1 advantage | Prototype corr. | Training effective advantage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Toy2 | candidate-context direction-pressure | 0.7490 | 1.0 | -0.0054 | -1.0000 | 0.2223 |
| Toy2 | pairwise direction-pressure | 0.5000 | 1.0 | 0.9396 | 0.99998 | 0.6948 |
| Toy4 | candidate-context direction-pressure | 0.9829 | 1.0 | 0.0113 | -1.0000 | 0.1931 |
| Toy4 | pairwise direction-pressure | 0.9994 | 1.0 | 0.9764 | 0.99982 | 0.6757 |

The pairwise critic makes the runtime action direction explicit and large in
magnitude. That is structurally cleaner than inferring a weak direction from a
candidate-context ceiling classifier. However, Toy4 still uses almost all of
the pressure budget, so this is not yet a lower-budget attention controller.

## Conclusion

The pairwise direction-pressure slice is viable:

- It preserves the Toy2/Toy4 quick gate.
- It gives the direction critic explicit candidate-action semantics.
- It slightly improves Toy4 mean TtC in this run.

The claim remains conservative. This is a prototype-distilled pairwise critic,
not a true counterfactual rollout critic. The next structural step is to replace
the pseudo-pairwise direction label with held-out counterfactual rollout or
ablation labels, then test whether the pressure module can reduce replay budget
without losing Toy4 ceiling arrival.
