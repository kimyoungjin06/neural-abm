# Toy2/Toy4 Future-Outcome Direction-Pressure Findings

## Question

Can the direction critic stop using the prototype basin-advantage sign as its
direct pairwise label and instead learn from observed future basin motion?

This slice tests:

```text
direction = future-outcome candidate-action basin critic
pressure = learned intervention-pressure replay scorer
```

The new `future_outcome_direction` label uses observed trajectory outcome:

- If the observed trajectory improves future basin payoff or reaches/maintains
  the ceiling within the horizon, the observed action is the positive candidate.
- If future basin payoff worsens, the counterfactual action is the positive
  candidate.

This is not a true rollout counterfactual label. It is an observed-outcome
direction label that removes the direct prototype-sign target used by
`prototype_direction`.

## Artifacts

- Direction-critic manifest:
  `experiments/evidence/toy24_basin_phase_critic_future_outcome_direction_quick.yaml`
- Direction-critic summary:
  `experiments/results/basin_critic/toy24_basin_phase_critic_future_outcome_direction_quick_summary.md`
- Evidence manifest:
  `experiments/evidence/toy24_basin_future_outcome_direction_pressure_quick.yaml`
- Evidence gate:
  `experiments/evidence/results/toy24_basin_future_outcome_direction_pressure_quick.summary.md`
- Effect matrix:
  `experiments/results/nabm_effect_matrix/toy24_basin_future_outcome_direction_pressure_quick_effects.md`

## Offline Direction-Critic Quality

| Toy | Label mode | Eval AUC | Pairwise rank | Abstain |
| --- | --- | ---: | ---: | ---: |
| Toy2 | `future_outcome_direction` | 0.9586 | 0.9586 | 0.0000 |
| Toy4 | `future_outcome_direction` | 0.9584 | 0.9584 | 0.0106 |

The learned label is slightly weaker than the prototype-distilled pairwise
critic on Toy2, but still well above chance on both toys.

## Runtime Evidence

Gate status: `pass`.

| Case | Main variant | Final ceiling hits | Mean TtC | Baseline TtC |
| --- | --- | ---: | ---: | ---: |
| Toy2 | `learned_future_outcome_direction_pressure_scorer_replay` | 3/3 | 9.33 | 23.67 |
| Toy4 | `learned_future_outcome_direction_pressure_scorer_replay` | 3/3 | 11.33 | 21.00 |

Relative to the prototype-distilled pairwise direction diagnostic:

| Toy | Prototype-pairwise TtC | Future-outcome TtC | Interpretation |
| --- | ---: | ---: | --- |
| Toy2 | 9.33 | 9.33 | Same quick-gate ceiling arrival. |
| Toy4 | 11.33 | 11.33 | Same quick-gate ceiling arrival. |

Relative to the objective+prototype diagnostic:

| Toy | Objective+prototype TtC | Future-outcome TtC |
| --- | ---: | ---: |
| Toy2 | 9.33 | 9.33 |
| Toy4 | 11.67 | 11.33 |

## Runtime Diagnostics

| Toy | Variant | Replay weight | Learned action1 advantage | Prototype corr. | Training effective advantage |
| --- | --- | ---: | ---: | ---: | ---: |
| Toy2 | prototype-pairwise direction-pressure | 0.5000 | 0.9396 | 0.99998 | 0.6948 |
| Toy2 | future-outcome direction-pressure | 0.5000 | 0.9089 | 0.99995 | 0.6794 |
| Toy4 | prototype-pairwise direction-pressure | 0.9994 | 0.9764 | 0.99982 | 0.6757 |
| Toy4 | future-outcome direction-pressure | 0.9993 | 0.9213 | 0.99975 | 0.6481 |

The future-outcome critic is still highly correlated with prototype direction
at runtime because the successful quick trajectories are strongly monotone
toward action 1. The important change is label provenance: the critic is now
trained from observed future basin motion instead of the prototype action-1
advantage sign.

## Conclusion

This slice is a structural step, not a performance breakthrough:

- It preserves the Toy2/Toy4 quick gate.
- It matches the prototype-pairwise direction-pressure runtime result.
- It removes the direct prototype-sign pairwise label from direction training.

The remaining limitation is causal. `future_outcome_direction` uses observed
trajectory outcomes and only assigns the counterfactual action when the
observed trajectory worsens. It does not estimate what would have happened if
the alternative action had actually been rolled out. The next structural slice
should therefore generate held-out counterfactual rollout or ablation labels.
