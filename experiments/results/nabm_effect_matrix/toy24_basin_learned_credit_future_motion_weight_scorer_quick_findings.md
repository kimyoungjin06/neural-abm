# Toy2/Toy4 Future-Motion Replay Weight Scorer Findings

Date: 2026-05-14

## Question

Can the learned replay-weight scorer stop imitating the current
`training_effective_advantage` magnitude and instead learn from forward basin
motion?

This slice adds a future-motion label to basin transition samples:

- `future_basin_horizon`
- `future_mean_payoff`
- `future_basin_score_delta`
- `future_ceiling_reached`
- `future_epochs_to_ceiling`
- `future_basin_motion_positive`

The new scorer target mode is `future_basin_motion`: positive
`future_basin_score_delta`, scaled by the train-set target quantile. The runtime
mechanism is still `learned_credit_replay_mode: learned_weight`; only the
offline supervision target changed.

## Artifacts

- Scorer training:
  `experiments/results/basin_critic/toy24_basin_replay_weight_scorer_future_motion_h5_q90_quick_summary.md`
- Evidence manifest:
  `experiments/evidence/toy24_basin_learned_credit_future_motion_weight_scorer_quick.yaml`
- Evidence gate:
  `experiments/evidence/results/toy24_basin_learned_credit_future_motion_weight_scorer_quick.summary.md`
- Runs:
  `experiments/results/nabm_effect_matrix/toy24_basin_learned_credit_future_motion_weight_scorer_quick_runs.csv`

## Offline Scorer Quality

| Toy | Target | Horizon | Eval MSE | Eval Corr | Eval Weight Mean | Eval Target Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Toy2 | future_basin_motion | 5 | 0.000377 | 0.990092 | 0.552042 | 0.542566 |
| Toy4 | future_basin_motion | 5 | 0.001043 | 0.971589 | 0.556355 | 0.543421 |

The target is learnable in the offline artifact. The important behavioral issue
is not fit quality; it is target scale. Compared with the previous
magnitude-supervised scorer, the future-motion scorer emits much weaker replay
weights.

## Evidence Result

Gate status: fail.

| Toy | Variant | Final Hits | Mean TtC | Final Payoff | Final Action Rate | Mean Replay Weight |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Toy2 | prototype escalation | 3/3 | 9.33 | 3.0 | 1.0 | 1.000 |
| Toy2 | magnitude scorer | 3/3 | 9.33 | 3.0 | 1.0 | 0.749 |
| Toy2 | future-motion scorer | 3/3 | 9.33 | 3.0 | 1.0 | 0.514 |
| Toy4 | prototype escalation | 3/3 | 11.67 | 0.6 | 1.0 | 1.000 |
| Toy4 | magnitude scorer | 3/3 | 11.67 | 0.6 | 1.0 | 0.985 |
| Toy4 | future-motion scorer | 3/3 | 12.00 | 0.6 | 1.0 | 0.514 |

Toy2 is insensitive to the lower replay weight in this quick gate. Toy4 is not:
seed TtCs were 12, 11, and 13, so the mean lands exactly at 12.0 and fails the
strict `mean_time_to_ceiling_lt: 12` criterion.

## Interpretation

This is a real structural change in the supervision signal, but it is not yet a
better mechanism than the magnitude scorer.

What improved structurally:

- The replay scorer no longer has to imitate the same transition-signal
  magnitude that is already driving the update.
- The training target is tied to observed forward basin motion.
- The transition sample artifact now carries reusable future labels for later
  critic and replay experiments.

What did not improve behaviorally:

- The future-motion target collapses the runtime weight toward the configured
  floor.
- Toy4 needs stronger replay pressure than this target currently provides.
- The quick gate cannot claim the future-motion scorer as a replacement for the
  magnitude scorer.

## Next Step

Keep the future-motion labels, but do not promote this h5/q90 scorer as the main
runtime policy. The next structural attempt should combine future motion with a
credit-action ranking or advantage sign constraint, rather than only predicting
absolute forward payoff movement. A useful candidate is a pairwise scorer target:
prefer replay candidates whose action-1 credit points toward states that reach
the basin sooner within the future horizon.
