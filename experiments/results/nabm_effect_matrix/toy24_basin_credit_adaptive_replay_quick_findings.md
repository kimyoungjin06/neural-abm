# Toy2/Toy4 Basin-Credit Adaptive Replay Findings

Manifest: `experiments/evidence/toy24_basin_credit_adaptive_replay_quick.yaml`

Workflow:

```bash
uv run python scripts/run_basin_credit_evidence_workflow.py --manifest experiments/evidence/toy24_basin_credit_adaptive_replay_quick.yaml
```

Outputs:

- `experiments/results/nabm_effect_matrix/toy24_basin_credit_adaptive_replay_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy24_basin_credit_adaptive_replay_quick_effects.csv`
- `experiments/results/nabm_effect_matrix/toy24_basin_credit_adaptive_replay_quick_pairwise_effects.csv`
- `experiments/results/nabm_effect_matrix/toy24_basin_credit_adaptive_replay_quick_effects.md`
- `experiments/evidence/results/toy24_basin_credit_adaptive_replay_quick.summary.json`
- `experiments/evidence/results/toy24_basin_credit_adaptive_replay_quick.summary.md`

Gate status: `pass`.

The pass is not evidence that the new adaptive schedule is the best variant. The gate passes because an eligible main variant still satisfies each case. The new adaptive candidate should be judged separately against fixed p2/p3 replay.

## Main Result

| Toy | Variant | Final ceiling hits | Mean time to ceiling | Final metric mean | Mean effective replay passes |
| --- | --- | ---: | ---: | ---: | ---: |
| Toy2 | `mixed_objective_basin_replay_all_p2_h1` | 3/3 | 10.33 | 3.000 | 2.00 |
| Toy2 | `mixed_objective_basin_replay_all_p3_h1` | 3/3 | 9.33 | 3.000 | 3.00 |
| Toy2 | `mixed_objective_basin_adaptive_score_p3_min2_h1` | 3/3 | 9.33 | 3.000 | 2.09 |
| Toy4 | `mixed_objective_basin_replay_all_p2_h1` | 3/3 | 13.67 | 0.600 | 2.00 |
| Toy4 | `mixed_objective_basin_replay_all_p3_h1` | 2/3 | 11.67 | 0.598 | 3.00 |
| Toy4 | `mixed_objective_basin_adaptive_score_p3_min2_h1` | 2/3 | 12.67 | 0.598 | 2.09 |

Adaptive replay used `training_pass_schedule=target_score_decay`, `training_passes=3`, `min_training_passes=2`, and `training_pass_score_threshold=0.995`.

Observed effective pass counts:

- Toy2 adaptive: 45-46 epochs used 2 passes, 4-5 epochs used 3 passes, mean pass count 2.08-2.10.
- Toy4 adaptive: 45-46 epochs used 2 passes, 4-5 epochs used 3 passes, mean pass count 2.08-2.10.

## Attribution

The earlier collapse pattern remains confirmed:

- `basin_credit_w1p0_h1_prototype` does not reach ceiling in Toy2 or Toy4.
- `mixed_individual_basin_w0p5_0p5_h1` collapses in both toys.
- `mixed_objective_basin_w0p5_0p5_h1` avoids the collapse but is slow, matching the previous objective-blend interpretation.

Replay scope was not the driver in this quick matrix:

- `revised_p2` and `all_p2` matched in Toy2 and Toy4.
- `revised_p3` and `all_p3` matched in Toy2 and Toy4.
- Under these configs the realized revision rate is effectively all agents, so the scope distinction is not exposed.

## Interpretation

The adaptive schedule is useful as instrumentation and budget control: it proves that per-step replay pass selection is wired through diagnostics and is visible in aggregate metrics.

It does not yet solve the Toy4 speed-stability tradeoff. The Toy4 result suggests that the first 4-5 epochs of p3 replay are enough to reproduce the later final-ceiling miss seen in fixed p3. Decaying from p3 to p2 after the target score is already high saves replay budget, but it does not reverse the early-policy state that produces the seed 3 final slip.

The next mechanism should avoid starting every run with p3 by default. A better direction is a selector that starts from the stable p2 regime and escalates replay only when basin-credit diagnostics indicate weak progress, instead of using p3 early and decaying after target-score saturation.
