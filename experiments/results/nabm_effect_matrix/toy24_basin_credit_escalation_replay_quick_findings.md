# Toy2/Toy4 Basin-Credit Credit-Signal Escalation Findings

Manifest: `experiments/evidence/toy24_basin_credit_escalation_replay_quick.yaml`

Workflow:

```bash
uv run python scripts/run_basin_credit_evidence_workflow.py --manifest experiments/evidence/toy24_basin_credit_escalation_replay_quick.yaml
```

Outputs:

- `experiments/results/nabm_effect_matrix/toy24_basin_credit_escalation_replay_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy24_basin_credit_escalation_replay_quick_effects.csv`
- `experiments/results/nabm_effect_matrix/toy24_basin_credit_escalation_replay_quick_pairwise_effects.csv`
- `experiments/results/nabm_effect_matrix/toy24_basin_credit_escalation_replay_quick_effects.md`
- `experiments/evidence/results/toy24_basin_credit_escalation_replay_quick.summary.json`
- `experiments/evidence/results/toy24_basin_credit_escalation_replay_quick.summary.md`

Gate status: `pass`.

## Main Result

| Toy | Variant | Final ceiling hits | Mean time to ceiling | Final metric mean | Mean effective replay passes |
| --- | --- | ---: | ---: | ---: | ---: |
| Toy2 | `mixed_objective_basin_replay_all_p2_h1` | 3/3 | 10.33 | 3.000 | 2.00 |
| Toy2 | `mixed_objective_basin_replay_all_p3_h1` | 3/3 | 9.33 | 3.000 | 3.00 |
| Toy2 | `mixed_objective_basin_adaptive_score_p3_min2_h1` | 3/3 | 9.33 | 3.000 | 2.09 |
| Toy2 | `mixed_objective_basin_escalate_credit_p3_min2_h1` | 3/3 | 9.33 | 3.000 | 2.96 |
| Toy4 | `mixed_objective_basin_replay_all_p2_h1` | 3/3 | 13.67 | 0.600 | 2.00 |
| Toy4 | `mixed_objective_basin_replay_all_p3_h1` | 2/3 | 11.67 | 0.598 | 3.00 |
| Toy4 | `mixed_objective_basin_adaptive_score_p3_min2_h1` | 2/3 | 12.67 | 0.598 | 2.09 |
| Toy4 | `mixed_objective_basin_escalate_credit_p3_min2_h1` | 3/3 | 11.67 | 0.600 | 2.97 |

The new escalation variant uses:

- `training_scope=all`
- `training_passes=3`
- `training_pass_schedule=credit_signal_escalation`
- `min_training_passes=2`
- `training_pass_credit_positive_threshold=0.6`
- `training_pass_credit_delta_threshold=0.0`

## Interpretation

The prior score-decay schedule saved replay budget but did not fix Toy4's p3 final miss. The credit-signal escalation schedule fixes that miss in this 3-seed quick matrix:

- Toy2 keeps fixed p3 speed: 3/3 hits, mean time-to-ceiling 9.33.
- Toy4 keeps fixed p3 speed: mean time-to-ceiling 11.67.
- Toy4 restores fixed p2 stability: 3/3 final ceiling hits.

This result supports the attribution from the previous run: Toy4 failure is not caused by late overtraining after the basin score saturates. It is more consistent with early amplification of noisy basin credit. The new gate leaves the run at two passes until the selected-action credit is non-negative on average and at least 60% positive, then escalates to three passes.

## Caveat

This is still an internal replay-control mechanism, not a learned basin critic and not enough to claim a fundamentally new algorithm. It is less arbitrary than a pure p2/p3 sweep because the schedule is tied to signed basin-credit diagnostics, but the `0.6` positive-rate threshold remains a hand-set protocol choice.

The practical next step is to reduce this hand-set threshold dependency by learning or calibrating a confidence gate from trajectory outcomes, or by replacing the prototype score with the reserved contrastive critic path.
