# Evidence Gate: toy24_basin_learned_credit_future_motion_weight_scorer_quick

Overall status: **fail**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | --- |
| toy2_basin_learned_credit_future_motion_weight_scorer | toy2 | pass | learned_candidate_context_future_motion_weight_scorer_replay | 3/3 | 9.33333 | true |
| toy4_basin_learned_credit_future_motion_weight_scorer | toy4 | fail | learned_candidate_context_future_motion_weight_scorer_replay | 3/3 | 12 | true |

## Next Diagnostics

- toy4_basin_learned_credit_future_motion_weight_scorer: inspect time-to-ceiling trajectories and seed variance.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| toy2_basin_learned_credit_future_motion_weight_scorer | linear_welfare_heavy | baseline | false | diagnostic_only | 3 | 23.6667 | 3 | not in main claim group |
| toy2_basin_learned_credit_future_motion_weight_scorer | mixed_objective_basin_escalate_credit_p3_min2_h1_prototype | diagnostic | false | diagnostic_only | 3 | 9.33333 | 3 | not in main claim group |
| toy2_basin_learned_credit_future_motion_weight_scorer | learned_candidate_context_all_replay | diagnostic | false | diagnostic_only | 3 | 9.33333 | 3 | not in main claim group |
| toy2_basin_learned_credit_future_motion_weight_scorer | learned_candidate_context_confident_agreement_soft_min50_replay | diagnostic | false | diagnostic_only | 3 | 9.33333 | 3 | not in main claim group |
| toy2_basin_learned_credit_future_motion_weight_scorer | learned_candidate_context_weight_scorer_replay | diagnostic | false | diagnostic_only | 3 | 9.33333 | 3 | not in main claim group |
| toy2_basin_learned_credit_future_motion_weight_scorer | learned_candidate_context_future_motion_weight_scorer_replay | nabm | true | pass | 3 | 9.33333 | 3 |  |
| toy4_basin_learned_credit_future_motion_weight_scorer | linear_welfare_heavy | baseline | false | diagnostic_only | 1 | 21 | 0.596 | not in main claim group |
| toy4_basin_learned_credit_future_motion_weight_scorer | mixed_objective_basin_escalate_credit_p3_min2_h1_prototype | diagnostic | false | diagnostic_only | 3 | 11.6667 | 0.6 | not in main claim group |
| toy4_basin_learned_credit_future_motion_weight_scorer | learned_candidate_context_all_replay | diagnostic | false | diagnostic_only | 3 | 11.6667 | 0.6 | not in main claim group |
| toy4_basin_learned_credit_future_motion_weight_scorer | learned_candidate_context_confident_agreement_soft_min50_replay | diagnostic | false | diagnostic_only | 3 | 11.6667 | 0.6 | not in main claim group |
| toy4_basin_learned_credit_future_motion_weight_scorer | learned_candidate_context_weight_scorer_replay | diagnostic | false | diagnostic_only | 3 | 11.6667 | 0.6 | not in main claim group |
| toy4_basin_learned_credit_future_motion_weight_scorer | learned_candidate_context_future_motion_weight_scorer_replay | nabm | true | fail | 3 | 12 | 0.6 | mean time-to-ceiling 12 >= 12 |
