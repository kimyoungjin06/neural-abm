# Evidence Gate: toy24_basin_learned_credit_replay_selection_quick

Overall status: **fail**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | --- |
| toy2_basin_learned_credit_replay_selection | toy2 | fail | learned_candidate_context_confident_agreement_replay | 0/3 |  | false |
| toy4_basin_learned_credit_replay_selection | toy4 | fail | learned_candidate_context_confident_agreement_replay | 0/3 |  | false |

## Next Diagnostics

- toy2_basin_learned_credit_replay_selection: inspect aggregate trajectories before adding a contrastive critic.
- toy4_basin_learned_credit_replay_selection: inspect aggregate trajectories before adding a contrastive critic.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| toy2_basin_learned_credit_replay_selection | linear_welfare_heavy | baseline | false | diagnostic_only | 3 | 23.6667 | 3 | not in main claim group |
| toy2_basin_learned_credit_replay_selection | mixed_objective_basin_escalate_credit_p3_min2_h1_prototype | diagnostic | false | diagnostic_only | 3 | 9.33333 | 3 | not in main claim group |
| toy2_basin_learned_credit_replay_selection | learned_candidate_context_all_replay | diagnostic | false | diagnostic_only | 3 | 9.33333 | 3 | not in main claim group |
| toy2_basin_learned_credit_replay_selection | learned_candidate_context_confident_agreement_replay | nabm | true | fail | 0 |  | 2.32833 | final ceiling hits 0 < 3 |
| toy2_basin_learned_credit_replay_selection | learned_candidate_context_confident_disagreement_replay | diagnostic | false | diagnostic_only | 3 | 24.6667 | 3 | not in main claim group |
| toy4_basin_learned_credit_replay_selection | linear_welfare_heavy | baseline | false | diagnostic_only | 1 | 21 | 0.596 | not in main claim group |
| toy4_basin_learned_credit_replay_selection | mixed_objective_basin_escalate_credit_p3_min2_h1_prototype | diagnostic | false | diagnostic_only | 3 | 11.6667 | 0.6 | not in main claim group |
| toy4_basin_learned_credit_replay_selection | learned_candidate_context_all_replay | diagnostic | false | diagnostic_only | 3 | 11.6667 | 0.6 | not in main claim group |
| toy4_basin_learned_credit_replay_selection | learned_candidate_context_confident_agreement_replay | nabm | true | fail | 0 |  | 0.298 | final ceiling hits 0 < 2 |
| toy4_basin_learned_credit_replay_selection | learned_candidate_context_confident_disagreement_replay | diagnostic | false | diagnostic_only | 0 |  | 0.276 | not in main claim group |
