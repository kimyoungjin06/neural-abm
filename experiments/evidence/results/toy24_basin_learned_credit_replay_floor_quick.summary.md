# Evidence Gate: toy24_basin_learned_credit_replay_floor_quick

Overall status: **fail**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | --- |
| toy2_basin_learned_credit_replay_floor | toy2 | fail | learned_candidate_context_confident_agreement_floor50_replay | 3/3 | 16 | true |
| toy4_basin_learned_credit_replay_floor | toy4 | fail | learned_candidate_context_confident_agreement_floor50_replay | 3/3 | 14.6667 | true |

## Next Diagnostics

- toy2_basin_learned_credit_replay_floor: inspect time-to-ceiling trajectories and seed variance.
- toy4_basin_learned_credit_replay_floor: inspect time-to-ceiling trajectories and seed variance.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| toy2_basin_learned_credit_replay_floor | linear_welfare_heavy | baseline | false | diagnostic_only | 3 | 23.6667 | 3 | not in main claim group |
| toy2_basin_learned_credit_replay_floor | mixed_objective_basin_escalate_credit_p3_min2_h1_prototype | diagnostic | false | diagnostic_only | 3 | 9.33333 | 3 | not in main claim group |
| toy2_basin_learned_credit_replay_floor | learned_candidate_context_all_replay | diagnostic | false | diagnostic_only | 3 | 9.33333 | 3 | not in main claim group |
| toy2_basin_learned_credit_replay_floor | learned_candidate_context_confident_agreement_replay | diagnostic | false | diagnostic_only | 0 |  | 2.32833 | not in main claim group |
| toy2_basin_learned_credit_replay_floor | learned_candidate_context_confident_agreement_floor25_replay | diagnostic | false | diagnostic_only | 3 | 28.6667 | 3 | not in main claim group |
| toy2_basin_learned_credit_replay_floor | learned_candidate_context_confident_agreement_floor50_replay | nabm | true | fail | 3 | 16 | 3 | mean time-to-ceiling 16 >= 10 |
| toy4_basin_learned_credit_replay_floor | linear_welfare_heavy | baseline | false | diagnostic_only | 1 | 21 | 0.596 | not in main claim group |
| toy4_basin_learned_credit_replay_floor | mixed_objective_basin_escalate_credit_p3_min2_h1_prototype | diagnostic | false | diagnostic_only | 3 | 11.6667 | 0.6 | not in main claim group |
| toy4_basin_learned_credit_replay_floor | learned_candidate_context_all_replay | diagnostic | false | diagnostic_only | 3 | 11.6667 | 0.6 | not in main claim group |
| toy4_basin_learned_credit_replay_floor | learned_candidate_context_confident_agreement_replay | diagnostic | false | diagnostic_only | 0 |  | 0.298 | not in main claim group |
| toy4_basin_learned_credit_replay_floor | learned_candidate_context_confident_agreement_floor25_replay | diagnostic | false | diagnostic_only | 3 | 19.3333 | 0.6 | not in main claim group |
| toy4_basin_learned_credit_replay_floor | learned_candidate_context_confident_agreement_floor50_replay | nabm | true | fail | 3 | 14.6667 | 0.6 | mean time-to-ceiling 14.6667 >= 12 |
