# Evidence Gate: toy24_basin_credit_escalation_replay_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | --- |
| toy2_basin_credit | toy2 | pass | mixed_objective_basin_replay_all_p3_h1 | 3/3 | 9.33333 | false |
| toy4_basin_credit | toy4 | pass | mixed_objective_basin_escalate_credit_p3_min2_h1 | 3/3 | 11.6667 | false |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| toy2_basin_credit | reputation_imitation | baseline | false | diagnostic_only | 3 | 2.66667 | 3 | not in main claim group |
| toy2_basin_credit | linear_welfare_heavy | baseline | false | diagnostic_only | 3 | 23.6667 | 3 | not in main claim group |
| toy2_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | diagnostic | false | diagnostic_only | 3 | 22.6667 | 3 | uses teacher/bootstrap/replay path |
| toy2_basin_credit | basin_credit_w1p0_h1_prototype | nabm | true | fail | 0 |  | 2.55667 | final ceiling hits 0 < 3 |
| toy2_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 0 |  | 1.06 | not in main claim group |
| toy2_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | nabm | true | fail | 3 | 22.6667 | 3 | mean time-to-ceiling 22.6667 >= 10 |
| toy2_basin_credit | mixed_objective_basin_replay_all_p2_h1 | nabm | true | fail | 3 | 10.3333 | 3 | mean time-to-ceiling 10.3333 >= 10 |
| toy2_basin_credit | mixed_objective_basin_replay_all_p3_h1 | nabm | true | pass | 3 | 9.33333 | 3 |  |
| toy2_basin_credit | mixed_objective_basin_adaptive_score_p3_min2_h1 | nabm | true | pass | 3 | 9.33333 | 3 |  |
| toy2_basin_credit | mixed_objective_basin_escalate_credit_p3_min2_h1 | nabm | true | pass | 3 | 9.33333 | 3 |  |
| toy4_basin_credit | reputation_imitation | baseline | false | diagnostic_only | 3 | 2.66667 | 0.6 | not in main claim group |
| toy4_basin_credit | linear_welfare_heavy | baseline | false | diagnostic_only | 1 | 21 | 0.596 | not in main claim group |
| toy4_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | diagnostic | false | diagnostic_only | 1 | 19.6667 | 0.596 | uses teacher/bootstrap/replay path |
| toy4_basin_credit | basin_credit_w1p0_h1_prototype | nabm | true | fail | 0 |  | 0.314 | final ceiling hits 0 < 2 |
| toy4_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 0 |  | 0.004 | not in main claim group |
| toy4_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | nabm | true | fail | 2 | 16.3333 | 0.598 | mean time-to-ceiling 16.3333 >= 12 |
| toy4_basin_credit | mixed_objective_basin_replay_all_p2_h1 | nabm | true | fail | 3 | 13.6667 | 0.6 | mean time-to-ceiling 13.6667 >= 12 |
| toy4_basin_credit | mixed_objective_basin_replay_all_p3_h1 | nabm | true | pass | 2 | 11.6667 | 0.598 |  |
| toy4_basin_credit | mixed_objective_basin_adaptive_score_p3_min2_h1 | nabm | true | fail | 2 | 12.6667 | 0.598 | mean time-to-ceiling 12.6667 >= 12 |
| toy4_basin_credit | mixed_objective_basin_escalate_credit_p3_min2_h1 | nabm | true | pass | 3 | 11.6667 | 0.6 |  |
