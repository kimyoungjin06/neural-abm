# Evidence Gate: toy24_basin_learned_credit_replay_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | --- |
| toy2_basin_learned_credit | toy2 | pass | mixed_objective_basin_escalate_credit_p3_min2_h1_learned_gated_prototype | 3/3 | 9.33333 | true |
| toy4_basin_learned_credit | toy4 | pass | mixed_objective_basin_escalate_credit_p3_min2_h1_learned_gated_prototype | 3/3 | 11.6667 | true |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| toy2_basin_learned_credit | linear_welfare_heavy | baseline | false | diagnostic_only | 3 | 23.6667 | 3 | not in main claim group |
| toy2_basin_learned_credit | mixed_objective_basin_escalate_credit_p3_min2_h1_prototype | diagnostic | false | diagnostic_only | 3 | 9.33333 | 3 | not in main claim group |
| toy2_basin_learned_credit | mixed_objective_basin_escalate_credit_p3_min2_h1_learned_gated_prototype | nabm | true | pass | 3 | 9.33333 | 3 |  |
| toy2_basin_learned_credit | mixed_objective_basin_escalate_credit_p3_min2_h1_learned_gated_zero | diagnostic | false | diagnostic_only | 3 | 9.33333 | 3 | not in main claim group |
| toy4_basin_learned_credit | linear_welfare_heavy | baseline | false | diagnostic_only | 1 | 21 | 0.596 | not in main claim group |
| toy4_basin_learned_credit | mixed_objective_basin_escalate_credit_p3_min2_h1_prototype | diagnostic | false | diagnostic_only | 3 | 11.6667 | 0.6 | not in main claim group |
| toy4_basin_learned_credit | mixed_objective_basin_escalate_credit_p3_min2_h1_learned_gated_prototype | nabm | true | pass | 3 | 11.6667 | 0.6 |  |
| toy4_basin_learned_credit | mixed_objective_basin_escalate_credit_p3_min2_h1_learned_gated_zero | diagnostic | false | diagnostic_only | 3 | 11.6667 | 0.6 | not in main claim group |
