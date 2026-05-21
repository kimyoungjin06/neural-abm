# Evidence Gate: toy24_basin_transition_dataset_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | --- |
| toy2_basin_dataset | toy2 | pass | mixed_objective_basin_escalate_credit_p3_min2_h1 | 3/3 | 9.33333 | true |
| toy4_basin_dataset | toy4 | pass | mixed_objective_basin_escalate_credit_p3_min2_h1 | 3/3 | 11.6667 | true |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| toy2_basin_dataset | linear_welfare_heavy | baseline | false | diagnostic_only | 3 | 23.6667 | 3 | not in main claim group |
| toy2_basin_dataset | mixed_objective_basin_escalate_credit_p3_min2_h1 | nabm | true | pass | 3 | 9.33333 | 3 |  |
| toy4_basin_dataset | linear_welfare_heavy | baseline | false | diagnostic_only | 1 | 21 | 0.596 | not in main claim group |
| toy4_basin_dataset | mixed_objective_basin_escalate_credit_p3_min2_h1 | nabm | true | pass | 3 | 11.6667 | 0.6 |  |
