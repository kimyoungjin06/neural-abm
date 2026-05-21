# Evidence Gate: toy24_basin_credit_quick

Overall status: **fail**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | --- |
| toy2_basin_credit | toy2 | fail | basin_credit_w1p0_h1_prototype | 0/3 |  | false |
| toy4_basin_credit | toy4 | fail | basin_credit_w1p0_h1_prototype | 0/3 |  | false |

## Next Diagnostics

- toy2_basin_credit: inspect aggregate trajectories before adding a contrastive critic.
- toy4_basin_credit: inspect aggregate trajectories before adding a contrastive critic.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| toy2_basin_credit | reputation_imitation | baseline | false | diagnostic_only | 3 | 2.66667 | 3 | not in main claim group |
| toy2_basin_credit | linear_welfare_heavy | baseline | false | diagnostic_only | 3 | 23.6667 | 3 | not in main claim group |
| toy2_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | diagnostic | false | diagnostic_only | 3 | 22.6667 | 3 | uses teacher/bootstrap/replay path |
| toy2_basin_credit | basin_credit_w1p0_h1_prototype | nabm | true | fail | 0 |  | 2.55667 | final ceiling hits 0 < 3 |
| toy2_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | nabm | true | fail | 0 |  | 1.06 | final ceiling hits 0 < 3 |
| toy4_basin_credit | reputation_imitation | baseline | false | diagnostic_only | 3 | 2.66667 | 0.6 | not in main claim group |
| toy4_basin_credit | linear_welfare_heavy | baseline | false | diagnostic_only | 1 | 21 | 0.596 | not in main claim group |
| toy4_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | diagnostic | false | diagnostic_only | 1 | 19.6667 | 0.596 | uses teacher/bootstrap/replay path |
| toy4_basin_credit | basin_credit_w1p0_h1_prototype | nabm | true | fail | 0 |  | 0.314 | final ceiling hits 0 < 2 |
| toy4_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | nabm | true | fail | 0 |  | 0.004 | final ceiling hits 0 < 2 |
