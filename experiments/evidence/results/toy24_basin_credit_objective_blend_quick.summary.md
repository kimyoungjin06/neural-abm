# Evidence Gate: toy24_basin_credit_objective_blend_quick

Overall status: **fail**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_basin_credit | toy2 | fail | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | 3/3 | 12 | 0 |  |  | false |
| toy4_basin_credit | toy4 | pass | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | 3/3 | 11 | 0 |  |  | false |

## Next Diagnostics

- toy2_basin_credit: inspect time-to-ceiling trajectories and seed variance.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_basin_credit | reputation_imitation | baseline | false | diagnostic_only | 3 | 2.66667 | 0 |  |  | 3 | not in main claim group |
| toy2_basin_credit | linear_welfare_heavy | baseline | false | diagnostic_only | 3 | 23.6667 | 0 |  |  | 3 | not in main claim group |
| toy2_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | diagnostic | false | diagnostic_only | 3 | 22.6667 | 0 |  |  | 3 | uses teacher/bootstrap/replay path |
| toy2_basin_credit | basin_credit_w1p0_h1_prototype | nabm | true | fail | 0 |  | 0 |  |  | 2.55667 | final ceiling hits 0 < 3 |
| toy2_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 0 |  | 0 |  |  | 1.06 | not in main claim group |
| toy2_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | nabm | true | fail | 3 | 22.6667 | 0 |  |  | 3 | mean time-to-ceiling 22.6667 >= 10 |
| toy2_basin_credit | mixed_objective_basin_confidence_social_w0p5_0p5_h1 | nabm | true | fail | 3 | 18.3333 | 0 |  |  | 3 | mean time-to-ceiling 18.3333 >= 10 |
| toy2_basin_credit | mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1 | nabm | true | fail | 3 | 21.3333 | 0 |  |  | 3 | mean time-to-ceiling 21.3333 >= 10 |
| toy2_basin_credit | mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1 | nabm | true | fail | 3 | 18.3333 | 0 |  |  | 3 | mean time-to-ceiling 18.3333 >= 10 |
| toy2_basin_credit | mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1 | nabm | true | fail | 3 | 16 | 0 |  |  | 3 | mean time-to-ceiling 16 >= 10 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | nabm | true | fail | 3 | 12 | 0 |  |  | 3 | mean time-to-ceiling 12 >= 10 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1 | nabm | true | fail | 3 | 12 | 0 |  |  | 3 | mean time-to-ceiling 12 >= 10 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | nabm | true | fail | 3 | 12 | 0 |  |  | 3 | mean time-to-ceiling 12 >= 10 |
| toy2_basin_credit | mixed_objective_basin_directional_social_w0p5_0p5_h1 | nabm | true | fail | 3 | 18.3333 | 0 |  |  | 3 | mean time-to-ceiling 18.3333 >= 10 |
| toy4_basin_credit | reputation_imitation | baseline | false | diagnostic_only | 3 | 2.66667 | 0 |  |  | 0.6 | not in main claim group |
| toy4_basin_credit | linear_welfare_heavy | baseline | false | diagnostic_only | 1 | 21 | 0 |  |  | 0.596 | not in main claim group |
| toy4_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | diagnostic | false | diagnostic_only | 1 | 19.6667 | 0 |  |  | 0.596 | uses teacher/bootstrap/replay path |
| toy4_basin_credit | basin_credit_w1p0_h1_prototype | nabm | true | fail | 0 |  | 0 |  |  | 0.314 | final ceiling hits 0 < 2 |
| toy4_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 0 |  | 0 |  |  | 0.004 | not in main claim group |
| toy4_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | nabm | true | fail | 2 | 16.3333 | 0 |  |  | 0.598 | mean time-to-ceiling 16.3333 >= 12 |
| toy4_basin_credit | mixed_objective_basin_confidence_social_w0p5_0p5_h1 | nabm | true | fail | 1 | 18 | 0 |  |  | 0.596 | final ceiling hits 1 < 2 |
| toy4_basin_credit | mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1 | nabm | true | fail | 2 | 18 | 0 |  |  | 0.598 | mean time-to-ceiling 18 >= 12 |
| toy4_basin_credit | mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1 | nabm | true | fail | 1 | 18 | 0 |  |  | 0.596 | final ceiling hits 1 < 2 |
| toy4_basin_credit | mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1 | nabm | true | fail | 3 | 13.6667 | 0 |  |  | 0.6 | mean time-to-ceiling 13.6667 >= 12 |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | nabm | true | pass | 3 | 11.3333 | 0 |  |  | 0.6 |  |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1 | nabm | true | pass | 3 | 11.3333 | 0 |  |  | 0.6 |  |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | nabm | true | pass | 3 | 11 | 0 |  |  | 0.6 |  |
| toy4_basin_credit | mixed_objective_basin_directional_social_w0p5_0p5_h1 | nabm | true | fail | 1 | 18 | 0 |  |  | 0.596 | final ceiling hits 1 < 2 |
