# Evidence Gate: toy5_neural_threshold_target_safety_frontier_combined

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_threshold_target_no_seed_safety | toy5 | pass | neural_threshold_target_no_seed_exposure_anchored_prior0p49 | 10/10 | 0 | 0 | 1 | 0 | false |
| toy5_threshold_target_seeded_frontier_spread | toy5 | pass | neural_threshold_target_frontier_exposure_anchored_w2p0 | 10/10 | 31.4 | 0 | 1 | 0.000205357 | true |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_threshold_target_no_seed_safety | neural_threshold_target_no_seed_output_average_prior0p49 | baseline | false | diagnostic_only | 10 | 0 | 0 | 1 | 0 | 1 | not in main claim group |
| toy5_threshold_target_no_seed_safety | neural_threshold_target_no_seed_non_directional_readiness_prior0p49 | diagnostic | false | diagnostic_only | 0 | 0 | 10 | 0 | 0.05 | 0 | not in main claim group |
| toy5_threshold_target_no_seed_safety | neural_threshold_target_no_seed_local_direction_prior0p49 | diagnostic | false | diagnostic_only | 10 | 0 | 0 | 1 | 0 | 1 | not in main claim group |
| toy5_threshold_target_no_seed_safety | neural_threshold_target_no_seed_augmented_direction_prior0p49 | diagnostic | false | diagnostic_only | 10 | 0 | 0 | 1 | 0 | 1 | not in main claim group |
| toy5_threshold_target_no_seed_safety | neural_threshold_target_no_seed_exposure_anchored_prior0p49 | directional_threshold_target | true | pass | 10 | 0 | 0 | 1 | 0 | 1 |  |
| toy5_threshold_target_seeded_frontier_spread | neural_threshold_target_frontier_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_target_seeded_frontier_spread | neural_threshold_target_frontier_readiness_propagation_w1p0 | diagnostic | false | diagnostic_only | 10 | 31.4 | 0 | 1 | 0.000205357 | 100 | not in main claim group |
| toy5_threshold_target_seeded_frontier_spread | neural_threshold_target_frontier_local_direction_w1p0 | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_target_seeded_frontier_spread | neural_threshold_target_frontier_readiness_augmented_w1p0 | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_target_seeded_frontier_spread | neural_threshold_target_frontier_exposure_anchored_w2p0 | directional_threshold_target | true | pass | 10 | 31.4 | 0 | 1 | 0.000205357 | 100 |  |
