# Evidence Gate: toy5_neural_threshold_target_frontier_stability

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_neural_threshold_target_frontier_stability | toy5 | pass | neural_threshold_target_exposure_anchored_w2p0 | 10/10 | 31.4 | 0 | 1 | 0.000205357 | true |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_neural_threshold_target_frontier_stability | neural_threshold_target_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_neural_threshold_target_frontier_stability | neural_threshold_target_readiness_propagation_w1p0 | diagnostic | false | diagnostic_only | 10 | 31.4 | 0 | 1 | 0.000205357 | 100 | not in main claim group |
| toy5_neural_threshold_target_frontier_stability | neural_threshold_target_local_direction_w1p0 | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_neural_threshold_target_frontier_stability | neural_threshold_target_readiness_augmented_w1p0 | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_neural_threshold_target_frontier_stability | neural_threshold_target_exposure_anchored_w2p0 | threshold_frontier | true | pass | 10 | 31.4 | 0 | 1 | 0.000205357 | 100 |  |
