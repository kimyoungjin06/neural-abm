# Evidence Gate: toy5_readiness_augmented_direction_frontier_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_readiness_augmented_direction_frontier | toy5 | pass | neural_argmax_readiness_augmented_direction_w1p0 | 3/3 | 9 | 0 | 1 | 0.00159812 | true |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_readiness_augmented_direction_frontier | neural_argmax_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 88.3333 | not in main claim group |
| toy5_readiness_augmented_direction_frontier | neural_argmax_precommitment_evidence | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | 89 | not in main claim group |
| toy5_readiness_augmented_direction_frontier | neural_argmax_readiness_propagation_w1p0 | diagnostic | false | diagnostic_only | 3 | 5.66667 | 0 | 1 | 0.00122222 | 100 | not in main claim group |
| toy5_readiness_augmented_direction_frontier | neural_argmax_local_threshold_direction_w1p0 | diagnostic | false | diagnostic_only | 1 | 11 | 0 | 0.333333 | 0 | 92.6667 | not in main claim group |
| toy5_readiness_augmented_direction_frontier | neural_argmax_readiness_augmented_direction_w1p0 | readiness_frontier | true | pass | 3 | 9 | 0 | 1 | 0.00159812 | 100 |  |
