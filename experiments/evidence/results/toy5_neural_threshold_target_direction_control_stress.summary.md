# Evidence Gate: toy5_neural_threshold_target_direction_control_stress

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_neural_threshold_target_direction_control_stress | toy5 | pass | neural_threshold_target_augmented_direction_control_w1p0 | 10/10 | 0 | 0 | 1 | 0 | false |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_neural_threshold_target_direction_control_stress | neural_threshold_target_no_seed_control | baseline | false | diagnostic_only | 10 | 0 | 0 | 1 | 0 | 1 | not in main claim group |
| toy5_neural_threshold_target_direction_control_stress | neural_threshold_target_non_directional_readiness_self_excitation | diagnostic | false | diagnostic_only | 0 | 0 | 10 | 0 | 0.05 | 0 | not in main claim group |
| toy5_neural_threshold_target_direction_control_stress | neural_threshold_target_local_direction_control | diagnostic | false | diagnostic_only | 10 | 0 | 0 | 1 | 0 | 1 | not in main claim group |
| toy5_neural_threshold_target_direction_control_stress | neural_threshold_target_augmented_direction_control_w1p0 | direction_control | true | pass | 10 | 0 | 0 | 1 | 0 | 1 |  |
