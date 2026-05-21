# Evidence Gate: toy5_neural_threshold_target_structural_robustness_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_threshold_target_no_seed_heterogeneous_safety | toy5 | pass | neural_threshold_target_no_seed_heterogeneous_exposure_anchor | 5/5 | 0 | 0 | 1 | 0 | false |
| toy5_threshold_target_random_seed_frontier_spread | toy5 | pass | neural_threshold_target_random_seed_frontier_exposure_anchor | 5/5 | 32.6 | 0 | 1 | 0.000363636 | true |
| toy5_threshold_target_heterogeneous_frontier_spread | toy5 | pass | neural_threshold_target_heterogeneous_frontier_exposure_anchor | 5/5 | 31 | 0 | 1 | 0.000410714 | true |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_threshold_target_no_seed_heterogeneous_safety | neural_threshold_target_no_seed_heterogeneous_output_average | baseline | false | diagnostic_only | 5 | 0 | 0 | 1 | 0 | 1 | not in main claim group |
| toy5_threshold_target_no_seed_heterogeneous_safety | neural_threshold_target_no_seed_heterogeneous_non_directional | diagnostic | false | diagnostic_only | 0 | 0 | 5 | 0 | 0.05 | 0 | not in main claim group |
| toy5_threshold_target_no_seed_heterogeneous_safety | neural_threshold_target_no_seed_heterogeneous_exposure_anchor | directional_threshold_target_robust | true | pass | 5 | 0 | 0 | 1 | 0 | 1 |  |
| toy5_threshold_target_random_seed_frontier_spread | neural_threshold_target_random_seed_frontier_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_target_random_seed_frontier_spread | neural_threshold_target_random_seed_frontier_readiness_propagation | diagnostic | false | diagnostic_only | 5 | 32.6 | 0 | 1 | 0.000363636 | 100 | not in main claim group |
| toy5_threshold_target_random_seed_frontier_spread | neural_threshold_target_random_seed_frontier_exposure_anchor | directional_threshold_target_robust | true | pass | 5 | 32.6 | 0 | 1 | 0.000363636 | 100 |  |
| toy5_threshold_target_heterogeneous_frontier_spread | neural_threshold_target_heterogeneous_frontier_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_target_heterogeneous_frontier_spread | neural_threshold_target_heterogeneous_frontier_readiness_propagation | diagnostic | false | diagnostic_only | 5 | 31 | 0 | 1 | 0.000410714 | 100 | not in main claim group |
| toy5_threshold_target_heterogeneous_frontier_spread | neural_threshold_target_heterogeneous_frontier_exposure_anchor | directional_threshold_target_robust | true | pass | 5 | 31 | 0 | 1 | 0.000410714 | 100 |  |
