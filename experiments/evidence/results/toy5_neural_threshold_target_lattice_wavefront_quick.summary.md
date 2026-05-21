# Evidence Gate: toy5_neural_threshold_target_lattice_wavefront_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_threshold_target_wavefront_no_seed_safety | toy5 | pass | neural_threshold_target_wavefront_no_seed_exposure_anchor | 5/5 | 0 | 0 | 1 | 0 | false |
| toy5_threshold_target_lattice_wavefront_spread | toy5 | pass | neural_threshold_target_lattice_max_wavefront_anchor | 5/5 | 23 | 0 | 1 | 0.00111111 | true |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_threshold_target_wavefront_no_seed_safety | neural_threshold_target_wavefront_no_seed_output_average | baseline | false | diagnostic_only | 5 | 0 | 0 | 1 | 0 | 1 | not in main claim group |
| toy5_threshold_target_wavefront_no_seed_safety | neural_threshold_target_wavefront_no_seed_non_directional | diagnostic | false | diagnostic_only | 0 | 0 | 5 | 0 | 0.05 | 0 | not in main claim group |
| toy5_threshold_target_wavefront_no_seed_safety | neural_threshold_target_wavefront_no_seed_exposure_anchor | directional_threshold_target_wavefront | true | pass | 5 | 0 | 0 | 1 | 0 | 1 |  |
| toy5_threshold_target_lattice_wavefront_spread | neural_threshold_target_lattice_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_target_lattice_wavefront_spread | neural_threshold_target_lattice_mean_exposure_anchor | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | 65 | not in main claim group |
| toy5_threshold_target_lattice_wavefront_spread | neural_threshold_target_lattice_max_wavefront_anchor | directional_threshold_target_wavefront | true | pass | 5 | 23 | 0 | 1 | 0.00111111 | 100 |  |
