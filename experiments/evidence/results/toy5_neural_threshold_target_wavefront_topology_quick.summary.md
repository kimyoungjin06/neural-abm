# Evidence Gate: toy5_neural_threshold_target_wavefront_topology_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_threshold_target_wavefront_topology_no_seed_safety | toy5 | pass | neural_threshold_target_topology_no_seed_exposure_anchor | 5/5 | 0 | 0 | 1 | 0 | false |
| toy5_threshold_target_lattice_k4_wavefront_spread | toy5 | pass | neural_threshold_target_lattice_k4_max_wavefront_anchor | 5/5 | 33 | 0 | 1 | 0.00176471 | true |
| toy5_threshold_target_lattice_k8_wavefront_spread | toy5 | pass | neural_threshold_target_lattice_k8_max_wavefront_anchor | 5/5 | 18 | 0 | 1 | 0.0009375 | true |
| toy5_threshold_target_rewired_p0p02_wavefront_spread | toy5 | pass | neural_threshold_target_rewired_p0p02_max_wavefront_anchor | 5/5 | 14.6 | 0 | 1 | 0.000681664 | true |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_threshold_target_wavefront_topology_no_seed_safety | neural_threshold_target_topology_no_seed_output_average | baseline | false | diagnostic_only | 5 | 0 | 0 | 1 | 0 | 1 | not in main claim group |
| toy5_threshold_target_wavefront_topology_no_seed_safety | neural_threshold_target_topology_no_seed_non_directional | diagnostic | false | diagnostic_only | 0 | 0 | 5 | 0 | 0.05 | 0 | not in main claim group |
| toy5_threshold_target_wavefront_topology_no_seed_safety | neural_threshold_target_topology_no_seed_exposure_anchor | directional_threshold_target_wavefront_topology | true | pass | 5 | 0 | 0 | 1 | 0 | 1 |  |
| toy5_threshold_target_lattice_k4_wavefront_spread | neural_threshold_target_lattice_k4_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_target_lattice_k4_wavefront_spread | neural_threshold_target_lattice_k4_mean_exposure_anchor | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | 49 | not in main claim group |
| toy5_threshold_target_lattice_k4_wavefront_spread | neural_threshold_target_lattice_k4_max_wavefront_anchor | directional_threshold_target_wavefront_topology | true | pass | 5 | 33 | 0 | 1 | 0.00176471 | 100 |  |
| toy5_threshold_target_lattice_k8_wavefront_spread | neural_threshold_target_lattice_k8_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_target_lattice_k8_wavefront_spread | neural_threshold_target_lattice_k8_mean_exposure_anchor | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | 83 | not in main claim group |
| toy5_threshold_target_lattice_k8_wavefront_spread | neural_threshold_target_lattice_k8_max_wavefront_anchor | directional_threshold_target_wavefront_topology | true | pass | 5 | 18 | 0 | 1 | 0.0009375 | 100 |  |
| toy5_threshold_target_rewired_p0p02_wavefront_spread | neural_threshold_target_rewired_p0p02_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_target_rewired_p0p02_wavefront_spread | neural_threshold_target_rewired_p0p02_mean_exposure_anchor | diagnostic | false | diagnostic_only | 4 | 46 | 0 | 0.48 | 0.0015 | 95.6 | not in main claim group |
| toy5_threshold_target_rewired_p0p02_wavefront_spread | neural_threshold_target_rewired_p0p02_max_wavefront_anchor | directional_threshold_target_wavefront_topology | true | pass | 5 | 14.6 | 0 | 1 | 0.000681664 | 100 |  |
