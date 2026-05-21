# Evidence Gate: toy5_neural_threshold_target_wavefront_stress_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_threshold_target_wavefront_stress_no_seed_heterogeneous_safety | toy5 | pass | neural_threshold_target_stress_no_seed_heterogeneous_exposure_anchor | 5/5 | 0 | 0 | 1 | 0 | false |
| toy5_threshold_target_lattice_k4_heterogeneous_h0p85_wavefront_spread | toy5 | pass | neural_threshold_target_lattice_k4_heterogeneous_h0p85_max_wavefront_anchor | 5/5 | 33 | 0 | 1 | 0.00176471 | true |
| toy5_threshold_target_lattice_k6_heterogeneous_h0p95_wavefront_spread | toy5 | pass | neural_threshold_target_lattice_k6_heterogeneous_h0p95_max_wavefront_anchor | 5/5 | 23 | 0 | 1 | 0.00111111 | true |
| toy5_threshold_target_rewired_p0p10_heterogeneous_h0p95_wavefront_spread | toy5 | pass | neural_threshold_target_rewired_p0p10_heterogeneous_h0p95_max_wavefront_anchor | 5/5 | 9.4 | 0 | 1 | 0 | true |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_threshold_target_wavefront_stress_no_seed_heterogeneous_safety | neural_threshold_target_stress_no_seed_heterogeneous_output_average | baseline | false | diagnostic_only | 5 | 0 | 0 | 1 | 0 | 1 | not in main claim group |
| toy5_threshold_target_wavefront_stress_no_seed_heterogeneous_safety | neural_threshold_target_stress_no_seed_heterogeneous_non_directional | diagnostic | false | diagnostic_only | 0 | 0 | 5 | 0 | 0.05 | 0 | not in main claim group |
| toy5_threshold_target_wavefront_stress_no_seed_heterogeneous_safety | neural_threshold_target_stress_no_seed_heterogeneous_exposure_anchor | directional_threshold_target_wavefront_stress | true | pass | 5 | 0 | 0 | 1 | 0 | 1 |  |
| toy5_threshold_target_lattice_k4_heterogeneous_h0p85_wavefront_spread | neural_threshold_target_lattice_k4_heterogeneous_h0p85_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_target_lattice_k4_heterogeneous_h0p85_wavefront_spread | neural_threshold_target_lattice_k4_heterogeneous_h0p85_mean_exposure_anchor | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | 49 | not in main claim group |
| toy5_threshold_target_lattice_k4_heterogeneous_h0p85_wavefront_spread | neural_threshold_target_lattice_k4_heterogeneous_h0p85_max_wavefront_anchor | directional_threshold_target_wavefront_stress | true | pass | 5 | 33 | 0 | 1 | 0.00176471 | 100 |  |
| toy5_threshold_target_lattice_k6_heterogeneous_h0p95_wavefront_spread | neural_threshold_target_lattice_k6_heterogeneous_h0p95_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_target_lattice_k6_heterogeneous_h0p95_wavefront_spread | neural_threshold_target_lattice_k6_heterogeneous_h0p95_mean_exposure_anchor | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | 65 | not in main claim group |
| toy5_threshold_target_lattice_k6_heterogeneous_h0p95_wavefront_spread | neural_threshold_target_lattice_k6_heterogeneous_h0p95_max_wavefront_anchor | directional_threshold_target_wavefront_stress | true | pass | 5 | 23 | 0 | 1 | 0.00111111 | 100 |  |
| toy5_threshold_target_rewired_p0p10_heterogeneous_h0p95_wavefront_spread | neural_threshold_target_rewired_p0p10_heterogeneous_h0p95_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_target_rewired_p0p10_heterogeneous_h0p95_wavefront_spread | neural_threshold_target_rewired_p0p10_heterogeneous_h0p95_mean_exposure_anchor | diagnostic | false | diagnostic_only | 5 | 31 | 0 | 1 | 0.000410714 | 100 | not in main claim group |
| toy5_threshold_target_rewired_p0p10_heterogeneous_h0p95_wavefront_spread | neural_threshold_target_rewired_p0p10_heterogeneous_h0p95_max_wavefront_anchor | directional_threshold_target_wavefront_stress | true | pass | 5 | 9.4 | 0 | 1 | 0 | 100 |  |
