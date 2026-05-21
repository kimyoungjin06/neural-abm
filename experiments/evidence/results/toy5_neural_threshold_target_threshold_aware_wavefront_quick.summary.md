# Evidence Gate: toy5_neural_threshold_target_threshold_aware_wavefront_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_threshold_aware_wavefront_no_seed_heterogeneous_safety | toy5 | pass | neural_threshold_aware_wavefront_no_seed_threshold_anchor | 5/5 | 0 | 0 | 1 | 0 | false |
| toy5_threshold_aware_lattice_k4_heterogeneous_h0p85_spread | toy5 | pass | neural_threshold_aware_lattice_k4_h0p85_threshold_anchor | 5/5 | 36.2 | 0 | 1 | 0.00273626 | true |
| toy5_threshold_aware_lattice_k6_heterogeneous_h0p95_spread | toy5 | pass | neural_threshold_aware_lattice_k6_h0p95_threshold_anchor | 5/5 | 25 | 0 | 1 | 0.00192 | true |
| toy5_threshold_aware_rewired_p0p10_heterogeneous_h0p95_spread | toy5 | pass | neural_threshold_aware_rewired_p0p10_h0p95_threshold_anchor | 5/5 | 10 | 0 | 1 | 0.000246404 | true |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_threshold_aware_wavefront_no_seed_heterogeneous_safety | neural_threshold_aware_wavefront_no_seed_output_average | baseline | false | diagnostic_only | 5 | 0 | 0 | 1 | 0 | 1 | not in main claim group |
| toy5_threshold_aware_wavefront_no_seed_heterogeneous_safety | neural_threshold_aware_wavefront_no_seed_non_directional | diagnostic | false | diagnostic_only | 0 | 0 | 5 | 0 | 0.05 | 0 | not in main claim group |
| toy5_threshold_aware_wavefront_no_seed_heterogeneous_safety | neural_threshold_aware_wavefront_no_seed_threshold_anchor | directional_threshold_target_threshold_aware_wavefront | true | pass | 5 | 0 | 0 | 1 | 0 | 1 |  |
| toy5_threshold_aware_lattice_k4_heterogeneous_h0p85_spread | neural_threshold_aware_lattice_k4_h0p85_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_aware_lattice_k4_heterogeneous_h0p85_spread | neural_threshold_aware_lattice_k4_h0p85_exposure_anchor | diagnostic | false | diagnostic_only | 5 | 33 | 0 | 1 | 0.00176471 | 100 | not in main claim group |
| toy5_threshold_aware_lattice_k4_heterogeneous_h0p85_spread | neural_threshold_aware_lattice_k4_h0p85_threshold_anchor | directional_threshold_target_threshold_aware_wavefront | true | pass | 5 | 36.2 | 0 | 1 | 0.00273626 | 100 |  |
| toy5_threshold_aware_lattice_k6_heterogeneous_h0p95_spread | neural_threshold_aware_lattice_k6_h0p95_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_aware_lattice_k6_heterogeneous_h0p95_spread | neural_threshold_aware_lattice_k6_h0p95_exposure_anchor | diagnostic | false | diagnostic_only | 5 | 23 | 0 | 1 | 0.00111111 | 100 | not in main claim group |
| toy5_threshold_aware_lattice_k6_heterogeneous_h0p95_spread | neural_threshold_aware_lattice_k6_h0p95_threshold_anchor | directional_threshold_target_threshold_aware_wavefront | true | pass | 5 | 25 | 0 | 1 | 0.00192 | 100 |  |
| toy5_threshold_aware_rewired_p0p10_heterogeneous_h0p95_spread | neural_threshold_aware_rewired_p0p10_h0p95_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 1 | not in main claim group |
| toy5_threshold_aware_rewired_p0p10_heterogeneous_h0p95_spread | neural_threshold_aware_rewired_p0p10_h0p95_exposure_anchor | diagnostic | false | diagnostic_only | 5 | 9.4 | 0 | 1 | 0 | 100 | not in main claim group |
| toy5_threshold_aware_rewired_p0p10_heterogeneous_h0p95_spread | neural_threshold_aware_rewired_p0p10_h0p95_threshold_anchor | directional_threshold_target_threshold_aware_wavefront | true | pass | 5 | 10 | 0 | 1 | 0.000246404 | 100 |  |
