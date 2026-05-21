# Evidence Gate: toy5_readiness_propagation_hard_argmax_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_readiness_propagation_hard_argmax | toy5 | pass | neural_argmax_readiness_propagation_w1p0 | 3/3 | 4.33333 | 0 | 1 | 0.000897436 | true |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_readiness_propagation_hard_argmax | neural_argmax_output_average | baseline | false | diagnostic_only | 0 |  | 0 | 0 |  | 74.6667 | not in main claim group |
| toy5_readiness_propagation_hard_argmax | neural_argmax_precommitment_evidence | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | 75 | not in main claim group |
| toy5_readiness_propagation_hard_argmax | neural_argmax_readiness_propagation_w1p0 | readiness_hard | true | pass | 3 | 4.33333 | 0 | 1 | 0.000897436 | 100 |  |
| toy5_readiness_propagation_hard_argmax | neural_argmax_readiness_direction_gated_w1p0 | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | 74.6667 | not in main claim group |
