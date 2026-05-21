# Evidence Gate: toy5_readiness_propagation_holdout_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_readiness_propagation_holdout | toy5 | pass | neural_readiness_propagation_w1p0 | 3/3 | 1.33333 | 0 | 1 | 0.00743197 | false |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy5_readiness_propagation_holdout | complex_threshold | diagnostic | false | diagnostic_only | 1 | 15 | 0 | 0.333333 | 0.0134286 | 34 | not in main claim group |
| toy5_readiness_propagation_holdout | neural_output_average | baseline | false | diagnostic_only | 3 | 1.33333 | 0 | 1 | 0.00791808 | 100 | not in main claim group |
| toy5_readiness_propagation_holdout | neural_precommitment_evidence | diagnostic | false | diagnostic_only | 3 | 1.33333 | 0 | 1 | 0.00743197 | 100 | not in main claim group |
| toy5_readiness_propagation_holdout | neural_readiness_propagation_w1p0 | readiness_holdout | true | pass | 3 | 1.33333 | 0 | 1 | 0.00743197 | 100 |  |
