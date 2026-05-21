# Evidence Gate: toy24_revision_operator_controls_quick

Overall status: **fail**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_revision_operator_controls | toy2 | fail | revision_operator_commitment_hysteresis | 3/3 | 15 | 0 | 1 | 0.000555556 | false |
| toy4_revision_operator_controls | toy4 | fail | revision_operator_commitment_hysteresis | 3/3 | 13.6667 | 0 | 0.933333 | 0.00147648 | false |

## Next Diagnostics

- toy2_revision_operator_controls: inspect time-to-ceiling trajectories and seed variance.
- toy4_revision_operator_controls: inspect time-to-ceiling trajectories and seed variance.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_revision_operator_controls | reputation_imitation | baseline | false | diagnostic_only | 3 | 2.66667 | 0 | 1 | 0 | 3 | not in main claim group |
| toy2_revision_operator_controls | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 2 | 19.3333 | 1 | 0.866667 | 0.0097805 | 2.99667 | not in main claim group |
| toy2_revision_operator_controls | revision_operator_commitment_hysteresis | control | true | fail | 3 | 15 | 0 | 1 | 0.000555556 | 3 | mean time-to-ceiling 15 >= 10 |
| toy2_revision_operator_controls | revision_operator_terminal_argmax_k1 | control | true | fail | 3 | 19.3333 | 0 | 0.933333 | 0.00967297 | 3 | mean time-to-ceiling 19.3333 >= 10 |
| toy2_revision_operator_controls | revision_operator_terminal_argmax_k5 | control | true | fail | 3 | 19.3333 | 0 | 1 | 0.00947689 | 3 | mean time-to-ceiling 19.3333 >= 10 |
| toy4_revision_operator_controls | reputation_imitation | baseline | false | diagnostic_only | 3 | 2.66667 | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_revision_operator_controls | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 1 | 19 | 2 | 0.8 | 0.00975806 | 0.596 | not in main claim group |
| toy4_revision_operator_controls | revision_operator_commitment_hysteresis | control | true | fail | 3 | 13.6667 | 0 | 0.933333 | 0.00147648 | 0.6 | mean time-to-ceiling 13.6667 >= 12 |
| toy4_revision_operator_controls | revision_operator_terminal_argmax_k1 | control | true | fail | 3 | 19 | 0 | 0.933333 | 0.00953149 | 0.6 | mean time-to-ceiling 19 >= 12 |
| toy4_revision_operator_controls | revision_operator_terminal_argmax_k5 | control | true | fail | 3 | 19 | 0 | 1 | 0.00929339 | 0.6 | mean time-to-ceiling 19 >= 12 |
