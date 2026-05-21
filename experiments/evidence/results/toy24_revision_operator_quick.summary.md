# Evidence Gate: toy24_revision_operator_quick

Overall status: **fail**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_revision_operator | toy2 | fail | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | 2/3 | 19.3333 | 1 | 0.866667 | 0.0097805 | false |
| toy4_revision_operator | toy4 | fail | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | 1/3 | 19 | 2 | 0.8 | 0.00975806 | false |

## Next Diagnostics

- toy2_revision_operator: separate final-epoch stochastic misses from true post-ceiling instability before adding new revision bias.
- toy4_revision_operator: separate final-epoch stochastic misses from true post-ceiling instability before adding new revision bias.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_revision_operator | reputation_imitation | baseline | false | diagnostic_only | 3 | 2.66667 | 0 | 1 | 0 | 3 | not in main claim group |
| toy2_revision_operator | linear_welfare_heavy | baseline | false | diagnostic_only | 3 | 23.6667 | 0 | 0.866667 | 0.00728836 | 3 | not in main claim group |
| toy2_revision_operator | mixed_objective_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 3 | 22.6667 | 0 | 0.866667 | 0.00769841 | 3 | not in main claim group |
| toy2_revision_operator | revision_operator_linear_welfare_heavy | diagnostic | false | diagnostic_only | 2 | 19.6667 | 1 | 0.933333 | 0.0100115 | 2.99667 | not in main claim group |
| toy2_revision_operator | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | revision | true | fail | 2 | 19.3333 | 1 | 0.866667 | 0.0097805 | 2.99667 | final ceiling hits 2 < 3 |
| toy4_revision_operator | reputation_imitation | baseline | false | diagnostic_only | 3 | 2.66667 | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_revision_operator | linear_welfare_heavy | baseline | false | diagnostic_only | 1 | 21 | 2 | 0.733333 | 0.00984633 | 0.596 | not in main claim group |
| toy4_revision_operator | mixed_objective_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 2 | 16.3333 | 1 | 0.866667 | 0.0101694 | 0.598 | not in main claim group |
| toy4_revision_operator | revision_operator_linear_welfare_heavy | diagnostic | false | diagnostic_only | 1 | 19.3333 | 2 | 0.733333 | 0.00971902 | 0.596 | not in main claim group |
| toy4_revision_operator | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | revision | true | fail | 1 | 19 | 2 | 0.8 | 0.00975806 | 0.596 | final ceiling hits 1 < 2 |
