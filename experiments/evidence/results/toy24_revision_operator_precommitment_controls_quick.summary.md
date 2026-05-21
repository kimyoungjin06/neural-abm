# Evidence Gate: toy24_revision_operator_precommitment_controls_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Trajectory | Failure Mode | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| toy2_revision_operator_precommitment_controls | toy2 | pass | revision_operator_precommitment_peer_evidence_w1p0 | 3/3 | 9.66667 | success |  | 0 | 1 | 0 | false |
| toy4_revision_operator_precommitment_controls | toy4 | pass | revision_operator_precommitment_peer_evidence_w1p0 | 3/3 | 9.33333 | success |  | 0 | 1 | 0 | false |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Trajectory | Failure Mode | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy2_revision_operator_precommitment_controls | reputation_imitation | baseline | false | diagnostic_only | 3 | 2.66667 | diagnostic | not_main_group | 0 | 1 | 0 | 3 | not in main claim group |
| toy2_revision_operator_precommitment_controls | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 2 | 19.3333 | diagnostic | not_main_group | 1 | 0.866667 | 0.0097805 | 2.99667 | not in main claim group |
| toy2_revision_operator_precommitment_controls | revision_operator_commitment_hysteresis | diagnostic | false | diagnostic_only | 3 | 15 | diagnostic | not_main_group | 0 | 1 | 0.000555556 | 3 | not in main claim group |
| toy2_revision_operator_precommitment_controls | revision_operator_precommitment_evidence | precommitment_control | true | fail | 3 | 11 | trajectory_success_slow_ttc | slow_time_to_ceiling | 0 | 1 | 0.00017094 | 3 | mean time-to-ceiling 11 >= 10 |
| toy2_revision_operator_precommitment_controls | revision_operator_precommitment_peer_evidence_w0p25 | precommitment_control | true | fail | 3 | 11 | trajectory_success_slow_ttc | slow_time_to_ceiling | 0 | 1 | 0 | 3 | mean time-to-ceiling 11 >= 10 |
| toy2_revision_operator_precommitment_controls | revision_operator_precommitment_peer_evidence_w0p5 | precommitment_control | true | fail | 3 | 10.3333 | trajectory_success_slow_ttc | slow_time_to_ceiling | 0 | 1 | 0.000325203 | 3 | mean time-to-ceiling 10.3333 >= 10 |
| toy2_revision_operator_precommitment_controls | revision_operator_precommitment_peer_evidence_w1p0 | precommitment_control | true | pass | 3 | 9.66667 | success |  | 0 | 1 | 0 | 3 |  |
| toy2_revision_operator_precommitment_controls | revision_operator_precommitment_commitment_hysteresis | precommitment_control | true | fail | 3 | 11 | trajectory_success_slow_ttc | slow_time_to_ceiling | 0 | 1 | 0.00017094 | 3 | mean time-to-ceiling 11 >= 10 |
| toy4_revision_operator_precommitment_controls | reputation_imitation | baseline | false | diagnostic_only | 3 | 2.66667 | diagnostic | not_main_group | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_revision_operator_precommitment_controls | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 1 | 19 | diagnostic | not_main_group | 2 | 0.8 | 0.00975806 | 0.596 | not in main claim group |
| toy4_revision_operator_precommitment_controls | revision_operator_commitment_hysteresis | diagnostic | false | diagnostic_only | 3 | 13.6667 | diagnostic | not_main_group | 0 | 0.933333 | 0.00147648 | 0.6 | not in main claim group |
| toy4_revision_operator_precommitment_controls | revision_operator_precommitment_evidence | precommitment_control | true | pass | 3 | 11 | success |  | 0 | 1 | 0.000854701 | 0.6 |  |
| toy4_revision_operator_precommitment_controls | revision_operator_precommitment_peer_evidence_w0p25 | precommitment_control | true | pass | 3 | 11 | success |  | 0 | 1 | 0.000854701 | 0.6 |  |
| toy4_revision_operator_precommitment_controls | revision_operator_precommitment_peer_evidence_w0p5 | precommitment_control | true | pass | 3 | 10.6667 | success |  | 0 | 1 | 0 | 0.6 |  |
| toy4_revision_operator_precommitment_controls | revision_operator_precommitment_peer_evidence_w1p0 | precommitment_control | true | pass | 3 | 9.33333 | success |  | 0 | 1 | 0 | 0.6 |  |
| toy4_revision_operator_precommitment_controls | revision_operator_precommitment_commitment_hysteresis | precommitment_control | true | pass | 3 | 11 | success |  | 0 | 1 | 0.000854701 | 0.6 |  |
