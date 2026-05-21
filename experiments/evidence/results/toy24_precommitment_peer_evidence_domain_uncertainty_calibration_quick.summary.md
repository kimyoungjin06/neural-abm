# Evidence Gate: toy24_precommitment_peer_evidence_domain_uncertainty_calibration_quick

Overall status: **fail**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_precommitment_peer_evidence_stag_hunt_calibration | toy2 | pass | revision_precommitment_peer_evidence_stag_hunt_p0p35 | 3/3 | 10 | 0 | 1 | 0 | false |
| toy4_precommitment_peer_evidence_resource_coupled_calibration | toy4 | fail | revision_precommitment_peer_evidence_resource_p0p35 | 0/3 |  | 0 | 0 |  | false |

## Next Diagnostics

- toy4_precommitment_peer_evidence_resource_coupled_calibration: inspect aggregate trajectories before adding a contrastive critic.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_precommitment_peer_evidence_stag_hunt_calibration | reputation_imitation_stag_hunt_p0p35 | baseline | false | diagnostic_only | 3 | 4 | 0 | 1 | 0 | 4 | not in main claim group |
| toy2_precommitment_peer_evidence_stag_hunt_calibration | objective_basin_stag_hunt_p0p35 | diagnostic | false | diagnostic_only | 2 | 16 | 1 | 0.866667 | 0.00396099 | 3.98333 | not in main claim group |
| toy2_precommitment_peer_evidence_stag_hunt_calibration | revision_objective_basin_stag_hunt_p0p35 | diagnostic | false | diagnostic_only | 3 | 16.6667 | 0 | 0.866667 | 0.00306638 | 4 | not in main claim group |
| toy2_precommitment_peer_evidence_stag_hunt_calibration | revision_precommitment_evidence_stag_hunt_p0p35 | diagnostic | false | diagnostic_only | 3 | 11 | 0 | 1 | 0.000544218 | 4 | not in main claim group |
| toy2_precommitment_peer_evidence_stag_hunt_calibration | revision_precommitment_peer_evidence_stag_hunt_p0p35 | peer_evidence_domain_uncertainty_calibration | true | pass | 3 | 10 | 0 | 1 | 0 | 4 |  |
| toy4_precommitment_peer_evidence_resource_coupled_calibration | reputation_imitation_resource_p0p35 | baseline | false | diagnostic_only | 3 | 15 | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_precommitment_peer_evidence_resource_coupled_calibration | objective_basin_resource_p0p35 | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | 0 | not in main claim group |
| toy4_precommitment_peer_evidence_resource_coupled_calibration | revision_objective_basin_resource_p0p35 | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | -0.00333333 | not in main claim group |
| toy4_precommitment_peer_evidence_resource_coupled_calibration | revision_precommitment_evidence_resource_p0p35 | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | -0.00333333 | not in main claim group |
| toy4_precommitment_peer_evidence_resource_coupled_calibration | revision_precommitment_peer_evidence_resource_p0p35 | peer_evidence_domain_uncertainty_calibration | true | fail | 0 |  | 0 | 0 |  | -0.00333333 | final ceiling hits 0 < 1 |
