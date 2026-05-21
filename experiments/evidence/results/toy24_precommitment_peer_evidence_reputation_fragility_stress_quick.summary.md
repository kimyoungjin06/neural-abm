# Evidence Gate: toy24_precommitment_peer_evidence_reputation_fragility_stress_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Trajectory | Failure Mode | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| toy2_precommitment_peer_evidence_reputation_fragility_stress | toy2 | pass | revision_precommitment_peer_evidence_open_sparse_noisy_p0p1_s1p0 | 5/5 | 9.4 | success |  | 0 | 1 | 0 | true |
| toy4_precommitment_peer_evidence_reputation_fragility_stress | toy4 | pass | revision_precommitment_peer_evidence_open_sparse_noisy_p0p1_s1p0 | 5/5 | 9 | success |  | 0 | 1 | 4.7619e-05 | true |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Trajectory | Failure Mode | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy2_precommitment_peer_evidence_reputation_fragility_stress | reputation_imitation_open_sparse_noisy_p0p1_s1p0 | baseline | false | diagnostic_only | 0 |  | diagnostic | not_main_group | 0 | 0 |  | 2.49733 | not in main claim group |
| toy2_precommitment_peer_evidence_reputation_fragility_stress | objective_basin_open_sparse_noisy_p0p1_s1p0 | diagnostic | false | diagnostic_only | 4 | 19 | diagnostic | not_main_group | 1 | 0.84 | 0.0111861 | 2.9985 | not in main claim group |
| toy2_precommitment_peer_evidence_reputation_fragility_stress | revision_objective_basin_open_sparse_noisy_p0p1_s1p0 | diagnostic | false | diagnostic_only | 4 | 20.2 | diagnostic | not_main_group | 1 | 0.76 | 0.00984958 | 2.9975 | not in main claim group |
| toy2_precommitment_peer_evidence_reputation_fragility_stress | revision_precommitment_evidence_open_sparse_noisy_p0p1_s1p0 | diagnostic | false | diagnostic_only | 5 | 10.8 | diagnostic | not_main_group | 0 | 1 | 0.000102564 | 3 | not in main claim group |
| toy2_precommitment_peer_evidence_reputation_fragility_stress | revision_precommitment_peer_evidence_open_sparse_noisy_p0p1_s1p0 | peer_evidence_reputation_fragility_stress | true | pass | 5 | 9.4 | success |  | 0 | 1 | 0 | 3 |  |
| toy4_precommitment_peer_evidence_reputation_fragility_stress | reputation_imitation_open_sparse_noisy_p0p1_s1p0 | baseline | false | diagnostic_only | 0 |  | diagnostic | not_main_group | 0 | 0 |  | 0.40833 | not in main claim group |
| toy4_precommitment_peer_evidence_reputation_fragility_stress | objective_basin_open_sparse_noisy_p0p1_s1p0 | diagnostic | false | diagnostic_only | 4 | 18.2 | diagnostic | not_main_group | 1 | 0.8 | 0.00778649 | 0.598552 | not in main claim group |
| toy4_precommitment_peer_evidence_reputation_fragility_stress | revision_objective_basin_open_sparse_noisy_p0p1_s1p0 | diagnostic | false | diagnostic_only | 4 | 19 | diagnostic | not_main_group | 1 | 0.8 | 0.00726735 | 0.598552 | not in main claim group |
| toy4_precommitment_peer_evidence_reputation_fragility_stress | revision_precommitment_evidence_open_sparse_noisy_p0p1_s1p0 | diagnostic | false | diagnostic_only | 5 | 9.8 | diagnostic | not_main_group | 0 | 1 | 0.000540306 | 0.6 | not in main claim group |
| toy4_precommitment_peer_evidence_reputation_fragility_stress | revision_precommitment_peer_evidence_open_sparse_noisy_p0p1_s1p0 | peer_evidence_reputation_fragility_stress | true | pass | 5 | 9 | success |  | 0 | 1 | 4.7619e-05 | 0.6 |  |
