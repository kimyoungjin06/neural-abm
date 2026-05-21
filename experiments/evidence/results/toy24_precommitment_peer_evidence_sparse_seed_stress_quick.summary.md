# Evidence Gate: toy24_precommitment_peer_evidence_sparse_seed_stress_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_precommitment_peer_evidence_sparse_seed_stress | toy2 | pass | revision_precommitment_peer_evidence_sparse_p0p1 | 5/5 | 9.4 | 0 | 1 | 0 | false |
| toy4_precommitment_peer_evidence_sparse_seed_stress | toy4 | pass | revision_precommitment_peer_evidence_sparse_p0p1 | 5/5 | 8.8 | 0 | 1 | 0 | false |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_precommitment_peer_evidence_sparse_seed_stress | reputation_imitation_sparse_p0p1 | baseline | false | diagnostic_only | 5 | 7.6 | 0 | 1 | 0 | 3 | not in main claim group |
| toy2_precommitment_peer_evidence_sparse_seed_stress | objective_basin_sparse_p0p1 | diagnostic | false | diagnostic_only | 4 | 20.8 | 1 | 0.84 | 0.00922321 | 2.998 | not in main claim group |
| toy2_precommitment_peer_evidence_sparse_seed_stress | revision_objective_basin_sparse_p0p1 | diagnostic | false | diagnostic_only | 4 | 20.2 | 1 | 0.72 | 0.00932206 | 2.998 | not in main claim group |
| toy2_precommitment_peer_evidence_sparse_seed_stress | revision_precommitment_evidence_sparse_p0p1 | diagnostic | false | diagnostic_only | 5 | 10.8 | 0 | 1 | 0.000102564 | 3 | not in main claim group |
| toy2_precommitment_peer_evidence_sparse_seed_stress | revision_precommitment_peer_evidence_sparse_p0p1 | peer_evidence_sparse_seed_stress | true | pass | 5 | 9.4 | 0 | 1 | 0 | 3 |  |
| toy4_precommitment_peer_evidence_sparse_seed_stress | reputation_imitation_sparse_p0p1 | baseline | false | diagnostic_only | 5 | 7.6 | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_precommitment_peer_evidence_sparse_seed_stress | objective_basin_sparse_p0p1 | diagnostic | false | diagnostic_only | 4 | 19.2 | 1 | 0.8 | 0.00738732 | 0.5988 | not in main claim group |
| toy4_precommitment_peer_evidence_sparse_seed_stress | revision_objective_basin_sparse_p0p1 | diagnostic | false | diagnostic_only | 4 | 19.8 | 1 | 0.8 | 0.0067185 | 0.5988 | not in main claim group |
| toy4_precommitment_peer_evidence_sparse_seed_stress | revision_precommitment_evidence_sparse_p0p1 | diagnostic | false | diagnostic_only | 5 | 10.2 | 0 | 1 | 0.000195122 | 0.6 | not in main claim group |
| toy4_precommitment_peer_evidence_sparse_seed_stress | revision_precommitment_peer_evidence_sparse_p0p1 | peer_evidence_sparse_seed_stress | true | pass | 5 | 8.8 | 0 | 1 | 0 | 0.6 |  |
