# Evidence Gate: toy24_revision_operator_precommitment_peer_evidence_stability_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_revision_operator_precommitment_peer_evidence_stability | toy2 | pass | revision_operator_precommitment_peer_evidence_w1p0 | 10/10 | 9.5 | 0 | 1 | 0 | false |
| toy4_revision_operator_precommitment_peer_evidence_stability | toy4 | pass | revision_operator_precommitment_peer_evidence_w1p0 | 10/10 | 8.9 | 0 | 1 | 9.52381e-05 | false |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_revision_operator_precommitment_peer_evidence_stability | reputation_imitation | baseline | false | diagnostic_only | 10 | 2.7 | 0 | 1 | 0 | 3 | not in main claim group |
| toy2_revision_operator_precommitment_peer_evidence_stability | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 9 | 20.4 | 1 | 0.76 | 0.0096309 | 2.999 | not in main claim group |
| toy2_revision_operator_precommitment_peer_evidence_stability | revision_operator_precommitment_evidence | diagnostic | false | diagnostic_only | 10 | 10.9 | 0 | 1 | 0.000152564 | 3 | not in main claim group |
| toy2_revision_operator_precommitment_peer_evidence_stability | revision_operator_precommitment_peer_evidence_w1p0 | precommitment_candidate | true | pass | 10 | 9.5 | 0 | 1 | 0 | 3 |  |
| toy4_revision_operator_precommitment_peer_evidence_stability | reputation_imitation | baseline | false | diagnostic_only | 10 | 2.7 | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_revision_operator_precommitment_peer_evidence_stability | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 8 | 19.5 | 2 | 0.74 | 0.00796859 | 0.5988 | not in main claim group |
| toy4_revision_operator_precommitment_peer_evidence_stability | revision_operator_precommitment_evidence | diagnostic | false | diagnostic_only | 10 | 10.4 | 0 | 1 | 0.000358974 | 0.6 | not in main claim group |
| toy4_revision_operator_precommitment_peer_evidence_stability | revision_operator_precommitment_peer_evidence_w1p0 | precommitment_candidate | true | pass | 10 | 8.9 | 0 | 1 | 9.52381e-05 | 0.6 |  |
