# Evidence Gate: toy24_precommitment_peer_evidence_closure_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_precommitment_peer_evidence_closure | toy2 | pass | revision_precommitment_peer_evidence_w1p0 | 5/5 | 9.4 | 0 | 1 | 0 | false |
| toy4_precommitment_peer_evidence_closure | toy4 | pass | revision_precommitment_peer_evidence_w1p0 | 5/5 | 9 | 0 | 1 | 9.52381e-05 | false |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy2_precommitment_peer_evidence_closure | reputation_imitation | baseline | false | diagnostic_only | 5 | 2.6 | 0 | 1 | 0 | 3 | not in main claim group |
| toy2_precommitment_peer_evidence_closure | objective_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 4 | 20.8 | 1 | 0.84 | 0.00880655 | 2.998 | not in main claim group |
| toy2_precommitment_peer_evidence_closure | revision_objective_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 4 | 21 | 1 | 0.76 | 0.00928179 | 2.998 | not in main claim group |
| toy2_precommitment_peer_evidence_closure | revision_precommitment_evidence | diagnostic | false | diagnostic_only | 5 | 10.8 | 0 | 1 | 0.000102564 | 3 | not in main claim group |
| toy2_precommitment_peer_evidence_closure | revision_precommitment_peer_evidence_w1p0 | peer_evidence_closure | true | pass | 5 | 9.4 | 0 | 1 | 0 | 3 |  |
| toy4_precommitment_peer_evidence_closure | reputation_imitation | baseline | false | diagnostic_only | 5 | 2.6 | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_precommitment_peer_evidence_closure | objective_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 4 | 19 | 1 | 0.76 | 0.0084539 | 0.5988 | not in main claim group |
| toy4_precommitment_peer_evidence_closure | revision_objective_basin_w0p5_0p5_h1 | diagnostic | false | diagnostic_only | 3 | 20.6 | 2 | 0.72 | 0.00795711 | 0.5976 | not in main claim group |
| toy4_precommitment_peer_evidence_closure | revision_precommitment_evidence | diagnostic | false | diagnostic_only | 5 | 10.6 | 0 | 1 | 0.000512821 | 0.6 | not in main claim group |
| toy4_precommitment_peer_evidence_closure | revision_precommitment_peer_evidence_w1p0 | peer_evidence_closure | true | pass | 5 | 9 | 0 | 1 | 9.52381e-05 | 0.6 |  |
