# Evidence Gate: toy4_resource_threshold_hardening_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy4_resource_threshold_hardening | toy4 | pass | revision_precommitment_peer_evidence_resource_threshold_local_envw2p0 | 5/5 | 30.8 | 0 | 1 | 0 | false |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy4_resource_threshold_hardening | reputation_imitation_resource_p0p35 | baseline | false | diagnostic_only | 5 | 15 | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_resource_threshold_hardening | revision_precommitment_peer_evidence_resource_threshold_population_envw2p0 | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | -0.298 | not in main claim group |
| toy4_resource_threshold_hardening | revision_precommitment_peer_evidence_resource_threshold_local_envw2p0 | resource_threshold_hardening | true | pass | 5 | 30.8 | 0 | 1 | 0 | 0.6 |  |
