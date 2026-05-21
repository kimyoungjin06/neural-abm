# Evidence Gate: toy4_resource_lookahead_probe_quick

Overall status: **fail**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy4_resource_lookahead_probe | toy4 | fail | revision_precommitment_peer_evidence_resource_lookahead_envw2p0 | 0/3 |  | 0 | 0 |  | false |

## Next Diagnostics

- toy4_resource_lookahead_probe: inspect aggregate trajectories before adding a contrastive critic.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy4_resource_lookahead_probe | reputation_imitation_resource_p0p35 | baseline | false | diagnostic_only | 3 | 15 | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_resource_lookahead_probe | revision_precommitment_peer_evidence_resource_pressure_envw2p0 | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | -0.319948 | not in main claim group |
| toy4_resource_lookahead_probe | revision_precommitment_peer_evidence_resource_lookahead_envw2p0 | resource_lookahead_probe | true | fail | 0 |  | 0 | 0 |  | -0.00333333 | final ceiling hits 0 < 1 |
| toy4_resource_lookahead_probe | revision_precommitment_peer_evidence_resource_lookahead_envw5p0 | resource_lookahead_probe | true | fail | 0 |  | 0 | 0 |  | -0.00333333 | final ceiling hits 0 < 1 |
| toy4_resource_lookahead_probe | revision_precommitment_peer_evidence_resource_lookahead_envw10p0 | resource_lookahead_probe | true | fail | 0 |  | 0 | 0 |  | -0.00333333 | final ceiling hits 0 < 1 |
