# Evidence Gate: toy4_resource_threshold_noisy_reputation_stress_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy4_resource_threshold_noisy_reputation_stress | toy4 | pass | revision_precommitment_peer_evidence_resource_threshold_local_envw2p0_reputation_noise_s2p0 | 5/5 | 30.8 | 0 | 1 | 0 | false |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy4_resource_threshold_noisy_reputation_stress | reputation_imitation_resource_p0p35_clean | baseline | false | diagnostic_only | 5 | 15 | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_resource_threshold_noisy_reputation_stress | reputation_imitation_resource_p0p35_noisy_s1p0 | diagnostic | false | diagnostic_only | 4 | 43.75 | 0 | 0.72 | 0 | 0.5952 | not in main claim group |
| toy4_resource_threshold_noisy_reputation_stress | reputation_imitation_resource_p0p35_noisy_s2p0 | diagnostic | false | diagnostic_only | 3 | 47.3333 | 0 | 0.6 | 0 | 0.493425 | not in main claim group |
| toy4_resource_threshold_noisy_reputation_stress | revision_precommitment_peer_evidence_resource_threshold_population_envw2p0 | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | -0.298 | not in main claim group |
| toy4_resource_threshold_noisy_reputation_stress | revision_precommitment_peer_evidence_resource_threshold_local_envw2p0_reputation_noise_s2p0 | resource_threshold_noisy_reputation_stress | true | pass | 5 | 30.8 | 0 | 1 | 0 | 0.6 |  |
