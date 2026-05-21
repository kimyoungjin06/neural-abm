# Evidence Gate: toy4_noisy_rep_local_obs_stress_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| toy4_noisy_rep_local_obs_stress | toy4 | pass | rev_local_sustain_obs_noisy_s2p0 | 5/5 | 30 | 0 | 1 | 0 | false |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| toy4_noisy_rep_local_obs_stress | rep_clean | baseline | false | diagnostic_only | 5 | 15 | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_noisy_rep_local_obs_stress | rep_noisy_s2p0 | diagnostic | false | diagnostic_only | 3 | 47.3333 | 0 | 0.6 | 0 | 0.493425 | not in main claim group |
| toy4_noisy_rep_local_obs_stress | rev_pop_global_obs_noisy_s2p0 | diagnostic | false | diagnostic_only | 0 |  | 0 | 0 |  | -0.298 | not in main claim group |
| toy4_noisy_rep_local_obs_stress | rev_local_global_obs_noisy_s2p0 | diagnostic | false | diagnostic_only | 5 | 30.8 | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_noisy_rep_local_obs_stress | rev_local_hidden_obs_noisy_s2p0 | diagnostic | false | diagnostic_only | 5 | 30.4 | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_noisy_rep_local_obs_stress | rev_local_sustain_obs_noisy_s2p0 | toy4_noisy_rep_local_obs_stress | true | pass | 5 | 30 | 0 | 1 | 0 | 0.6 |  |
