# Evidence Gate: toy4_hetero_local_obs_stress_quick

Overall status: **pass**

## Main Claim Cases

| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Trajectory | Failure Mode | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |
| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| toy4_hetero_local_obs_stress | toy4 | pass | rev_local_sustain_obs_noisy_s2p0_hetero | 5/5 | 31.8 | success |  | 0 | 1 | 0 | false |

## Next Diagnostics

- Gate passed; preserve these artifacts before expanding claims.

## Variant Details

| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Trajectory | Failure Mode | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy4_hetero_local_obs_stress | rep_clean_hetero | baseline | false | diagnostic_only | 5 | 15 | diagnostic | not_main_group | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_hetero_local_obs_stress | rep_noisy_s2p0_hetero | diagnostic | false | diagnostic_only | 3 | 47.3333 | diagnostic | not_main_group | 0 | 0.6 | 0 | 0.494039 | not in main claim group |
| toy4_hetero_local_obs_stress | rev_pop_global_obs_noisy_s2p0_hetero | diagnostic | false | diagnostic_only | 0 |  | diagnostic | not_main_group | 0 | 0 |  | -0.302 | not in main claim group |
| toy4_hetero_local_obs_stress | rev_local_global_obs_noisy_s2p0_hetero | diagnostic | false | diagnostic_only | 5 | 33 | diagnostic | not_main_group | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_hetero_local_obs_stress | rev_local_hidden_obs_noisy_s2p0_hetero | diagnostic | false | diagnostic_only | 5 | 32.2 | diagnostic | not_main_group | 0 | 1 | 0 | 0.6 | not in main claim group |
| toy4_hetero_local_obs_stress | rev_local_sustain_obs_noisy_s2p0_hetero | toy4_hetero_local_obs_stress | true | pass | 5 | 31.8 | success |  | 0 | 1 | 0 | 0.6 |  |
