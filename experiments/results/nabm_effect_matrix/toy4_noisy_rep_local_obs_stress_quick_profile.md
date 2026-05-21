# Evidence Profile: toy4_noisy_rep_local_obs_stress_quick

## Inputs

- Manifest: `experiments/evidence/toy4_resource_threshold_noisy_reputation_local_observation_stress_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy4_noisy_rep_local_obs_stress_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy4_noisy_rep_local_obs_stress_quick.summary.json`

## Overview

- Gate status: `pass`
- Passed: `True`
- Notes: toy24_objective_basin_evidence, toy24_revision_operator_evidence

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy4_noisy_rep_local_obs_stress | toy4 | pass | rev_local_sustain_obs_noisy_s2p0 | 5/5 | 30 | 0.6 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy4_noisy_rep_local_obs_stress | rep_clean | baseline | diagnostic_only | 5/5 | 15 | 0.6 | 1 |  |
| toy4_noisy_rep_local_obs_stress | rep_noisy_s2p0 | diagnostic | diagnostic_only | 3/5 | 47.3333 | 0.493425 | 0.6 | seed_level_final_miss |
| toy4_noisy_rep_local_obs_stress | rev_pop_global_obs_noisy_s2p0 | diagnostic | diagnostic_only | 0/5 |  | -0.298 | 0 | seed_level_final_miss |
| toy4_noisy_rep_local_obs_stress | rev_local_global_obs_noisy_s2p0 | diagnostic | diagnostic_only | 5/5 | 30.8 | 0.6 | 1 |  |
| toy4_noisy_rep_local_obs_stress | rev_local_hidden_obs_noisy_s2p0 | diagnostic | diagnostic_only | 5/5 | 30.4 | 0.6 | 1 |  |
| toy4_noisy_rep_local_obs_stress | rev_local_sustain_obs_noisy_s2p0 | main | pass | 5/5 | 30 | 0.6 | 1 |  |
