# Evidence Profile: toy4_resource_environment_weight_probe_quick

## Inputs

- Manifest: `experiments/evidence/toy4_resource_environment_weight_probe_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy4_resource_environment_weight_probe_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy4_resource_environment_weight_probe_quick.summary.json`

## Overview

- Gate status: `fail`
- Passed: `False`
- Notes: profile_has_case_issues, toy24_objective_basin_evidence, toy24_revision_operator_evidence

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy4_resource_environment_weight_probe | toy4 | fail | revision_precommitment_peer_evidence_resource_envw0p5 | 0/3 |  | -0.00333333 | gate_case_fail, toy24_best_main_ceiling_miss | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path, toy24_main_candidate_ceiling_miss |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy4_resource_environment_weight_probe | reputation_imitation_resource_p0p35 | baseline | diagnostic_only | 3/3 | 15 | 0.6 | 1 |  |
| toy4_resource_environment_weight_probe | revision_precommitment_peer_evidence_resource_envw0p0 | diagnostic | diagnostic_only | 0/3 |  | -0.00333333 | 0 | seed_level_final_miss |
| toy4_resource_environment_weight_probe | revision_precommitment_peer_evidence_resource_envw0p5 | main | fail | 0/3 |  | -0.00333333 | 0 | main_final_ceiling_miss, seed_level_final_miss |
| toy4_resource_environment_weight_probe | revision_precommitment_peer_evidence_resource_envw1p0 | main | fail | 0/3 |  | -0.14 | 0 | main_final_ceiling_miss, seed_level_final_miss |
| toy4_resource_environment_weight_probe | revision_precommitment_peer_evidence_resource_envw2p0 | main | fail | 0/3 |  | -0.319948 | 0 | main_final_ceiling_miss, seed_level_final_miss |
