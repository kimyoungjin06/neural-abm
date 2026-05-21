# Evidence Profile: toy24_precommitment_peer_evidence_domain_uncertainty_calibration_quick

## Inputs

- Manifest: `experiments/evidence/toy24_precommitment_peer_evidence_domain_uncertainty_calibration_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_domain_uncertainty_calibration_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy24_precommitment_peer_evidence_domain_uncertainty_calibration_quick.summary.json`

## Overview

- Gate status: `fail`
- Passed: `False`
- Notes: profile_has_case_issues, toy24_objective_basin_evidence, toy24_revision_operator_evidence

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy2_precommitment_peer_evidence_stag_hunt_calibration | toy2 | pass | revision_precommitment_peer_evidence_stag_hunt_p0p35 | 3/3 | 10 | 4 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path |
| toy4_precommitment_peer_evidence_resource_coupled_calibration | toy4 | fail | revision_precommitment_peer_evidence_resource_p0p35 | 0/3 |  | -0.00333333 | gate_case_fail, toy24_best_main_ceiling_miss | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path, toy24_main_candidate_ceiling_miss |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy2_precommitment_peer_evidence_stag_hunt_calibration | reputation_imitation_stag_hunt_p0p35 | baseline | diagnostic_only | 3/3 | 4 | 4 | 1 |  |
| toy2_precommitment_peer_evidence_stag_hunt_calibration | objective_basin_stag_hunt_p0p35 | diagnostic | diagnostic_only | 2/3 | 16 | 3.98333 | 0.866667 | seed_level_final_miss |
| toy2_precommitment_peer_evidence_stag_hunt_calibration | revision_objective_basin_stag_hunt_p0p35 | diagnostic | diagnostic_only | 3/3 | 16.6667 | 4 | 0.866667 |  |
| toy2_precommitment_peer_evidence_stag_hunt_calibration | revision_precommitment_evidence_stag_hunt_p0p35 | diagnostic | diagnostic_only | 3/3 | 11 | 4 | 1 |  |
| toy2_precommitment_peer_evidence_stag_hunt_calibration | revision_precommitment_peer_evidence_stag_hunt_p0p35 | main | pass | 3/3 | 10 | 4 | 1 |  |
| toy4_precommitment_peer_evidence_resource_coupled_calibration | reputation_imitation_resource_p0p35 | baseline | diagnostic_only | 3/3 | 15 | 0.6 | 1 |  |
| toy4_precommitment_peer_evidence_resource_coupled_calibration | objective_basin_resource_p0p35 | diagnostic | diagnostic_only | 0/3 |  | 0 | 0 | seed_level_final_miss |
| toy4_precommitment_peer_evidence_resource_coupled_calibration | revision_objective_basin_resource_p0p35 | diagnostic | diagnostic_only | 0/3 |  | -0.00333333 | 0 | seed_level_final_miss |
| toy4_precommitment_peer_evidence_resource_coupled_calibration | revision_precommitment_evidence_resource_p0p35 | diagnostic | diagnostic_only | 0/3 |  | -0.00333333 | 0 | seed_level_final_miss |
| toy4_precommitment_peer_evidence_resource_coupled_calibration | revision_precommitment_peer_evidence_resource_p0p35 | main | fail | 0/3 |  | -0.00333333 | 0 | main_final_ceiling_miss, seed_level_final_miss |
