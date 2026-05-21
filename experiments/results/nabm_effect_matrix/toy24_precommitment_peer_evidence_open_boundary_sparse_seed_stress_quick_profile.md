# Evidence Profile: toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick

## Inputs

- Manifest: `experiments/evidence/toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick.summary.json`

## Overview

- Gate status: `pass`
- Passed: `True`
- Notes: toy24_objective_basin_evidence, toy24_revision_operator_evidence

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy2_precommitment_peer_evidence_open_boundary_sparse_seed_stress | toy2 | pass | revision_precommitment_peer_evidence_open_sparse_p0p1 | 5/5 | 9.4 | 3 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path |
| toy4_precommitment_peer_evidence_open_boundary_sparse_seed_stress | toy4 | pass | revision_precommitment_peer_evidence_open_sparse_p0p1 | 5/5 | 9 | 0.6 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path, toy24_late_flip_hazard |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy2_precommitment_peer_evidence_open_boundary_sparse_seed_stress | reputation_imitation_open_sparse_p0p1 | baseline | diagnostic_only | 5/5 | 9.2 | 3 | 1 |  |
| toy2_precommitment_peer_evidence_open_boundary_sparse_seed_stress | objective_basin_open_sparse_p0p1 | diagnostic | diagnostic_only | 4/5 | 19 | 2.9985 | 0.84 | seed_level_final_miss |
| toy2_precommitment_peer_evidence_open_boundary_sparse_seed_stress | revision_objective_basin_open_sparse_p0p1 | diagnostic | diagnostic_only | 4/5 | 20.2 | 2.9975 | 0.76 | seed_level_final_miss |
| toy2_precommitment_peer_evidence_open_boundary_sparse_seed_stress | revision_precommitment_evidence_open_sparse_p0p1 | diagnostic | diagnostic_only | 5/5 | 10.8 | 3 | 1 |  |
| toy2_precommitment_peer_evidence_open_boundary_sparse_seed_stress | revision_precommitment_peer_evidence_open_sparse_p0p1 | main | pass | 5/5 | 9.4 | 3 | 1 |  |
| toy4_precommitment_peer_evidence_open_boundary_sparse_seed_stress | reputation_imitation_open_sparse_p0p1 | baseline | diagnostic_only | 5/5 | 8.8 | 0.6 | 1 |  |
| toy4_precommitment_peer_evidence_open_boundary_sparse_seed_stress | objective_basin_open_sparse_p0p1 | diagnostic | diagnostic_only | 4/5 | 18.2 | 0.598552 | 0.8 | seed_level_final_miss |
| toy4_precommitment_peer_evidence_open_boundary_sparse_seed_stress | revision_objective_basin_open_sparse_p0p1 | diagnostic | diagnostic_only | 4/5 | 19 | 0.598552 | 0.8 | seed_level_final_miss |
| toy4_precommitment_peer_evidence_open_boundary_sparse_seed_stress | revision_precommitment_evidence_open_sparse_p0p1 | diagnostic | diagnostic_only | 5/5 | 9.8 | 0.6 | 1 |  |
| toy4_precommitment_peer_evidence_open_boundary_sparse_seed_stress | revision_precommitment_peer_evidence_open_sparse_p0p1 | main | pass | 5/5 | 9 | 0.6 | 1 |  |
