# Evidence Profile: toy4_resource_threshold_heterogeneous_extraction_stress_quick

## Inputs

- Manifest: `experiments/evidence/toy4_resource_threshold_heterogeneous_extraction_stress_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy4_resource_threshold_heterogeneous_extraction_stress_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy4_resource_threshold_heterogeneous_extraction_stress_quick.summary.json`

## Overview

- Gate status: `pass`
- Passed: `True`
- Notes: toy24_objective_basin_evidence, toy24_revision_operator_evidence

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy4_resource_threshold_heterogeneous_extraction_stress | toy4 | pass | revision_precommitment_peer_evidence_resource_threshold_local_envw2p0_heterogeneous_extraction_h1p0 | 5/5 | 33 | 0.6 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy4_resource_threshold_heterogeneous_extraction_stress | reputation_imitation_resource_p0p35_heterogeneous_extraction_h1p0 | baseline | diagnostic_only | 5/5 | 15 | 0.6 | 1 |  |
| toy4_resource_threshold_heterogeneous_extraction_stress | reputation_imitation_resource_p0p35_noisy_s2p0_heterogeneous_extraction_h1p0 | diagnostic | diagnostic_only | 3/5 | 47.3333 | 0.494039 | 0.6 | seed_level_final_miss |
| toy4_resource_threshold_heterogeneous_extraction_stress | revision_precommitment_peer_evidence_resource_threshold_population_envw2p0_heterogeneous_extraction_h1p0 | diagnostic | diagnostic_only | 0/5 |  | -0.302 | 0 | seed_level_final_miss |
| toy4_resource_threshold_heterogeneous_extraction_stress | revision_precommitment_peer_evidence_resource_threshold_local_envw2p0_heterogeneous_extraction_h1p0 | main | pass | 5/5 | 33 | 0.6 | 1 |  |
