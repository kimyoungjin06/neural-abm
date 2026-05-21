# Evidence Profile: toy4_resource_threshold_hardening_quick

## Inputs

- Manifest: `experiments/evidence/toy4_resource_threshold_hardening_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy4_resource_threshold_hardening_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy4_resource_threshold_hardening_quick.summary.json`

## Overview

- Gate status: `pass`
- Passed: `True`
- Notes: toy24_objective_basin_evidence, toy24_revision_operator_evidence

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy4_resource_threshold_hardening | toy4 | pass | revision_precommitment_peer_evidence_resource_threshold_local_envw2p0 | 5/5 | 30.8 | 0.6 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy4_resource_threshold_hardening | reputation_imitation_resource_p0p35 | baseline | diagnostic_only | 5/5 | 15 | 0.6 | 1 |  |
| toy4_resource_threshold_hardening | revision_precommitment_peer_evidence_resource_threshold_population_envw2p0 | diagnostic | diagnostic_only | 0/5 |  | -0.298 | 0 | seed_level_final_miss |
| toy4_resource_threshold_hardening | revision_precommitment_peer_evidence_resource_threshold_local_envw2p0 | main | pass | 5/5 | 30.8 | 0.6 | 1 |  |
