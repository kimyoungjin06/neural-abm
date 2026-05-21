# Evidence Profile: toy24_precommitment_peer_evidence_noisy_reputation_stress_quick

## Inputs

- Manifest: `experiments/evidence/toy24_precommitment_peer_evidence_noisy_reputation_stress_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_noisy_reputation_stress_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy24_precommitment_peer_evidence_noisy_reputation_stress_quick.summary.json`

## Overview

- Gate status: `pass`
- Passed: `True`
- Notes: toy24_objective_basin_evidence, toy24_revision_operator_evidence

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy2_precommitment_peer_evidence_noisy_reputation_stress | toy2 | pass | revision_precommitment_peer_evidence_noisy_s1p0 | 5/5 | 9.4 | 3 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path |
| toy4_precommitment_peer_evidence_noisy_reputation_stress | toy4 | pass | revision_precommitment_peer_evidence_noisy_s1p0 | 5/5 | 9 | 0.6 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path, toy24_late_flip_hazard |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy2_precommitment_peer_evidence_noisy_reputation_stress | reputation_imitation_noisy_s1p0 | baseline | diagnostic_only | 4/5 | 33.25 | 2.932 | 0.8 | seed_level_final_miss |
| toy2_precommitment_peer_evidence_noisy_reputation_stress | objective_basin_noisy_s1p0 | diagnostic | diagnostic_only | 4/5 | 20.8 | 2.998 | 0.84 | seed_level_final_miss |
| toy2_precommitment_peer_evidence_noisy_reputation_stress | revision_objective_basin_noisy_s1p0 | diagnostic | diagnostic_only | 4/5 | 21 | 2.998 | 0.76 | seed_level_final_miss |
| toy2_precommitment_peer_evidence_noisy_reputation_stress | revision_precommitment_evidence_noisy_s1p0 | diagnostic | diagnostic_only | 5/5 | 10.8 | 3 | 1 |  |
| toy2_precommitment_peer_evidence_noisy_reputation_stress | revision_precommitment_peer_evidence_noisy_s1p0 | main | pass | 5/5 | 9.4 | 3 | 1 |  |
| toy4_precommitment_peer_evidence_noisy_reputation_stress | reputation_imitation_noisy_s1p0 | baseline | diagnostic_only | 4/5 | 33.25 | 0.57 | 0.8 | seed_level_final_miss |
| toy4_precommitment_peer_evidence_noisy_reputation_stress | objective_basin_noisy_s1p0 | diagnostic | diagnostic_only | 4/5 | 19 | 0.5988 | 0.76 | seed_level_final_miss |
| toy4_precommitment_peer_evidence_noisy_reputation_stress | revision_objective_basin_noisy_s1p0 | diagnostic | diagnostic_only | 3/5 | 20.6 | 0.5976 | 0.72 | seed_level_final_miss |
| toy4_precommitment_peer_evidence_noisy_reputation_stress | revision_precommitment_evidence_noisy_s1p0 | diagnostic | diagnostic_only | 5/5 | 10.6 | 0.6 | 1 |  |
| toy4_precommitment_peer_evidence_noisy_reputation_stress | revision_precommitment_peer_evidence_noisy_s1p0 | main | pass | 5/5 | 9 | 0.6 | 1 |  |
