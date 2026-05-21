# Evidence Profile: toy24_revision_operator_precommitment_controls_quick

## Inputs

- Manifest: `experiments/evidence/toy24_revision_operator_precommitment_controls_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy24_revision_operator_precommitment_controls_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy24_revision_operator_precommitment_controls_quick.summary.json`

## Overview

- Gate status: `pass`
- Passed: `True`
- Notes: toy24_objective_basin_evidence, toy24_revision_operator_evidence, toy24_triage_success_evidence, toy24_baseline_favored_environment_evidence

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy2_revision_operator_precommitment_controls | toy2 | pass | revision_operator_precommitment_peer_evidence_w1p0 | 3/3 | 9.66667 | 3 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path, toy24_late_flip_hazard, toy24_triage_success, toy24_triage_baseline_favored_environment |
| toy4_revision_operator_precommitment_controls | toy4 | pass | revision_operator_precommitment_peer_evidence_w1p0 | 3/3 | 9.33333 | 0.6 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path, toy24_late_flip_hazard, toy24_triage_success, toy24_triage_baseline_favored_environment |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy2_revision_operator_precommitment_controls | reputation_imitation | baseline | diagnostic_only | 3/3 | 2.66667 | 3 | 1 |  |
| toy2_revision_operator_precommitment_controls | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | diagnostic | diagnostic_only | 2/3 | 19.3333 | 2.99667 | 0.866667 | seed_level_final_miss |
| toy2_revision_operator_precommitment_controls | revision_operator_commitment_hysteresis | diagnostic | diagnostic_only | 3/3 | 15 | 3 | 1 |  |
| toy2_revision_operator_precommitment_controls | revision_operator_precommitment_evidence | main | fail | 3/3 | 11 | 3 | 1 |  |
| toy2_revision_operator_precommitment_controls | revision_operator_precommitment_peer_evidence_w0p25 | main | fail | 3/3 | 11 | 3 | 1 |  |
| toy2_revision_operator_precommitment_controls | revision_operator_precommitment_peer_evidence_w0p5 | main | fail | 3/3 | 10.3333 | 3 | 1 |  |
| toy2_revision_operator_precommitment_controls | revision_operator_precommitment_peer_evidence_w1p0 | main | pass | 3/3 | 9.66667 | 3 | 1 |  |
| toy2_revision_operator_precommitment_controls | revision_operator_precommitment_commitment_hysteresis | main | fail | 3/3 | 11 | 3 | 1 |  |
| toy4_revision_operator_precommitment_controls | reputation_imitation | baseline | diagnostic_only | 3/3 | 2.66667 | 0.6 | 1 |  |
| toy4_revision_operator_precommitment_controls | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | diagnostic | diagnostic_only | 1/3 | 19 | 0.596 | 0.8 | seed_level_final_miss |
| toy4_revision_operator_precommitment_controls | revision_operator_commitment_hysteresis | diagnostic | diagnostic_only | 3/3 | 13.6667 | 0.6 | 0.933333 |  |
| toy4_revision_operator_precommitment_controls | revision_operator_precommitment_evidence | main | pass | 3/3 | 11 | 0.6 | 1 |  |
| toy4_revision_operator_precommitment_controls | revision_operator_precommitment_peer_evidence_w0p25 | main | pass | 3/3 | 11 | 0.6 | 1 |  |
| toy4_revision_operator_precommitment_controls | revision_operator_precommitment_peer_evidence_w0p5 | main | pass | 3/3 | 10.6667 | 0.6 | 1 |  |
| toy4_revision_operator_precommitment_controls | revision_operator_precommitment_peer_evidence_w1p0 | main | pass | 3/3 | 9.33333 | 0.6 | 1 |  |
| toy4_revision_operator_precommitment_controls | revision_operator_precommitment_commitment_hysteresis | main | pass | 3/3 | 11 | 0.6 | 1 |  |
