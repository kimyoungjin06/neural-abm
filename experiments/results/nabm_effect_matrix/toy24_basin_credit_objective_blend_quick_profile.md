# Evidence Profile: toy24_basin_credit_objective_blend_quick

## Inputs

- Manifest: `experiments/evidence/toy24_basin_credit_objective_blend_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy24_basin_credit_objective_blend_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy24_basin_credit_objective_blend_quick.summary.json`

## Overview

- Gate status: `fail`
- Passed: `False`
- Notes: profile_has_case_issues, toy24_objective_basin_evidence, toy24_material_basin_collapse_contrast, toy24_final_epoch_hazard_evidence

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy2_basin_credit | toy2 | fail | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | 3/3 | 12 | 3 | gate_case_fail | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_material_basin_collapse_diagnostic, toy24_main_candidate_ceiling_miss |
| toy4_basin_credit | toy4 | pass | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | 3/3 | 11 | 0.6 | toy24_final_epoch_hazard | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_material_basin_collapse_diagnostic, toy24_final_vs_ever_gap, toy24_main_candidate_ceiling_miss |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy2_basin_credit | reputation_imitation | baseline | diagnostic_only | 3/3 | 2.66667 | 3 |  |  |
| toy2_basin_credit | linear_welfare_heavy | baseline | diagnostic_only | 3/3 | 23.6667 | 3 |  |  |
| toy2_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | diagnostic | diagnostic_only | 3/3 | 22.6667 | 3 |  |  |
| toy2_basin_credit | basin_credit_w1p0_h1_prototype | main | fail | 0/3 |  | 2.55667 |  | main_final_ceiling_miss, seed_level_final_miss |
| toy2_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | diagnostic | diagnostic_only | 0/3 |  | 1.06 |  | seed_level_final_miss |
| toy2_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | main | fail | 3/3 | 22.6667 | 3 |  |  |
| toy2_basin_credit | mixed_objective_basin_confidence_social_w0p5_0p5_h1 | main | fail | 3/3 | 18.3333 | 3 |  |  |
| toy2_basin_credit | mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1 | main | fail | 3/3 | 21.3333 | 3 |  |  |
| toy2_basin_credit | mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1 | main | fail | 3/3 | 18.3333 | 3 |  |  |
| toy2_basin_credit | mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1 | main | fail | 3/3 | 16 | 3 |  |  |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | main | fail | 3/3 | 12 | 3 |  |  |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1 | main | fail | 3/3 | 12 | 3 |  |  |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | main | fail | 3/3 | 12 | 3 |  |  |
| toy2_basin_credit | mixed_objective_basin_directional_social_w0p5_0p5_h1 | main | fail | 3/3 | 18.3333 | 3 |  |  |
| toy4_basin_credit | reputation_imitation | baseline | diagnostic_only | 3/3 | 2.66667 | 0.6 |  |  |
| toy4_basin_credit | linear_welfare_heavy | baseline | diagnostic_only | 1/3 | 21 | 0.596 |  | seed_level_final_miss |
| toy4_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | diagnostic | diagnostic_only | 1/3 | 19.6667 | 0.596 |  | seed_level_final_miss |
| toy4_basin_credit | basin_credit_w1p0_h1_prototype | main | fail | 0/3 |  | 0.314 |  | main_final_ceiling_miss, seed_level_final_miss |
| toy4_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | diagnostic | diagnostic_only | 0/3 |  | 0.004 |  | seed_level_final_miss |
| toy4_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | main | fail | 2/3 | 16.3333 | 0.598 |  | main_final_ceiling_miss, seed_level_final_miss |
| toy4_basin_credit | mixed_objective_basin_confidence_social_w0p5_0p5_h1 | main | fail | 1/3 | 18 | 0.596 |  | main_final_ceiling_miss, seed_level_final_miss |
| toy4_basin_credit | mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1 | main | fail | 2/3 | 18 | 0.598 |  | main_final_ceiling_miss, seed_level_final_miss |
| toy4_basin_credit | mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1 | main | fail | 1/3 | 18 | 0.596 |  | main_final_ceiling_miss, seed_level_final_miss |
| toy4_basin_credit | mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1 | main | fail | 3/3 | 13.6667 | 0.6 |  |  |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | main | pass | 3/3 | 11.3333 | 0.6 |  |  |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1 | main | pass | 3/3 | 11.3333 | 0.6 |  |  |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | main | pass | 3/3 | 11 | 0.6 |  |  |
| toy4_basin_credit | mixed_objective_basin_directional_social_w0p5_0p5_h1 | main | fail | 1/3 | 18 | 0.596 |  | main_final_ceiling_miss, seed_level_final_miss |
