# Evidence Profile: toy24_revision_operator_quick

## Inputs

- Manifest: `experiments/evidence/toy24_revision_operator_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy24_revision_operator_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy24_revision_operator_quick.summary.json`

## Overview

- Gate status: `fail`
- Passed: `False`
- Notes: profile_has_case_issues, toy24_objective_basin_evidence, toy24_revision_operator_evidence, toy24_final_epoch_hazard_evidence, toy24_stochastic_gate_brittleness_evidence, toy24_baseline_favored_environment_evidence

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy2_revision_operator | toy2 | fail | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | 2/3 | 19.3333 | 2.99667 | gate_case_fail, toy24_final_epoch_hazard, toy24_best_main_ceiling_miss, toy24_stochastic_gate_brittleness | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path, toy24_final_vs_ever_gap, toy24_late_flip_hazard, toy24_main_candidate_ceiling_miss, toy24_triage_stochastic_gate_brittleness, toy24_triage_baseline_favored_environment |
| toy4_revision_operator | toy4 | fail | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | 1/3 | 19 | 0.596 | gate_case_fail, toy24_final_epoch_hazard, toy24_best_main_ceiling_miss, toy24_stochastic_gate_brittleness | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path, toy24_final_vs_ever_gap, toy24_late_flip_hazard, toy24_main_candidate_ceiling_miss, toy24_triage_stochastic_gate_brittleness, toy24_triage_baseline_favored_environment |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy2_revision_operator | reputation_imitation | baseline | diagnostic_only | 3/3 | 2.66667 | 3 | 1 |  |
| toy2_revision_operator | linear_welfare_heavy | baseline | diagnostic_only | 3/3 | 23.6667 | 3 | 0.866667 |  |
| toy2_revision_operator | mixed_objective_basin_w0p5_0p5_h1 | diagnostic | diagnostic_only | 3/3 | 22.6667 | 3 | 0.866667 |  |
| toy2_revision_operator | revision_operator_linear_welfare_heavy | diagnostic | diagnostic_only | 2/3 | 19.6667 | 2.99667 | 0.933333 | seed_level_final_miss |
| toy2_revision_operator | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | main | fail | 2/3 | 19.3333 | 2.99667 | 0.866667 | main_final_ceiling_miss, seed_level_final_miss |
| toy4_revision_operator | reputation_imitation | baseline | diagnostic_only | 3/3 | 2.66667 | 0.6 | 1 |  |
| toy4_revision_operator | linear_welfare_heavy | baseline | diagnostic_only | 1/3 | 21 | 0.596 | 0.733333 | seed_level_final_miss |
| toy4_revision_operator | mixed_objective_basin_w0p5_0p5_h1 | diagnostic | diagnostic_only | 2/3 | 16.3333 | 0.598 | 0.866667 | seed_level_final_miss |
| toy4_revision_operator | revision_operator_linear_welfare_heavy | diagnostic | diagnostic_only | 1/3 | 19.3333 | 0.596 | 0.733333 | seed_level_final_miss |
| toy4_revision_operator | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | main | fail | 1/3 | 19 | 0.596 | 0.8 | main_final_ceiling_miss, seed_level_final_miss |
