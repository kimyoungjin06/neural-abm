# Evidence Profile: toy5_neural_threshold_target_threshold_aware_grid_quick

## Inputs

- Manifest: `experiments/evidence/toy5_neural_threshold_target_threshold_aware_grid_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_threshold_aware_grid_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy5_neural_threshold_target_threshold_aware_grid_quick.summary.json`

## Overview

- Gate status: `pass`
- Passed: `True`
- Notes: toy5_threshold_aware_evidence

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy5_threshold_aware_grid_no_seed_heterogeneous_safety | toy5 | pass | neural_threshold_aware_grid_no_seed_threshold_anchor | 5/5 | 0 | 1 |  | toy5_no_seed_safety_case, toy5_direction_gate_separates_self_excitation, toy5_threshold_aware_direction |
| toy5_threshold_aware_grid_lattice_k4_h0p85_spread | toy5 | pass | neural_threshold_aware_grid_lattice_k4_h0p85_threshold_anchor | 5/5 | 36.2 | 100 |  | toy5_threshold_aware_direction |
| toy5_threshold_aware_grid_lattice_k4_h0p95_spread | toy5 | pass | neural_threshold_aware_grid_lattice_k4_h0p95_threshold_anchor | 5/5 | 37 | 100 |  | toy5_threshold_aware_direction |
| toy5_threshold_aware_grid_lattice_k6_h0p85_spread | toy5 | pass | neural_threshold_aware_grid_lattice_k6_h0p85_threshold_anchor | 5/5 | 25 | 100 |  | toy5_threshold_aware_direction |
| toy5_threshold_aware_grid_lattice_k6_h0p95_spread | toy5 | pass | neural_threshold_aware_grid_lattice_k6_h0p95_threshold_anchor | 5/5 | 25 | 100 |  | toy5_threshold_aware_direction |
| toy5_threshold_aware_grid_rewired_k6_p0p10_h0p85_spread | toy5 | pass | neural_threshold_aware_grid_rewired_k6_p0p10_h0p85_threshold_anchor | 5/5 | 9.6 | 100 |  | toy5_threshold_aware_direction |
| toy5_threshold_aware_grid_rewired_k6_p0p10_h0p95_spread | toy5 | pass | neural_threshold_aware_grid_rewired_k6_p0p10_h0p95_threshold_anchor | 5/5 | 10 | 100 |  | toy5_threshold_aware_direction |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy5_threshold_aware_grid_no_seed_heterogeneous_safety | neural_threshold_aware_grid_no_seed_output_average | baseline | diagnostic_only | 5/5 | 0 | 1 | 1 |  |
| toy5_threshold_aware_grid_no_seed_heterogeneous_safety | neural_threshold_aware_grid_no_seed_non_directional | diagnostic | diagnostic_only | 0/5 | 0 | 0 | 0 | seed_level_final_miss |
| toy5_threshold_aware_grid_no_seed_heterogeneous_safety | neural_threshold_aware_grid_no_seed_threshold_anchor | main | pass | 5/5 | 0 | 1 | 1 |  |
| toy5_threshold_aware_grid_lattice_k4_h0p85_spread | neural_threshold_aware_grid_lattice_k4_h0p85_output_average | baseline | diagnostic_only | 0/5 |  | 1 | 0 | seed_level_final_miss |
| toy5_threshold_aware_grid_lattice_k4_h0p85_spread | neural_threshold_aware_grid_lattice_k4_h0p85_exposure_anchor | diagnostic | diagnostic_only | 5/5 | 33 | 100 | 1 |  |
| toy5_threshold_aware_grid_lattice_k4_h0p85_spread | neural_threshold_aware_grid_lattice_k4_h0p85_threshold_anchor | main | pass | 5/5 | 36.2 | 100 | 1 |  |
| toy5_threshold_aware_grid_lattice_k4_h0p95_spread | neural_threshold_aware_grid_lattice_k4_h0p95_output_average | baseline | diagnostic_only | 0/5 |  | 1 | 0 | seed_level_final_miss |
| toy5_threshold_aware_grid_lattice_k4_h0p95_spread | neural_threshold_aware_grid_lattice_k4_h0p95_exposure_anchor | diagnostic | diagnostic_only | 5/5 | 33 | 100 | 1 |  |
| toy5_threshold_aware_grid_lattice_k4_h0p95_spread | neural_threshold_aware_grid_lattice_k4_h0p95_threshold_anchor | main | pass | 5/5 | 37 | 100 | 1 |  |
| toy5_threshold_aware_grid_lattice_k6_h0p85_spread | neural_threshold_aware_grid_lattice_k6_h0p85_output_average | baseline | diagnostic_only | 0/5 |  | 1 | 0 | seed_level_final_miss |
| toy5_threshold_aware_grid_lattice_k6_h0p85_spread | neural_threshold_aware_grid_lattice_k6_h0p85_exposure_anchor | diagnostic | diagnostic_only | 5/5 | 23 | 100 | 1 |  |
| toy5_threshold_aware_grid_lattice_k6_h0p85_spread | neural_threshold_aware_grid_lattice_k6_h0p85_threshold_anchor | main | pass | 5/5 | 25 | 100 | 1 |  |
| toy5_threshold_aware_grid_lattice_k6_h0p95_spread | neural_threshold_aware_grid_lattice_k6_h0p95_output_average | baseline | diagnostic_only | 0/5 |  | 1 | 0 | seed_level_final_miss |
| toy5_threshold_aware_grid_lattice_k6_h0p95_spread | neural_threshold_aware_grid_lattice_k6_h0p95_exposure_anchor | diagnostic | diagnostic_only | 5/5 | 23 | 100 | 1 |  |
| toy5_threshold_aware_grid_lattice_k6_h0p95_spread | neural_threshold_aware_grid_lattice_k6_h0p95_threshold_anchor | main | pass | 5/5 | 25 | 100 | 1 |  |
| toy5_threshold_aware_grid_rewired_k6_p0p10_h0p85_spread | neural_threshold_aware_grid_rewired_k6_p0p10_h0p85_output_average | baseline | diagnostic_only | 0/5 |  | 1 | 0 | seed_level_final_miss |
| toy5_threshold_aware_grid_rewired_k6_p0p10_h0p85_spread | neural_threshold_aware_grid_rewired_k6_p0p10_h0p85_exposure_anchor | diagnostic | diagnostic_only | 5/5 | 9.4 | 100 | 1 |  |
| toy5_threshold_aware_grid_rewired_k6_p0p10_h0p85_spread | neural_threshold_aware_grid_rewired_k6_p0p10_h0p85_threshold_anchor | main | pass | 5/5 | 9.6 | 100 | 1 |  |
| toy5_threshold_aware_grid_rewired_k6_p0p10_h0p95_spread | neural_threshold_aware_grid_rewired_k6_p0p10_h0p95_output_average | baseline | diagnostic_only | 0/5 |  | 1 | 0 | seed_level_final_miss |
| toy5_threshold_aware_grid_rewired_k6_p0p10_h0p95_spread | neural_threshold_aware_grid_rewired_k6_p0p10_h0p95_exposure_anchor | diagnostic | diagnostic_only | 5/5 | 9.4 | 100 | 1 |  |
| toy5_threshold_aware_grid_rewired_k6_p0p10_h0p95_spread | neural_threshold_aware_grid_rewired_k6_p0p10_h0p95_threshold_anchor | main | pass | 5/5 | 10 | 100 | 1 |  |
