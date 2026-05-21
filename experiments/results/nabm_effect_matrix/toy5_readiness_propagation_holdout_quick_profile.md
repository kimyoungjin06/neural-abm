# Evidence Profile: toy5_readiness_propagation_holdout_quick

## Inputs

- Manifest: `experiments/evidence/toy5_readiness_propagation_holdout_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy5_readiness_propagation_holdout_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy5_readiness_propagation_holdout_quick.summary.json`

## Overview

- Gate status: `pass`
- Passed: `True`

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy5_readiness_propagation_holdout | toy5 | pass | neural_readiness_propagation_w1p0 | 3/3 | 1.33333 | 100 |  |  |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy5_readiness_propagation_holdout | complex_threshold | diagnostic | diagnostic_only | 1/3 | 15 | 34 | 0.333333 | seed_level_final_miss |
| toy5_readiness_propagation_holdout | neural_output_average | baseline | diagnostic_only | 3/3 | 1.33333 | 100 | 1 |  |
| toy5_readiness_propagation_holdout | neural_precommitment_evidence | diagnostic | diagnostic_only | 3/3 | 1.33333 | 100 | 1 |  |
| toy5_readiness_propagation_holdout | neural_readiness_propagation_w1p0 | main | pass | 3/3 | 1.33333 | 100 | 1 |  |
