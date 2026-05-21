# Evidence Profile: toy24_precommitment_peer_evidence_closure_quick

## Inputs

- Manifest: `experiments/evidence/toy24_precommitment_peer_evidence_closure_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_closure_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy24_precommitment_peer_evidence_closure_quick.summary.json`

## Overview

- Gate status: `pass`
- Passed: `True`
- Notes: toy24_objective_basin_evidence, toy24_revision_operator_evidence

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy2_precommitment_peer_evidence_closure | toy2 | pass | revision_precommitment_peer_evidence_w1p0 | 5/5 | 9.4 | 3 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path |
| toy4_precommitment_peer_evidence_closure | toy4 | pass | revision_precommitment_peer_evidence_w1p0 | 5/5 | 9 | 0.6 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path, toy24_late_flip_hazard |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy2_precommitment_peer_evidence_closure | reputation_imitation | baseline | diagnostic_only | 5/5 | 2.6 | 3 | 1 |  |
| toy2_precommitment_peer_evidence_closure | objective_basin_w0p5_0p5_h1 | diagnostic | diagnostic_only | 4/5 | 20.8 | 2.998 | 0.84 | seed_level_final_miss |
| toy2_precommitment_peer_evidence_closure | revision_objective_basin_w0p5_0p5_h1 | diagnostic | diagnostic_only | 4/5 | 21 | 2.998 | 0.76 | seed_level_final_miss |
| toy2_precommitment_peer_evidence_closure | revision_precommitment_evidence | diagnostic | diagnostic_only | 5/5 | 10.8 | 3 | 1 |  |
| toy2_precommitment_peer_evidence_closure | revision_precommitment_peer_evidence_w1p0 | main | pass | 5/5 | 9.4 | 3 | 1 |  |
| toy4_precommitment_peer_evidence_closure | reputation_imitation | baseline | diagnostic_only | 5/5 | 2.6 | 0.6 | 1 |  |
| toy4_precommitment_peer_evidence_closure | objective_basin_w0p5_0p5_h1 | diagnostic | diagnostic_only | 4/5 | 19 | 0.5988 | 0.76 | seed_level_final_miss |
| toy4_precommitment_peer_evidence_closure | revision_objective_basin_w0p5_0p5_h1 | diagnostic | diagnostic_only | 3/5 | 20.6 | 0.5976 | 0.72 | seed_level_final_miss |
| toy4_precommitment_peer_evidence_closure | revision_precommitment_evidence | diagnostic | diagnostic_only | 5/5 | 10.6 | 0.6 | 1 |  |
| toy4_precommitment_peer_evidence_closure | revision_precommitment_peer_evidence_w1p0 | main | pass | 5/5 | 9 | 0.6 | 1 |  |
