# Evidence Profile: toy24_revision_operator_precommitment_peer_evidence_stability_quick

## Inputs

- Manifest: `experiments/evidence/toy24_revision_operator_precommitment_peer_evidence_stability_quick.yaml`
- Runs: `experiments/results/nabm_effect_matrix/toy24_revision_operator_precommitment_peer_evidence_stability_quick_runs.csv`
- Gate summary: `experiments/evidence/results/toy24_revision_operator_precommitment_peer_evidence_stability_quick.summary.json`

## Overview

- Gate status: `pass`
- Passed: `True`
- Notes: toy24_objective_basin_evidence, toy24_revision_operator_evidence

## Case Summary

| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| toy2_revision_operator_precommitment_peer_evidence_stability | toy2 | pass | revision_operator_precommitment_peer_evidence_w1p0 | 10/10 | 9.5 | 3 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path |
| toy4_revision_operator_precommitment_peer_evidence_stability | toy4 | pass | revision_operator_precommitment_peer_evidence_w1p0 | 10/10 | 8.9 | 0.6 |  | toy24_basin_credit_evidence, toy24_objective_basin_blend, toy24_revision_operator_path, toy24_late_flip_hazard |

## Variant Details

| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| toy2_revision_operator_precommitment_peer_evidence_stability | reputation_imitation | baseline | diagnostic_only | 10/10 | 2.7 | 3 | 1 |  |
| toy2_revision_operator_precommitment_peer_evidence_stability | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | diagnostic | diagnostic_only | 9/10 | 20.4 | 2.999 | 0.76 | seed_level_final_miss |
| toy2_revision_operator_precommitment_peer_evidence_stability | revision_operator_precommitment_evidence | diagnostic | diagnostic_only | 10/10 | 10.9 | 3 | 1 |  |
| toy2_revision_operator_precommitment_peer_evidence_stability | revision_operator_precommitment_peer_evidence_w1p0 | main | pass | 10/10 | 9.5 | 3 | 1 |  |
| toy4_revision_operator_precommitment_peer_evidence_stability | reputation_imitation | baseline | diagnostic_only | 10/10 | 2.7 | 0.6 | 1 |  |
| toy4_revision_operator_precommitment_peer_evidence_stability | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | diagnostic | diagnostic_only | 8/10 | 19.5 | 0.5988 | 0.74 | seed_level_final_miss |
| toy4_revision_operator_precommitment_peer_evidence_stability | revision_operator_precommitment_evidence | diagnostic | diagnostic_only | 10/10 | 10.4 | 0.6 | 1 |  |
| toy4_revision_operator_precommitment_peer_evidence_stability | revision_operator_precommitment_peer_evidence_w1p0 | main | pass | 10/10 | 8.9 | 0.6 | 1 |  |
