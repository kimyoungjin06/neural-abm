# Toy2/Toy4 Revision Operator Quick Findings

Manifest: `experiments/evidence/toy24_revision_operator_quick.yaml`

Run summary:

- The first workflow run was interrupted before matrix CSV/gate artifacts were
  written.
- Resume reused 19 complete run artifacts and reran 11 missing or partial Toy4
  runs.
- Outputs:
  - `experiments/results/nabm_effect_matrix/toy24_revision_operator_quick_runs.csv`
  - `experiments/results/nabm_effect_matrix/toy24_revision_operator_quick_effects.md`
  - `experiments/evidence/results/toy24_revision_operator_quick.summary.md`
- The matrix/gate artifacts were refreshed after the initial run to add
  post-ceiling instability diagnostics without rerunning the simulations.

Gate result: **fail**.

| Case | Main revision variant | Final hits | Ever hits | Ever-final misses | Mean TtC | Terminal ceiling rate | Late flip rate | Metric mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | `revision_operator_mixed_objective_basin_w0p5_0p5_h1` | 2/3 | 3/3 | 1 | 19.33 | 0.867 | 0.00978 | 2.99667 |
| Toy4 | `revision_operator_mixed_objective_basin_w0p5_0p5_h1` | 1/3 | 3/3 | 2 | 19.00 | 0.800 | 0.00976 | 0.59600 |

Diagnostic comparison:

| Case | Variant | Group | Final hits | Mean TtC | Final action rate | Final flip rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Toy2 | `mixed_objective_basin_w0p5_0p5_h1` | diagnostic | 3/3 | 22.67 | 1.000 | 0.000 |
| Toy2 | `revision_operator_mixed_objective_basin_w0p5_0p5_h1` | revision | 2/3 | 19.33 | 0.997 | 0.003 |
| Toy4 | `mixed_objective_basin_w0p5_0p5_h1` | diagnostic | 2/3 | 16.33 | 0.997 | 0.003 |
| Toy4 | `revision_operator_mixed_objective_basin_w0p5_0p5_h1` | revision | 1/3 | 19.00 | 0.993 | 0.007 |

Interpretation:

- The policy-probability-to-revision adapter is structurally connected and
  logged correctly, but it is not yet a successful mechanism.
- The main failure is late instability, not total inability to reach the basin:
  every main revision run reached the ceiling at least once, but Toy2 seed 1
  and Toy4 seeds 1 and 3 ended with one defector at epoch 50.
- Final policy probabilities were already high, roughly 0.995-0.998, but in a
  100-agent sampled revision process that still leaves enough residual switch
  probability for one late flip.
- Therefore the current adapter mostly re-expresses the old action-probability
  semantics. It does not yet add the ABM-native inertia/stay mechanism that a
  real revision operator needs.

Revised next direction:

- Do not add a new revision inertia helper yet. The current evidence is still
  consistent with final-epoch stochastic gate brittleness.
- First run diagnostic controls that separate evaluation brittleness from
  algorithmic failure: existing commitment hysteresis, terminal argmax
  decision mode as a diagnostic-only control, and terminal-window ceiling
  reporting.
- Only if those controls show a real revision-rule gap should a new
  operator-native stay/switch mechanism be introduced. That mechanism must be
  evaluated as a small sweep, not as a single tuned point.
