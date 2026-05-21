# Toy5 Threshold-Aware Grid Findings

Date: 2026-05-21

## Run

- Manifest:
  `experiments/evidence/toy5_neural_threshold_target_threshold_aware_grid_quick.yaml`
- Runs:
  `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_threshold_aware_grid_quick_runs.csv`
- Gate summary:
  `experiments/evidence/results/toy5_neural_threshold_target_threshold_aware_grid_quick.summary.md`
- Profile:
  `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_threshold_aware_grid_quick_profile.md`

Command:

```bash
uv run python scripts/run_basin_credit_evidence_workflow.py \
  --manifest experiments/evidence/toy5_neural_threshold_target_threshold_aware_grid_quick.yaml
```

Gate status: `pass`.

## Case Results

| Case | Baseline final hits | Negative-control hits | Main final hits | Main mean TtC | Main metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-seed heterogeneous safety | 5/5 | 0/5 | 5/5 | 0.0 | 1.0 non-adoption |
| lattice k4 h0.85 spread | 0/5 | 5/5 | 5/5 | 36.2 | 100 cascade size |
| lattice k4 h0.95 spread | 0/5 | 5/5 | 5/5 | 37.0 | 100 cascade size |
| lattice k6 h0.85 spread | 0/5 | 5/5 | 5/5 | 25.0 | 100 cascade size |
| lattice k6 h0.95 spread | 0/5 | 5/5 | 5/5 | 25.0 | 100 cascade size |
| rewired k6 p0.10 h0.85 spread | 0/5 | 5/5 | 5/5 | 9.6 | 100 cascade size |
| rewired k6 p0.10 h0.95 spread | 0/5 | 5/5 | 5/5 | 10.0 | 100 cascade size |

## Interpretation

This completes the first Gate 2 expansion from a single Toy5 hard holdout to a
small topology/threshold grid.

Supported:

- The threshold-aware main path is robust on the tested grid:
  lattice `k=4`, lattice `k=6`, and rewired `k=6, p=0.10`, each at
  heterogeneous high thresholds `0.85` and `0.95`.
- The no-seed safety case remains protected: output-average and threshold-aware
  main both preserve non-adoption, while the non-directional negative control
  self-excites and fails safety.
- The output-average baseline remains unable to spread from the single seeded
  high-threshold cases: `0/5` final ceiling hits in all six spread cases.

Important caveat:

- The exposure-anchor negative control also reaches `5/5` final ceiling hits in
  every spread case and is slightly faster than the threshold-aware main.
- Therefore this artifact supports threshold-aware robustness and safety
  separation, but it does not prove that subtracting threshold is necessary for
  spread once exposure-anchored direction is available.

Bounded claim:

> In the tested Toy5 grid, the shared NABM unit lifecycle can carry a
> threshold-aware readiness adapter that preserves no-seed safety and recovers
> full cascades where output averaging stalls. The stronger claim that
> threshold-aware direction is uniquely required for seeded spread is not
> supported by this grid because the exposure-anchor negative control also
> spreads.

## Next Step

Do not expand this by only adding more thresholds. The next useful step is to
make the negative-control distinction sharper:

- add cases where exposure alone is expected to over-spread or self-excite;
- keep no-seed and sparse-seed safety as primary guards;
- report safety and spread separately rather than compressing them into a
  single pass/fail gate.
