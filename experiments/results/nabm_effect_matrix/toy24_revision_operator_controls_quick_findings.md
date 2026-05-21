# Toy2/Toy4 Revision Operator Control Findings

Manifest: `experiments/evidence/toy24_revision_operator_controls_quick.yaml`

Purpose:

- Test whether the failed revision-operator gate was mostly a final-epoch
  stochastic brittleness problem before adding a new inertia helper.
- Keep the controls diagnostic-only for interpretation: existing commitment
  hysteresis and terminal argmax are not new NABM mechanisms.

Run artifacts:

- `experiments/results/nabm_effect_matrix/toy24_revision_operator_controls_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy24_revision_operator_controls_quick_effects.md`
- `experiments/evidence/results/toy24_revision_operator_controls_quick.summary.md`

Gate result: **fail**. The failure shifted from final ceiling misses to
time-to-ceiling.

| Case | Variant | Final hits | Ever hits | Ever-final misses | Mean TtC | Terminal ceiling rate | Late flip rate | Metric mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | `revision_operator_mixed_objective_basin_w0p5_0p5_h1` | 2/3 | 3/3 | 1 | 19.33 | 0.867 | 0.00978 | 2.99667 |
| Toy2 | `revision_operator_commitment_hysteresis` | 3/3 | 3/3 | 0 | 15.00 | 1.000 | 0.00056 | 3.00000 |
| Toy2 | `revision_operator_terminal_argmax_k1` | 3/3 | 3/3 | 0 | 19.33 | 0.933 | 0.00967 | 3.00000 |
| Toy2 | `revision_operator_terminal_argmax_k5` | 3/3 | 3/3 | 0 | 19.33 | 1.000 | 0.00948 | 3.00000 |
| Toy4 | `revision_operator_mixed_objective_basin_w0p5_0p5_h1` | 1/3 | 3/3 | 2 | 19.00 | 0.800 | 0.00976 | 0.59600 |
| Toy4 | `revision_operator_commitment_hysteresis` | 3/3 | 3/3 | 0 | 13.67 | 0.933 | 0.00148 | 0.60000 |
| Toy4 | `revision_operator_terminal_argmax_k1` | 3/3 | 3/3 | 0 | 19.00 | 0.933 | 0.00953 | 0.60000 |
| Toy4 | `revision_operator_terminal_argmax_k5` | 3/3 | 3/3 | 0 | 19.00 | 1.000 | 0.00929 | 0.60000 |

Interpretation:

- The original revision operator's final misses are real stochastic terminal
  brittleness: both commitment hysteresis and terminal argmax remove all
  ever-final misses.
- Terminal argmax only repairs the evaluation endpoint. It does not improve
  time-to-ceiling, so it should remain a diagnostic control rather than a claim
  variant.
- Existing commitment hysteresis is the stronger diagnostic control. It removes
  final misses and reduces late flip rate by roughly an order of magnitude, but
  it still misses the current TtC gate: Toy2 15.00 vs required <10, Toy4 13.67
  vs required <12.
- Therefore the next structural problem is no longer just final-epoch hazard.
  The model still commits too late relative to the gate and baseline reference.

Next direction:

- Do not relax the gate yet. The control run shows that final-hit success can be
  manufactured by endpoint decisions, but that does not solve early basin entry.
- Treat commitment hysteresis as a useful diagnostic boundary: a real
  revision-rule improvement should match or improve its final stability while
  reducing TtC.
- The next mechanism should target the pre-ceiling transition, not only
  post-ceiling stay probability. A reasonable next slice is an operator-native
  confidence/stability sweep that is evaluated against both TtC and
  ever-final-miss metrics.
