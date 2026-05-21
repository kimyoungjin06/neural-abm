# Toy5 Neural Threshold-Target Lattice Wavefront Findings

## Scope

This run targets the low-mixing lattice boundary found in the structural
robustness probe. The mechanism change is narrow: readiness propagation can
aggregate peer readiness by `max` instead of the default `mean`.

Artifacts:

- Manifest: `experiments/evidence/toy5_neural_threshold_target_lattice_wavefront_quick.yaml`
- Gate summary: `experiments/evidence/results/toy5_neural_threshold_target_lattice_wavefront_quick.summary.md`
- Run rows: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_lattice_wavefront_quick_runs.csv`
- Effect report: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_lattice_wavefront_quick_effects.md`

## Gate Result

| Case | Main variant | Final ceiling hits | Mean TtC | Metric mean | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `toy5_threshold_target_wavefront_no_seed_safety` | `neural_threshold_target_wavefront_no_seed_exposure_anchor` | 5/5 | 0.0 | 1.0 non-adoption | 1.0 |
| `toy5_threshold_target_lattice_wavefront_spread` | `neural_threshold_target_lattice_max_wavefront_anchor` | 5/5 | 23.0 | 100 cascade size | 1.0 |

Overall gate status: pass.

## Interpretation

The previous lattice boundary was a wavefront-speed problem. With mean peer
readiness aggregation, each frontier agent only receives a degree-normalized
fraction of the ready neighborhood. The run therefore reaches a stable partial
cascade of 65/100 by epoch 50.

The `max` aggregation treats exposure to any ready peer as a full local
wavefront signal. Under the same lattice, same policy prior, same
threshold-target learning, and same exposure-anchored direction source, this
reaches 100/100 in 5/5 seeds with mean TtC 23.0.

No-seed safety is preserved. The max-aggregation exposure anchor reaches 5/5
safety hits with no forced adoption, while the non-directional readiness
diagnostic still fails safety with 0/5 hits. This keeps the distinction between
real exposure propagation and self-excitation.

## Claim Boundary

This supports a more structural Toy5 claim: low-mixing lattice spread needs a
wavefront-style readiness aggregation, not just more epochs. The new aggregation
is opt-in and default behavior remains `mean`, so existing evidence semantics
are preserved.

This is still not a universal topology claim. It validates a single
Watts-Strogatz lattice boundary (`rewire_probability=0.0`, `k=6`) under the
current calibrated threshold-target settings.
