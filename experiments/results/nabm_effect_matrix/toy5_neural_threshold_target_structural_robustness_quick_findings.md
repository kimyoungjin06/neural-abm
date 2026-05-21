# Toy5 Neural Threshold-Target Structural Robustness Findings

## Scope

This run expands the Toy5 exposure-anchored threshold-target claim beyond the
original homogeneous first-seed frontier. It keeps the mechanism fixed and
varies structural conditions:

- no-seed safety under heterogeneous thresholds;
- seeded frontier spread with random seed placement;
- seeded frontier spread under heterogeneous thresholds.

Artifacts:

- Manifest: `experiments/evidence/toy5_neural_threshold_target_structural_robustness_quick.yaml`
- Gate summary: `experiments/evidence/results/toy5_neural_threshold_target_structural_robustness_quick.summary.md`
- Run rows: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_structural_robustness_quick_runs.csv`
- Effect report: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_structural_robustness_quick_effects.md`
- Boundary probe rows: `experiments/results/nabm_effect_matrix/toy5_threshold_target_robustness_probe_runs.csv`

## Gate Result

| Case | Main variant | Final ceiling hits | Mean TtC | Metric mean | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `toy5_threshold_target_no_seed_heterogeneous_safety` | `neural_threshold_target_no_seed_heterogeneous_exposure_anchor` | 5/5 | 0.0 | 1.0 non-adoption | 1.0 |
| `toy5_threshold_target_random_seed_frontier_spread` | `neural_threshold_target_random_seed_frontier_exposure_anchor` | 5/5 | 32.6 | 100 cascade size | 1.0 |
| `toy5_threshold_target_heterogeneous_frontier_spread` | `neural_threshold_target_heterogeneous_frontier_exposure_anchor` | 5/5 | 31.0 | 100 cascade size | 1.0 |

Overall gate status: pass.

## Interpretation

The exposure-anchored direction mechanism is not only solving the original
first-agent homogeneous frontier. It also preserves safety when thresholds are
heterogeneous and no real adopter exists, and it still spreads when the initial
seed is randomly placed or when frontier thresholds are heterogeneous.

The no-seed diagnostic remains important. Non-directional readiness propagation
fails heterogeneous no-seed safety with 0/5 safety hits, showing that the useful
component is not generic readiness amplification. The exposure anchor blocks the
self-excitation path by keeping direction scores negative when no adopter or
readiness exposure exists.

The frontier diagnostics remain consistent with the earlier combined run:
plain output averaging stays at the initial seed, while exposure-anchored
readiness propagation reaches full cascade in every tested seed.

## Boundary

A small boundary probe found that the mean-aggregation mechanism does not solve
the lattice-like frontier with `domain.graph.rewire_probability=0.0`: 0/3 seeds
reached ceiling and the mean final cascade was 65.0. This is not in this main
gate, because it defines a separate topology-specific mechanism question.

This means the current claim can cover random seed placement and threshold
heterogeneity under the default mean-readiness aggregation, but not arbitrary
topology. The lattice-specific follow-up is now captured by the
`toy5_neural_threshold_target_lattice_wavefront_quick` evidence manifest, where
max-readiness aggregation acts as an explicit spatial wavefront mechanism.
