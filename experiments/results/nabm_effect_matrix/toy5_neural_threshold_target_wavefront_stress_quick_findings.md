# Toy5 Wavefront Stress Quick Findings

## Scope

This run attempted to find a failure boundary for the opt-in max-readiness
wavefront mechanism by combining low-mixing topology with stronger threshold
heterogeneity.

Artifacts:

- Manifest: `experiments/evidence/toy5_neural_threshold_target_wavefront_stress_quick.yaml`
- Gate summary: `experiments/evidence/results/toy5_neural_threshold_target_wavefront_stress_quick.summary.md`
- Run rows: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_wavefront_stress_quick_runs.csv`
- Effect report: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_wavefront_stress_quick_effects.md`

## Gate Result

Overall gate status: pass.

| Case | Main variant | Final ceiling hits | Mean TtC | Metric mean | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `toy5_threshold_target_wavefront_stress_no_seed_heterogeneous_safety` | `neural_threshold_target_stress_no_seed_heterogeneous_exposure_anchor` | 5/5 | 0.0 | 1.0 non-adoption | 1.0 |
| `toy5_threshold_target_lattice_k4_heterogeneous_h0p85_wavefront_spread` | `neural_threshold_target_lattice_k4_heterogeneous_h0p85_max_wavefront_anchor` | 5/5 | 33.0 | 100 cascade size | 1.0 |
| `toy5_threshold_target_lattice_k6_heterogeneous_h0p95_wavefront_spread` | `neural_threshold_target_lattice_k6_heterogeneous_h0p95_max_wavefront_anchor` | 5/5 | 23.0 | 100 cascade size | 1.0 |
| `toy5_threshold_target_rewired_p0p10_heterogeneous_h0p95_wavefront_spread` | `neural_threshold_target_rewired_p0p10_heterogeneous_h0p95_max_wavefront_anchor` | 5/5 | 9.4 | 100 cascade size | 1.0 |

## Diagnostics

Mean aggregation still stalls under the hard lattice slices:

| Case | Mean diagnostic final hits | Mean diagnostic metric mean | Max main final hits | Max main metric mean |
| --- | ---: | ---: | ---: | ---: |
| `toy5_threshold_target_lattice_k4_heterogeneous_h0p85_wavefront_spread` | 0/5 | 49 | 5/5 | 100 |
| `toy5_threshold_target_lattice_k6_heterogeneous_h0p95_wavefront_spread` | 0/5 | 65 | 5/5 | 100 |
| `toy5_threshold_target_rewired_p0p10_heterogeneous_h0p95_wavefront_spread` | 5/5 | 100 | 5/5 | 100 |

No-seed safety remains separated from self-excitation: exposure-anchored max
readiness has 5/5 safety hits, while the non-directional max-readiness
diagnostic has 0/5 safety hits.

## Interpretation

The run did not find a failure boundary for the current exposure-anchored
wavefront. Instead, it exposed a more precise modeling issue: the current
`readiness_exposure_with_action_anchor` direction source is intentionally
exposure-based and does not subtract each agent's adoption threshold. Raising
`heterogeneous_threshold_high` therefore does not stress the direction source as
strongly as the manifest name suggests.

The result is still useful. It confirms that max-readiness propagation remains
stable under these harder slices, but the threshold-heterogeneity claim is
carried by the follow-up threshold-aware manifest, not by this exposure-source
stress run.

## Claim Boundary

This run should not be used to claim threshold-heterogeneous robustness. It is
evidence for exposure-wavefront robustness, not threshold-aware adoption
semantics.

The follow-up threshold-aware manifest compares this exposure source against
`readiness_augmented_threshold_with_action_anchor`, which keeps the action
anchor but subtracts each agent's threshold in the direction score.
