# Toy5 Threshold-Aware Wavefront Quick Findings

## Scope

This run tests whether the Toy5 wavefront mechanism still works when the
precommitment direction source is threshold-aware. It compares the current
exposure source against `readiness_augmented_threshold_with_action_anchor`,
which keeps the action anchor but subtracts each agent's adoption threshold in
the direction score.

The run also validates a structural fix: Toy5 direction scoring now uses the
same configured peer-readiness aggregation as readiness propagation. Before
that fix, readiness evidence could use `max` while the direction score still
used mean peer readiness, creating a split-brain wavefront semantics.

Artifacts:

- Manifest: `experiments/evidence/toy5_neural_threshold_target_threshold_aware_wavefront_quick.yaml`
- Gate summary: `experiments/evidence/results/toy5_neural_threshold_target_threshold_aware_wavefront_quick.summary.md`
- Run rows: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_threshold_aware_wavefront_quick_runs.csv`
- Effect report: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_threshold_aware_wavefront_quick_effects.md`

## Gate Result

Overall gate status: pass.

| Case | Threshold-aware main final hits | Mean TtC | Metric mean | Exposure diagnostic final hits | Exposure diagnostic TtC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `toy5_threshold_aware_wavefront_no_seed_heterogeneous_safety` | 5/5 | 0.0 | 1.0 non-adoption | n/a | n/a |
| `toy5_threshold_aware_lattice_k4_heterogeneous_h0p85_spread` | 5/5 | 36.2 | 100 cascade size | 5/5 | 33.0 |
| `toy5_threshold_aware_lattice_k6_heterogeneous_h0p95_spread` | 5/5 | 25.0 | 100 cascade size | 5/5 | 23.0 |
| `toy5_threshold_aware_rewired_p0p10_heterogeneous_h0p95_spread` | 5/5 | 10.0 | 100 cascade size | 5/5 | 9.4 |

No-seed safety remains intact for the threshold-aware source. The
non-directional diagnostic still fails safety, so the direction gate is doing
real work.

## Interpretation

The previous threshold-aware failure was not evidence that threshold semantics
were incompatible with the wavefront mechanism. It exposed a wiring mismatch:
readiness propagation honored `precommitment_peer_readiness_aggregation=max`,
but Toy5 direction scores were still computed from mean peer readiness.

After aligning direction scoring with the configured readiness aggregation, the
threshold-aware path clears all tested stress slices. The threshold-aware
variant remains slightly slower than exposure-only in lattice cases, which is
expected because it subtracts each agent's threshold instead of accepting any
exposure as sufficient direction.

This makes the Toy5 mechanism cleaner: direction and evidence now share the
same local wavefront semantics, while no-seed safety still blocks
self-excitation.

## Claim Boundary

This supports a stronger but still bounded claim: for the tested
Watts-Strogatz slices, max-readiness wavefront semantics can be made
threshold-aware without collapsing cascade spread or no-seed safety.

The claim is not universal. The tested grid is still small: `k=4` with
`high=0.85`, `k=6` with `high=0.95`, and `p=0.10` with `high=0.95`. Broader
claims need a larger topology/threshold grid and preferably seed-neighborhood
diagnostics for cases where initial ignition is marginal.
