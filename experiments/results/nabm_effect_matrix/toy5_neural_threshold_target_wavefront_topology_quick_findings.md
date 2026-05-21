# Toy5 Wavefront Topology Quick Findings

## Scope

This run expands the previous single-lattice wavefront check without changing
the mechanism. The tested mechanism is still opt-in peer readiness aggregation:
`precommitment_peer_readiness_aggregation=max`.

The goal is to test whether the earlier `k=6`, `rewire_probability=0.0`
success was a single topology artifact. The manifest keeps the same
threshold-target learner, exposure-anchored direction source, policy prior, and
precommitment settings while varying only graph degree and light rewiring.

Artifacts:

- Manifest: `experiments/evidence/toy5_neural_threshold_target_wavefront_topology_quick.yaml`
- Gate summary: `experiments/evidence/results/toy5_neural_threshold_target_wavefront_topology_quick.summary.md`
- Run rows: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_wavefront_topology_quick_runs.csv`
- Effect report: `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_wavefront_topology_quick_effects.md`

## Gate Result

Overall gate status: pass.

| Case | Main variant | Final ceiling hits | Mean TtC | Metric mean | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `toy5_threshold_target_wavefront_topology_no_seed_safety` | `neural_threshold_target_topology_no_seed_exposure_anchor` | 5/5 | 0.0 | 1.0 non-adoption | 1.0 |
| `toy5_threshold_target_lattice_k4_wavefront_spread` | `neural_threshold_target_lattice_k4_max_wavefront_anchor` | 5/5 | 33.0 | 100 cascade size | 1.0 |
| `toy5_threshold_target_lattice_k8_wavefront_spread` | `neural_threshold_target_lattice_k8_max_wavefront_anchor` | 5/5 | 18.0 | 100 cascade size | 1.0 |
| `toy5_threshold_target_rewired_p0p02_wavefront_spread` | `neural_threshold_target_rewired_p0p02_max_wavefront_anchor` | 5/5 | 14.6 | 100 cascade size | 1.0 |

## Diagnostics

The mean-aggregation diagnostics still expose the frontier-speed boundary:

| Case | Mean diagnostic final hits | Mean diagnostic metric mean | Max main final hits | Max main metric mean |
| --- | ---: | ---: | ---: | ---: |
| `toy5_threshold_target_lattice_k4_wavefront_spread` | 0/5 | 49 | 5/5 | 100 |
| `toy5_threshold_target_lattice_k8_wavefront_spread` | 0/5 | 83 | 5/5 | 100 |
| `toy5_threshold_target_rewired_p0p02_wavefront_spread` | 4/5 | 95.6 | 5/5 | 100 |

No-seed safety remains separated from self-excitation. The exposure-anchored
main variant has 5/5 safety hits, while the non-directional max-readiness
diagnostic has 0/5 safety hits.

## Interpretation

The topology pattern matches a wavefront-speed interpretation. Lower degree
slows propagation (`k=4`, TtC 33.0), higher degree accelerates it (`k=8`, TtC
18.0), and light rewiring accelerates it further (`p=0.02`, TtC 14.6). The
effect is not caused by changing the neural learner or by adding epochs; the
same local learning and direction source are held fixed.

The key mechanism difference is how peer readiness is represented at the
frontier. Mean aggregation degree-normalizes sparse frontier contact and leaves
partial cascades. Max aggregation treats any ready neighbor as a full local
wavefront exposure signal, which is the intended contagion semantics for this
Toy5 boundary.

## Claim Boundary

This supports a limited structural claim: for the tested low-mixing
Watts-Strogatz slices, threshold-target Toy5 needs a wavefront-style readiness
operator to avoid mean-aggregation frontier stalls.

This is not yet a universal topology robustness claim. The run covers
`k in {4, 8}` at `rewire_probability=0.0` plus `k=6` at
`rewire_probability=0.02`, all with homogeneous threshold `0.75` and
first-agent seeding. Broader claims need a larger topology grid and explicit
threshold heterogeneity coverage.
