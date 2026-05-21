# Toy 1 Current Findings

This document consolidates the current Toy 1 evidence so the next experiments
can build on a single interpretation rather than scattered run summaries.

Toy model:

`Neural HK Classification`

Core setup:

- 50 neural agents.
- Biased private data shards.
- MLP classifier `2 -> 16 -> 2`.
- Static Watts-Strogatz candidate graph.
- Synchronous updates.
- Shared probe set for social comparison and metrics.

## Research Questions

The current Toy 1 experiments test four questions:

1. Does explicit social mixing improve task performance over independent local
   learning?
2. Does social mixing increase prediction consensus?
3. Do output, latent, and parameter paths behave differently?
4. Is parameter averaging sensitive to initialization and peer-threshold choice?

## Executed Experiments

| Experiment | Seeds | Purpose | Summary |
| --- | ---: | --- | --- |
| `toy1_neural_hk_baseline` | 1 | Validate runner and logs. | [summary](toy1_neural_hk_baseline_seed01_summary.md) |
| `toy1_first_ablation_seeds01_05` | 5 | Compare no-social, output, latent, and parameter paths. | [grouped summary](toy1_first_ablation_seeds01_05_grouped_summary.md) |
| `toy1_alpha_threshold_sweep_seed01` | 1 | Pilot alpha/threshold phase behavior. | [grouped summary](toy1_alpha_threshold_sweep_seed01_grouped_summary.md) |
| `toy1_param_independent_low_threshold_seed01` | 1 | Diagnose independent-init parameter threshold sensitivity. | [grouped summary](toy1_param_independent_low_threshold_seed01_grouped_summary.md) |
| `toy1_param_independent_low_threshold_seeds01_05` | 5 | Check whether the independent-init parameter threshold transition is stable across seeds. | [grouped summary](toy1_param_independent_low_threshold_seeds01_05_grouped_summary.md) |
| `toy1_output_alpha_sweep_seeds01_05` | 5 | Check stable output-mixing alpha response. | [grouped summary](toy1_output_alpha_sweep_seeds01_05_grouped_summary.md) |
| `toy1_param_alignment_diagnostic_seeds01_05` | 5 | Separate parameter averaging alignment from peer-selection alignment. | [summary](toy1_parameter_alignment_diagnostic_summary.md) |
| `toy1_parameter_cluster_compare_seed01` | 1 | Visualize connected, partial, and fragmented parameter-path regimes over time. | [summary](toy1_parameter_cluster_comparison_summary.md) |

## Finding 1: Output Social Mixing Gives a Stable Small Task Gain

The clearest current result is the 5-seed output-average alpha sweep.

Source:

`experiments/results/toy1_output_alpha_sweep_seeds01_05_grouped_summary.csv`

| Alpha | Accuracy Mean | Consensus Mean | Fragmentation Mean |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.888605 | 0.946704 | 1.00 |
| 0.10 | 0.890332 | 0.955479 | 1.00 |
| 0.25 | 0.891126 | 0.959370 | 1.00 |
| 0.50 | 0.891625 | 0.963434 | 1.00 |

Interpretation:

- Increasing `alpha` monotonically increases mean accuracy and mean consensus.
- The effect is modest for accuracy but clear for consensus.
- Fragmentation remains connected, so the change is not driven by graph
  connectivity changes.
- This is currently the strongest first paper figure candidate.

Figure:

`paper/figures/toy1_output_alpha_accuracy_consensus.png`

## Finding 2: Mixer Paths Are Not Equivalent

The 5-seed first ablation shows different behavior across mixer families.

Source:

`experiments/results/toy1_first_ablation_seeds01_05_grouped_summary.csv`

| Case | Accuracy Mean | Consensus Mean | Fragmentation Mean |
| --- | ---: | ---: | ---: |
| `none_none_same_init` | 0.887050 | 0.945778 | 50.00 |
| `output_average_output_similarity_same_init` | 0.891126 | 0.959370 | 1.00 |
| `latent_average_state_similarity_same_init` | 0.889104 | 0.950259 | 1.00 |
| `parameter_average_state_similarity_same_init` | 0.893134 | 0.976601 | 1.00 |
| `parameter_average_state_similarity_independent_init` | 0.887794 | 0.947597 | 50.00 |

Interpretation:

- Output averaging is more effective than latent averaging in the current Toy 1
  setup.
- Parameter averaging with same initialization is strongest in this small,
  homogeneous setting.
- Parameter averaging with independent initialization fails under the original
  `0.8` state-similarity threshold because the filtered peer graph fragments.

Figure:

`paper/figures/toy1_mixer_comparison.png`

## Finding 3: Alpha Is Currently the Cleanest Influence Knob

The seed-1 alpha/threshold sweep shows that same-init social mixers are mostly
insensitive to thresholds `0.6`, `0.8`, and `0.95`.

Source:

`experiments/results/toy1_alpha_threshold_sweep_seed01_grouped_summary.csv`

Interpretation:

- For same-init output, latent, and parameter paths, the peer graph remains
  connected across the tested thresholds.
- Increasing `alpha` consistently raises consensus and usually improves
  accuracy.
- The current threshold range is too narrow or too high-level to reveal a phase
  transition for same-init agents.

## Finding 4: Independent-Init Parameter Averaging Has a Sharp Threshold Transition

The 5-seed low-threshold parameter diagnostic shows that independent-init
parameter averaging can work, but only if thresholding permits enough peers.

Source:

`experiments/results/toy1_param_independent_low_threshold_seeds01_05_grouped_summary.csv`

Selected rows:

| Alpha | Threshold | Accuracy Mean | Consensus Mean | Fragmentation Mean |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | -0.2 | 0.892609 | 0.976626 | 1.00 |
| 0.25 | 0.0 | 0.892311 | 0.974288 | 1.00 |
| 0.25 | 0.2 | 0.890282 | 0.958260 | 26.00 |
| 0.25 | 0.4 | 0.887877 | 0.948572 | 48.40 |
| 0.25 | 0.6 | 0.887794 | 0.947597 | 50.00 |

Interpretation:

- Independent initialization does not make parameter averaging impossible.
- Raw parameter cosine similarity needs a much lower threshold than same-init
  settings.
- The transition is stable across seeds: thresholds `-0.2` and `0.0` remain
  connected, `0.2` becomes partially fragmented, and `0.4` or higher is near
  independent-agent behavior.
- This motivates separating the averaging operator from the peer-selection
  similarity before making broad claims about parameter-level social learning.

## Finding 5: Parameter Alignment Mainly Fixes Peer Selection

The parameter alignment diagnostic compares three independent-init paths at
`alpha=0.25`: raw averaging with raw parameter peers, aligned averaging with raw
parameter peers, and aligned averaging with aligned parameter peers.

Source:

`experiments/results/toy1_param_alignment_diagnostic_seeds01_05_grouped_summary.csv`

Selected rows:

| Method | Threshold | Accuracy Mean | Consensus Mean | Fragmentation Mean |
| --- | ---: | ---: | ---: | ---: |
| Raw parameter average + raw peers | 0.2 | 0.890282 | 0.958260 | 26.00 |
| Aligned parameter average + raw peers | 0.2 | 0.890865 | 0.962082 | 26.80 |
| Aligned parameter average + aligned peers | 0.2 | 0.893686 | 0.983375 | 1.00 |
| Raw parameter average + raw peers | 0.6 | 0.887794 | 0.947597 | 50.00 |
| Aligned parameter average + raw peers | 0.6 | 0.887794 | 0.947597 | 50.00 |
| Aligned parameter average + aligned peers | 0.6 | 0.893695 | 0.984123 | 1.00 |
| Aligned parameter average + aligned peers | 0.8 | 0.892145 | 0.972957 | 9.80 |

Interpretation:

- Hidden-unit alignment before averaging helps slightly when raw peer selection
  already connects agents.
- It does not repair fragmentation when the peer graph is still built from raw
  parameter cosine similarity.
- Aligned parameter similarity shifts the fragmentation transition upward:
  thresholds `0.2` and `0.6` remain connected, and `0.8` becomes only partially
  fragmented.
- The independent-init parameter path failure is therefore partly a
  representation-comparison problem, not only an averaging problem.

Figure:

`paper/figures/toy1_parameter_alignment_diagnostic.png`

## Finding 6: Fragmentation Has a Visible Output-Divergence Signature

The cluster comparison run visualizes three independent-init parameter-path
regimes at `alpha=0.25`: connected (`threshold=0.0`), partially fragmented
(`threshold=0.2`), and fragmented (`threshold=0.6`).

Source:

`experiments/results/toy1_parameter_cluster_compare_seed01_summary.csv`

Selected rows:

| Threshold | Final Accuracy | Final Consensus | Final Fragmentation |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.897956 | 0.979239 | 1 |
| 0.2 | 0.894148 | 0.959692 | 29 |
| 0.6 | 0.891972 | 0.949434 | 50 |

Interpretation:

- The connected regime keeps output divergence to the population mean low.
- The partial regime shows persistent peer components and rising per-agent
  output divergence.
- The fragmented regime stays near 50 peer components and behaves closer to
  independent local learning.
- This is a single-seed visualization of the 5-seed phase result, not a
  separate robustness claim.

Figure:

`paper/figures/toy1_parameter_cluster_comparison.png`

## Current Figure Candidates

| Figure | Status | Message |
| --- | --- | --- |
| `toy1_output_alpha_accuracy_consensus.png` | Created | Output social mixing gives a monotonic alpha response. |
| `toy1_mixer_comparison.png` | Created | Output, latent, and parameter paths are behaviorally different. |
| `toy1_parameter_independent_phase_diagram.png` | Created | Independent-init parameter averaging has a connectivity threshold transition. |
| `toy1_parameter_independent_phase_diagram_seeds01_05.png` | Created | The independent-init parameter threshold transition persists over 5 seeds. |
| `toy1_parameter_alignment_diagnostic.png` | Created | Hidden-unit alignment mainly improves parameter-path peer selection. |
| `toy1_parameter_independent_cluster_dynamics.png` | Created | A partially fragmented parameter run shows agent prediction-cluster dynamics. |
| `toy1_parameter_cluster_comparison.png` | Created | Connected, partial, and fragmented parameter regimes have distinct peer/output dynamics. |

## Current Claims We Can Defend

The current Toy 1 evidence supports these limited claims:

- The runner can generate reproducible multi-agent micro-state and aggregate
  logs.
- Output-level social mixing gives a small but stable task improvement in this
  supervised toy setting.
- Output-level social mixing gives a clearer consensus improvement than task
  improvement.
- Parameter-level social mixing is highly sensitive to initialization and peer
  thresholding.
- Independent-init parameter averaging shows a reproducible threshold-driven
  fragmentation transition in the current Toy 1 setup.
- Hidden-unit alignment can substantially change parameter-based peer
  selection for independent-init agents.
- Peer fragmentation is visible in micro-state logs as higher per-agent output
  divergence from the population mean.
- Same-init parameter averaging is strong in this homogeneous toy setup, but it
  is not yet a robust general design recommendation.

## Claims We Should Not Make Yet

Do not claim yet that:

- Social mixing always improves task performance.
- Parameter averaging is generally better than output or latent mixing.
- Hidden-unit alignment is sufficient for deeper or heterogeneous architectures.
- The exact threshold transition point is universal across models or graph
  regimes.
- The result transfers to game dynamics or non-supervised ABM.
- The current Neural ABM Node is a Transformer-grade primitive.

## Next Experiments

Recommended next steps:

1. Freeze Toy 1 as the current baseline result set.
2. Begin Toy 2 Neural Spatial Prisoner's Dilemma using the updated social
   pipeline design.
