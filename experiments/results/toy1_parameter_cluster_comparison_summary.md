# Toy 1 Parameter Cluster Comparison

Source CSV:

`experiments/results/toy1_parameter_cluster_compare_seed01_summary.csv`

Figure:

`paper/figures/toy1_parameter_cluster_comparison.png`

## Setup

- Toy model: Neural HK Classification.
- Mixer: `parameter_average`.
- Peer rule: `state_similarity`.
- Init mode: `independent_init`.
- Alpha: `0.25`.
- Seed: `1`.
- Snapshot logging: enabled for every epoch.

Compared thresholds:

| Threshold | Final Accuracy | Final Consensus | Final Fragmentation |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.897956 | 0.979239 | 1 |
| 0.2 | 0.894148 | 0.959692 | 29 |
| 0.6 | 0.891972 | 0.949434 | 50 |

## Interpretation

- The connected regime (`threshold=0.0`) keeps the peer graph mostly connected
  after the first few epochs and produces the lowest output divergence to the
  population mean.
- The partial regime (`threshold=0.2`) shows a persistent multi-component peer
  graph, with output divergence increasing over time for several agents.
- The fragmented regime (`threshold=0.6`) behaves like near-independent local
  learning: peer components stay near 50 and output divergence continues to
  grow.
- The figure is a single-seed diagnostic visualization. The stability of the
  threshold transition is supported separately by the 5-seed phase sweep.

## Source Runs

| Threshold | Run Directory |
| ---: | --- |
| 0.0 | `experiments/runs/20260429_193044_toy1_parameter_cluster_compare_seed01_parameter_average_state_similarity_independent_init_a0p25_t0p0_seed01` |
| 0.2 | `experiments/runs/20260429_193053_toy1_parameter_cluster_compare_seed01_parameter_average_state_similarity_independent_init_a0p25_t0p2_seed01` |
| 0.6 | `experiments/runs/20260429_193103_toy1_parameter_cluster_compare_seed01_parameter_average_state_similarity_independent_init_a0p25_t0p6_seed01` |
