# Toy 1 Parameter Independent Cluster Snapshot

Run:

`experiments/runs/20260429_164952_toy1_parameter_independent_cluster_snapshot_seed01`

Config:

`experiments/configs/toy1_parameter_independent_cluster_snapshot.yaml`

## Setup

- Mixer: `parameter_average`.
- Peer rule: `state_similarity`.
- Init mode: `independent_init`.
- Alpha: `0.25`.
- Threshold: `0.2`.
- Probe prediction snapshots: enabled every epoch.
- Epochs: 50.

## Final Metrics

| Metric | Value |
| --- | ---: |
| Final mean global accuracy | 0.894148 |
| Final mean consensus | 0.959692 |
| Final fragmentation components | 29 |
| Probe prediction snapshots | 50 |

## Figures

- `paper/figures/toy1_parameter_independent_cluster_dynamics.png`

## Interpretation

This run targets a partially fragmented point from the independent-init
parameter sweep. It is intended to show agent differentiation over time rather
than to establish a multi-seed claim.

