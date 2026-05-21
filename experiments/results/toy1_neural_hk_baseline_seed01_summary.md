# Toy 1 Baseline Result: Seed 1

Run:

`experiments/runs/20260429_143336_toy1_neural_hk_baseline_seed01`

Config:

`experiments/configs/toy1_neural_hk_baseline.yaml`

## Setup

- Toy model: Neural HK Classification.
- Mixer: `output_average`.
- Peer rule: `output_similarity`.
- Init mode: `same_init`.
- Agents: 50.
- Epochs: 50.
- Graph: Watts-Strogatz, `k = 6`, `rewire_probability = 0.1`.
- Device: CPU.

## Final Metrics

| Metric | Value |
| --- | ---: |
| Final mean global accuracy | 0.897808 |
| Final mean consensus | 0.967374 |
| Final fragmentation components | 1 |

## Log Checks

- `micro_state.csv`: 2,501 lines, matching 50 epochs x 50 agents plus header.
- `aggregate_metrics.csv`: 51 lines, matching 50 epochs plus header.
- `summary.json`: written.

## Interpretation

This is a pipeline validation run, not yet an ablation result. It confirms that
the baseline config can produce per-agent micro-state logs and aggregate metrics
through the common Toy 1 runner.

