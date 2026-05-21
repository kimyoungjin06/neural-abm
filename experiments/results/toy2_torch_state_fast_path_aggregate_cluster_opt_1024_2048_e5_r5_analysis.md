# Toy2 Torch-State Aggregate Metric Cache and Cluster Optimization

Input artifacts:

- `toy2_torch_state_fast_path_prealloc_init_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_prealloc_init_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_aggregate_cache_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_aggregate_cache_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_aggregate_cluster_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_aggregate_cluster_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_aggregate_cluster_opt_1024_2048_e5_r5_comparison.csv`

Command shape:

```text
uv run python scripts/benchmark_toy_gpu_core.py \
  --toys toy2 \
  --devices cpu cuda \
  --agent-counts 1024 2048 \
  --epochs 5 \
  --repeats 5 \
  --training-backends tensor_batched \
  --mixers output_average
```

## Changes

Toy2 now caches static peer aggregate metrics for
`neural_policy + output_average + peer_rule=none`, avoiding repeated NetworkX
peer graph construction for fragmentation, mean peer count, and edge entropy.

Toy2 action cluster metrics now use a cached graph edge array plus union-find
instead of constructing a NetworkX cooperator subgraph each aggregate row.

The direct initializer was also changed from list-plus-`torch.stack` to
preallocated batched tensors filled by agent slice. This preserves the existing
seed sequence and gives a modest CPU setup improvement.

## Results

Aggregate metric stages from prealloc-init to aggregate-cluster-opt:

| Device | Agents | Stage | Before | After | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| CPU | 1024 | initial_aggregate_row | 0.005236 | 0.002347 | 2.23x |
| CPU | 1024 | aggregate_row | 0.004351 | 0.001188 | 3.66x |
| CPU | 2048 | initial_aggregate_row | 0.010413 | 0.004231 | 2.46x |
| CPU | 2048 | aggregate_row | 0.008608 | 0.002073 | 4.15x |
| CUDA | 1024 | initial_aggregate_row | 0.006294 | 0.002885 | 2.18x |
| CUDA | 1024 | aggregate_row | 0.004568 | 0.001508 | 3.03x |
| CUDA | 2048 | initial_aggregate_row | 0.012788 | 0.004251 | 3.01x |
| CUDA | 2048 | aggregate_row | 0.012199 | 0.002357 | 5.17x |

Full-run wall-clock from prealloc-init to aggregate-cluster-opt:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.385648 | 0.390853 | 0.99x |
| CPU | 2048 | 0.599714 | 0.576808 | 1.04x |
| CUDA | 1024 | 0.506975 | 0.523062 | 0.97x |
| CUDA | 2048 | 0.665699 | 0.568580 | 1.17x |

The short five-epoch wall-clock remains noisy, but the aggregate stages are
consistently lower. CUDA 2048 benefits most because aggregate rows had become
a visible fraction of total runtime after the social fast path.

## Current Bottlenecks

The largest remaining short-run cost is still setup, mainly RNG-driven initial
parameter generation and tensor transfer. In steady-state rows, local tensor
training is now the main target.
