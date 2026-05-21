# Toy2 Torch-State Fast Path Benchmark

Input artifacts:

- `toy2_torch_state_fast_path_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_consensus_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_consensus_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_consensus_opt_1024_2048_e5_r5_comparison.csv`

Command shape:

```text
uv run python scripts/benchmark_toy_gpu_core.py \
  --toys toy2 \
  --devices cpu cuda \
  --agent-counts 1024 2048 \
  --epochs 5 \
  --repeats 5 \
  --training-backends loop batched tensor_batched \
  --mixers output_average \
  --require-backend-parity
```

CUDA was available for this run. All non-loop backends passed loop parity for:

- `final_action_rate`
- `final_mean_policy_action_probability`
- `final_mean_reputation`

## Main Results

After the torch-state path plus consensus optimization, Toy2 `tensor_batched`
has the following loop speedups:

| Device | Agents | tensor_batched seconds | loop speedup |
| --- | ---: | ---: | ---: |
| CPU | 1024 | 0.7451 | 7.38x |
| CUDA | 1024 | 1.0895 | 8.23x |
| CPU | 2048 | 1.5633 | 6.79x |
| CUDA | 2048 | 2.1837 | 8.10x |

The consensus optimization improved the full benchmark wall time for
`tensor_batched` by:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 1.7161 | 0.7451 | 2.30x |
| CUDA | 1024 | 2.2244 | 1.0895 | 2.04x |
| CPU | 2048 | 5.3482 | 1.5633 | 3.42x |
| CUDA | 2048 | 5.7235 | 2.1837 | 2.62x |

## Stage Findings

The first formal torch-state benchmark showed `aggregate_row` and
`initial_aggregate_row` dominating `tensor_batched`, especially at 2048 agents.
The cause was Toy2 `policy_consensus`, which computed mean pairwise policy
distance with an O(n^2) Python loop.

The replacement uses the sorted absolute-difference identity, reducing that
metric to O(n log n) and keeping the same scalar definition.

Stage timing changes for `tensor_batched`:

| Device | Agents | Stage | Before | After | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| CPU | 1024 | initial_aggregate_row | 0.155213 | 0.005749 | 27.00x |
| CPU | 1024 | aggregate_row | 0.141546 | 0.005353 | 26.44x |
| CPU | 2048 | initial_aggregate_row | 0.604753 | 0.010988 | 55.04x |
| CPU | 2048 | aggregate_row | 0.599437 | 0.008723 | 68.72x |
| CUDA | 1024 | initial_aggregate_row | 0.149680 | 0.005849 | 25.59x |
| CUDA | 1024 | aggregate_row | 0.146185 | 0.005149 | 28.39x |
| CUDA | 2048 | initial_aggregate_row | 0.560486 | 0.010345 | 54.18x |
| CUDA | 2048 | aggregate_row | 0.575506 | 0.016158 | 35.62x |

Post-optimization, the top `tensor_batched` stage is `initial_state`, which is
mostly model and optimizer state construction rather than steady-state epoch
work. The largest steady-state stages are now `hooked_step_total` and
`finalize_hook_step`, with CUDA 2048 showing `finalize_hook_step` at roughly
0.043 seconds per repeat.

## Next Bottlenecks

The next useful optimizations are not broad runner changes:

- Treat `initial_state` separately from steady-state throughput reports because
  it is one-time setup.
- Inspect CUDA `finalize_hook_step`, which now stands out after aggregate
  metrics were fixed.
- If mobility-heavy Toy2 scenarios become a target, replace the current
  NumPy mirror mobility copy-back with a torch-native swap path.
- If output-similarity-heavy Toy2 scenarios become a target, avoid materializing
  cooperation probabilities on CPU during peer selection.
