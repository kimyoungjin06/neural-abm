# Toy2 Torch-State Generator-Based Initialization

Input artifacts:

- `toy2_torch_state_fast_path_aggregate_cluster_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_aggregate_cluster_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_generator_init_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_generator_init_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_generator_init_opt_1024_2048_e5_r5_comparison.csv`

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

## Change

Toy2 `tensor_batched` independent initialization now uses one CPU
`torch.Generator` per agent and passes it directly into the PyTorch initializer
calls. This avoids repeatedly resetting the global torch RNG during batched
parameter construction.

The generated parameters and final global RNG state are kept identical to the
existing agent-based initialization path. The final RNG state is restored from
the last per-agent generator.

## Results

`initial_state` stage:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.295522 | 0.024631 | 12.00x |
| CPU | 2048 | 0.426845 | 0.056894 | 7.50x |
| CUDA | 1024 | 0.218441 | 0.027162 | 8.04x |
| CUDA | 2048 | 0.420669 | 0.060050 | 7.01x |

Full-run wall-clock from aggregate-cluster-opt to generator-init-opt:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.390853 | 0.148592 | 2.63x |
| CPU | 2048 | 0.576808 | 0.380602 | 1.52x |
| CUDA | 1024 | 0.523062 | 0.424140 | 1.23x |
| CUDA | 2048 | 0.568580 | 0.436443 | 1.30x |

Steady-state stages vary in this short five-epoch smoke benchmark, but setup
time and full-run medians improve for every tested device/agent count.

## Current Bottlenecks

After setup optimization, the remaining steady-state target is local tensor
training: trainable parameter cloning, loss forward, autograd, and Adam update.
