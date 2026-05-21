# Toy2 Torch-State Finalize and Initial-State Optimizations

Input artifacts:

- `toy2_torch_state_fast_path_consensus_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_consensus_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_finalize_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_finalize_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_initial_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_initial_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_initial_opt_1024_2048_e5_r5_comparison.csv`

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

The finalize optimization defers per-agent payoff EMA synchronization in
`tensor_batched` runs. Torch-state observations and local baselines read the
device-resident EMA tensors directly, so agent object fields only need syncing
when there are agent objects to flush at summary time.

The initial-state optimization builds `TensorBatchedMLPRuntime` directly from
initial models and zero Adam tensors. It preserves the existing model seed
order, including `same_init` RNG side effects, while skipping per-agent Adam
optimizer construction for `tensor_batched` state.

## Stage Results

`finalize_hook_step` after the finalize optimization:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.003093 | 0.000021 | 146.72x |
| CPU | 2048 | 0.006252 | 0.000025 | 253.44x |
| CUDA | 1024 | 0.019986 | 0.000027 | 745.18x |
| CUDA | 2048 | 0.041815 | 0.000041 | 1020.21x |

`initial_state` after the direct runtime initialization:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 1.210200 | 0.918639 | 1.32x |
| CPU | 2048 | 2.392417 | 1.026184 | 2.33x |
| CUDA | 1024 | 1.091976 | 0.765622 | 1.43x |
| CUDA | 2048 | 2.761704 | 1.722814 | 1.60x |

Full-run wall-clock from finalize-opt to initial-opt:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 1.959499 | 1.151117 | 1.70x |
| CPU | 2048 | 3.393727 | 1.406365 | 2.41x |
| CUDA | 1024 | 2.078481 | 1.835700 | 1.13x |
| CUDA | 2048 | 4.169952 | 2.334200 | 1.79x |

CUDA 1024 remains noisy in these short five-epoch runs; the stage medians still
show the direct runtime construction reducing setup time.

## Current Bottlenecks

After these changes, `initial_state` is still the largest single stage because
the safe implementation still instantiates one initial model per agent before
stacking parameters. The largest steady-state stages are now inside
`hooked_step_total`, especially local/social tensor training and policy readout.

Good next targets:

- Generate initial batched MLP parameters directly, avoiding per-agent module
  objects while preserving current seed semantics.
- Reduce repeated trainable parameter cloning inside local and social updates.
- Keep mobility as a separate target; current support is correct but still uses
  a NumPy mirror copy-back path.
