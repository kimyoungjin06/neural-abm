# Toy2 Torch-State Trainable-View Update

Input artifacts:

- `toy2_torch_state_fast_path_generator_init_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_generator_init_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_trainable_view_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_trainable_view_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_trainable_view_opt_1024_2048_e5_r5_comparison.csv`

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

`TensorBatchedMLPRuntime.trainable_parameters()` now returns detached,
grad-enabled views over the runtime-owned parameter tensors instead of detached
clones. The Adam update still runs through the tensor runtime and the returned
runtime snapshot is detached after each update.

This keeps the optimized Toy2 tensor-state path from copying all batched MLP
weights before each local and social update.

## Results

Full-run wall-clock from generator-init-opt to trainable-view-opt:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.148592 | 0.130453 | 1.14x |
| CPU | 2048 | 0.380602 | 0.234282 | 1.62x |
| CUDA | 1024 | 0.424140 | 0.378980 | 1.12x |
| CUDA | 2048 | 0.436443 | 0.244761 | 1.78x |

`hooked_step_total` stage:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.060636 | 0.047210 | 1.28x |
| CPU | 2048 | 0.132997 | 0.064621 | 2.06x |
| CUDA | 1024 | 0.213771 | 0.189546 | 1.13x |
| CUDA | 2048 | 0.122570 | 0.070921 | 1.73x |

`local_trainable_parameters` stage:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.001416 | 0.000070 | 20.12x |
| CPU | 2048 | 0.004240 | 0.000074 | 57.26x |
| CUDA | 1024 | 0.000972 | 0.000147 | 6.61x |
| CUDA | 2048 | 0.000804 | 0.000352 | 2.29x |

`social_trainable_parameters` stage:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.001287 | 0.000045 | 28.72x |
| CPU | 2048 | 0.003338 | 0.000049 | 68.48x |
| CUDA | 1024 | 0.000396 | 0.000102 | 3.89x |
| CUDA | 2048 | 0.000748 | 0.000112 | 6.67x |

The final public metrics stayed identical to the previous run for each tested
device and agent count.

## Current Bottlenecks

The next steady-state targets are loss forward, autograd, and Adam update. The
trainable-parameter construction stage is now small enough that further gains
should focus on update math or active-agent mask handling.
