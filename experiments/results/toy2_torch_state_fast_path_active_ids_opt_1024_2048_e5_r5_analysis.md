# Toy2 Torch-State Active-ID Local Update

Input artifacts:

- `toy2_torch_state_fast_path_trainable_view_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_trainable_view_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_active_ids_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_active_ids_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_active_ids_opt_1024_2048_e5_r5_comparison.csv`

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

Toy2 sampled local training now carries canonical active-agent ids directly into
the tensor policy-gradient update. When every agent is revised, the local update
passes `None` to the tensor runtime, avoiding the full agent-id list and the
temporary bool update mask.

The revision-mask API remains available for Toy4/Toy5 and other existing
callers.

## Results

Full-run wall-clock from trainable-view-opt to active-ids-opt:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.130453 | 0.113995 | 1.14x |
| CPU | 2048 | 0.234282 | 0.177323 | 1.32x |
| CUDA | 1024 | 0.378980 | 0.282199 | 1.34x |
| CUDA | 2048 | 0.244761 | 0.190723 | 1.28x |

`hooked_step_total` stage:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.047210 | 0.040943 | 1.15x |
| CPU | 2048 | 0.064621 | 0.050401 | 1.28x |
| CUDA | 1024 | 0.189546 | 0.126164 | 1.50x |
| CUDA | 2048 | 0.070921 | 0.049644 | 1.43x |

`local_step` stage:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.026562 | 0.023248 | 1.14x |
| CPU | 2048 | 0.037805 | 0.028995 | 1.30x |
| CUDA | 1024 | 0.150088 | 0.098637 | 1.52x |
| CUDA | 2048 | 0.038148 | 0.026218 | 1.46x |

`local_training` stage:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.019162 | 0.016966 | 1.13x |
| CPU | 2048 | 0.025077 | 0.019563 | 1.28x |
| CUDA | 1024 | 0.088183 | 0.053363 | 1.65x |
| CUDA | 2048 | 0.021731 | 0.013948 | 1.56x |

The final public metrics stayed identical to the trainable-view run for each
tested device and agent count.

## Current Bottlenecks

After the active-id cleanup, remaining steady-state cost is concentrated in
loss forward, autograd, and Adam update. Further optimization should avoid
changing RNG or policy-update semantics.
