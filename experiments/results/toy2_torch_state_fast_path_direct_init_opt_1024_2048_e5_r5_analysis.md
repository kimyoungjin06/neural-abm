# Toy2 Torch-State Direct Batched Initialization

Input artifacts:

- `toy2_torch_state_fast_path_social_static_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_social_static_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_direct_init_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_direct_init_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_direct_init_opt_1024_2048_e5_r5_comparison.csv`

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

Toy2 `tensor_batched` initialization now builds `BatchedMLPParameters`
directly instead of instantiating one `PolicyMLP` module per agent. The helper
reproduces the PyTorch `Linear.reset_parameters` sequence used by the existing
module path:

- initialize `fc1`
- initialize `fc2`
- apply the optional policy-head prior by overwriting `fc2`

It also preserves seed semantics for both `independent_init` and `same_init`,
including the extra RNG consumption from overwritten per-agent models in
`same_init`.

## Results

`initial_state` stage:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.506079 | 0.305804 | 1.65x |
| CPU | 2048 | 0.771969 | 0.496484 | 1.55x |
| CUDA | 1024 | 0.540145 | 0.223766 | 2.41x |
| CUDA | 2048 | 1.060869 | 0.458214 | 2.32x |

Full-run wall-clock from static-social-opt to direct-init-opt:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.579944 | 0.417228 | 1.39x |
| CPU | 2048 | 0.946850 | 0.741665 | 1.28x |
| CUDA | 1024 | 0.803192 | 0.502657 | 1.60x |
| CUDA | 2048 | 1.283566 | 0.663568 | 1.93x |

Some CPU steady-state stage medians move in the wrong direction in this short
five-epoch run, but the setup stage and full-run medians improve for every
tested device/agent count.

## Current Bottlenecks

`initial_state` is still visible, now mostly tensor allocation, stacking, and
host-to-device transfer. For steady-state runs, the largest remaining work is
local tensor training and aggregate row construction.
