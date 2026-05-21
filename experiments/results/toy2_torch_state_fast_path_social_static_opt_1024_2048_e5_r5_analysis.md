# Toy2 Torch-State Static Social Peer Fast Path

Input artifacts:

- `toy2_torch_state_fast_path_initial_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_initial_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_social_static_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_social_static_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_social_static_opt_1024_2048_e5_r5_comparison.csv`

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

Toy2 now reuses static spatial peer ids for
`neural_policy + output_average + peer_rule=none`. In this regime social peers
are always the spatial neighbors, so the runner can avoid converting policy
probabilities to CPU just to rediscover the same peer ids.

The tensor social update also receives cached uniform or flattened peer indexes,
and all-agent social updates are passed to the runtime as the all-active case
rather than a full `list(range(agent_count))`.

## Stage Results

| Device | Agents | Stage | Before | After | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| CPU | 1024 | select_peers | 0.000692 | 0.000002 | 279.80x |
| CPU | 1024 | social_mix | 0.002146 | 0.000232 | 9.25x |
| CPU | 1024 | hooked_step_total | 0.022458 | 0.005899 | 3.81x |
| CPU | 2048 | select_peers | 0.001190 | 0.000003 | 460.50x |
| CPU | 2048 | social_mix | 0.003549 | 0.000311 | 11.42x |
| CPU | 2048 | hooked_step_total | 0.024202 | 0.008191 | 2.95x |
| CUDA | 1024 | select_peers | 0.000971 | 0.000009 | 104.60x |
| CUDA | 1024 | social_mix | 0.003018 | 0.000415 | 7.27x |
| CUDA | 1024 | hooked_step_total | 0.017069 | 0.006976 | 2.45x |
| CUDA | 2048 | select_peers | 0.002434 | 0.000010 | 252.01x |
| CUDA | 2048 | social_mix | 0.008030 | 0.000433 | 18.54x |
| CUDA | 2048 | hooked_step_total | 0.037386 | 0.008171 | 4.58x |

Full-run wall-clock from initial-opt to static-social-opt:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 1.151117 | 0.579944 | 1.98x |
| CPU | 2048 | 1.406365 | 0.946850 | 1.49x |
| CUDA | 1024 | 1.835700 | 0.803192 | 2.29x |
| CUDA | 2048 | 2.334200 | 1.283566 | 1.82x |

## Current Bottlenecks

For short five-epoch runs, `initial_state` remains the largest single stage.
Steady-state work is now closer to the local tensor training path and aggregate
row computation. The next likely optimization is direct batched MLP parameter
initialization, which would remove the remaining per-agent module construction
from setup.
