# Toy2 Torch-State Binary Loss Update

Input artifacts:

- `toy2_torch_state_fast_path_active_ids_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_active_ids_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_binary_loss_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_binary_loss_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_binary_loss_opt_1024_2048_e5_r5_comparison.csv`

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

The batched binary policy-gradient loss now uses the binary logit difference:

```text
logit_delta = logit(action=1) - logit(action=0)
log p(action=1) = logsigmoid(logit_delta)
log p(action=0) = logsigmoid(-logit_delta)
```

This preserves the two-action policy-gradient objective while avoiding the
general 2-class `log_softmax` graph in the local update. Aggregate policy means
for pre-revision, post-local, and post-social tensors are also stacked into a
single tensor-to-CPU transfer.

## Results

Full-run wall-clock from active-ids-opt to binary-loss-opt:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.113995 | 0.104498 | 1.09x |
| CPU | 2048 | 0.177323 | 0.153912 | 1.15x |
| CUDA | 1024 | 0.282199 | 0.260021 | 1.09x |
| CUDA | 2048 | 0.190723 | 0.180843 | 1.05x |

`local_loss_forward` stage:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.001816 | 0.001769 | 1.03x |
| CPU | 2048 | 0.002297 | 0.001828 | 1.26x |
| CUDA | 1024 | 0.011969 | 0.009731 | 1.23x |
| CUDA | 2048 | 0.002748 | 0.002003 | 1.37x |

`local_training` stage:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.016966 | 0.016674 | 1.02x |
| CPU | 2048 | 0.019563 | 0.014485 | 1.35x |
| CUDA | 1024 | 0.053363 | 0.049385 | 1.08x |
| CUDA | 2048 | 0.013948 | 0.010796 | 1.29x |

The final public metrics stayed identical to the active-ids run for each tested
device and agent count.

## Current Bottlenecks

The remaining steady-state cost is mostly Adam update, autograd, policy
readout, and aggregate/domain metrics. Attempts to specialize Adam scalar step
correction and all-revised action selection did not produce stable end-to-end
speedups in this smoke benchmark, so those changes were not retained.
