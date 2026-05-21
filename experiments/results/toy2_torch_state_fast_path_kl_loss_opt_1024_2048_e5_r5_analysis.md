# Toy2 Torch-State KL Loss Update

Input artifacts:

- `toy2_torch_state_fast_path_binary_loss_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_binary_loss_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_kl_loss_opt_1024_2048_e5_r5.csv`
- `toy2_torch_state_fast_path_kl_loss_opt_1024_2048_e5_r5_stage_timings.csv`
- `toy2_torch_state_fast_path_kl_loss_opt_1024_2048_e5_r5_comparison.csv`

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

The tensor distribution loss keeps the public KL loss value but removes the
target-entropy term from the autograd graph:

```text
KL(target || policy) = cross_entropy(target, policy) - entropy(target)
```

`entropy(target)` is independent of policy parameters, so the optimizer sees the
same gradient while the reported per-agent social loss remains unchanged.

## Results

Full-run wall-clock from binary-loss-opt to kl-loss-opt:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.104498 | 0.096439 | 1.08x |
| CPU | 2048 | 0.153912 | 0.151329 | 1.02x |
| CUDA | 1024 | 0.260021 | 0.262372 | 0.99x |
| CUDA | 2048 | 0.180843 | 0.155775 | 1.16x |

`social_distillation` stage:

| Device | Agents | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1024 | 0.015290 | 0.015272 | 1.00x |
| CPU | 2048 | 0.015374 | 0.016109 | 0.95x |
| CUDA | 1024 | 0.017458 | 0.015846 | 1.10x |
| CUDA | 2048 | 0.012060 | 0.010569 | 1.14x |

The final public metrics stayed identical to the binary-loss run for each
tested device and agent count.

## Current Bottlenecks

The next likely target is Adam update/autograd fusion or reducing policy
readout passes. The latter needs care because post-local and post-social
probabilities are part of the public aggregate and micro-state contract.
