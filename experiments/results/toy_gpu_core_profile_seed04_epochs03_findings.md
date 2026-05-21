# Toy2/4/5 GPU Core Stage Profile Findings

Date: 2026-05-07

Environment:

- Torch: `2.11.0+cu130`
- GPU: `NVIDIA GeForce RTX 4080`
- Profile summary CSV: `experiments/results/toy_gpu_core_profile_seed04_epochs03.csv`
- Detailed stage CSV: `experiments/results/toy_gpu_core_profile_seed04_epochs03_stage_timings.csv`

Command:

```bash
uv run python scripts/benchmark_toy_gpu_core.py \
  --devices cpu cuda \
  --toys toy2 toy4 toy5 \
  --agent-counts 100 256 512 \
  --mixers none output_average \
  --epochs 3 \
  --seed 4 \
  --output experiments/results/toy_gpu_core_profile_seed04_epochs03.csv \
  --stage-output experiments/results/toy_gpu_core_profile_seed04_epochs03_stage_timings.csv \
  --run-output-dir experiments/runs/toy_gpu_core_profile_seed04_epochs03
```

The stage profiler synchronizes CUDA around timed sections, so use these values
for bottleneck attribution rather than direct wall-clock comparisons with the
non-profile benchmark.

## Summary

The bottleneck is not batched policy readout. In neural `none` scenarios,
`local_training` dominates the epoch step:

- Toy2: `90-94%` of CPU hooked-step time and `66-97%` of CUDA hooked-step time.
- Toy4: `90-91%` of CPU hooked-step time and `94-97%` of CUDA hooked-step time.
- Toy5: `87-93%` of CPU hooked-step time and `95-97%` of CUDA hooked-step time.

In `output_average` scenarios, the dominant cost splits between per-agent local
training and per-agent social distillation:

- CPU local training: about `46-50%` of hooked-step time.
- CPU social distillation: about `44-48%` of hooked-step time.
- CUDA local training: about `34-55%` of hooked-step time.
- CUDA social distillation: about `42-63%` of hooked-step time.

Policy readout is consistently small after the cache path:

- `policy_readout` is generally below `1%` of hooked-step time at 256/512 agents.
- `cache_refresh` is material but secondary, usually `1-5%` of hooked-step time.

## Implication

The next GPU work should not focus on more inference optimization. The fastest
path to end-to-end CUDA gains is:

1. Batch local training for same-shaped MLP agents.
2. Batch social distillation target generation and commit.
3. Reduce cache refresh cost after model updates are no longer purely per-agent.

Toy4 and Toy5 are the best first implementation targets because their neural
local losses are simpler than Toy2 counterfactual payoff training.
