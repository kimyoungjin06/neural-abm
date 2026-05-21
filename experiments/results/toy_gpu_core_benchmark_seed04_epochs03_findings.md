# Toy2/4/5 GPU Core Benchmark Findings

Date: 2026-05-07

Environment:

- Torch: `2.11.0+cu130`
- GPU: `NVIDIA GeForce RTX 4080`
- Benchmark run CSV: `experiments/results/toy_gpu_core_benchmark_seed04_epochs03.csv`
- Policy-core CSVs:
  - `experiments/results/accelerator_policy_core_cpu_seed04_counts.csv`
  - `experiments/results/accelerator_policy_core_cuda_seed04_counts.csv`

Commands:

```bash
uv run python scripts/benchmark_toy_gpu_core.py \
  --devices cpu cuda \
  --toys toy2 toy4 toy5 \
  --agent-counts 100 256 512 \
  --mixers none output_average \
  --epochs 3 \
  --seed 4 \
  --output experiments/results/toy_gpu_core_benchmark_seed04_epochs03.csv \
  --run-output-dir experiments/runs/toy_gpu_core_benchmark_seed04_epochs03

uv run python scripts/benchmark_accelerator_policy_core.py \
  --device cpu \
  --agent-counts 100 256 512 \
  --input-dim 6 \
  --hidden-dim 16 \
  --warmup 10 \
  --repeats 200 \
  --output experiments/results/accelerator_policy_core_cpu_seed04_counts.csv

uv run python scripts/benchmark_accelerator_policy_core.py \
  --device cuda \
  --agent-counts 100 256 512 \
  --input-dim 6 \
  --hidden-dim 16 \
  --warmup 10 \
  --repeats 200 \
  --output experiments/results/accelerator_policy_core_cuda_seed04_counts.csv
```

## Summary

The cached batched neural readout works, but v1 is not yet an end-to-end GPU
speedup. Across 18 Toy2/4/5 CPU-vs-CUDA pairs, CUDA averaged `0.57x` CPU
throughput by `agent_steps_per_second`; only one small Toy2 `neural_none`
case was slightly faster on CUDA (`1.04x`).

This points to the expected v1 bottleneck: policy inference is batched, but
local learning, Adam optimizer updates, distillation commits, and cache refresh
still execute in per-agent loops. The GPU path pays transfer/kernel overhead
without moving enough of the epoch workload into fused kernels.

## End-To-End CUDA/CPU Throughput

`cuda_vs_cpu_speedup = cuda agent_steps_per_second / cpu agent_steps_per_second`.

| Toy | Scenario | Agents | CUDA/CPU |
| --- | --- | ---: | ---: |
| toy2 | neural_none | 100 | 1.04 |
| toy2 | neural_none | 256 | 0.54 |
| toy2 | neural_none | 512 | 0.50 |
| toy2 | neural_output_average | 100 | 0.49 |
| toy2 | neural_output_average | 256 | 0.58 |
| toy2 | neural_output_average | 512 | 0.48 |
| toy4 | neural_none | 100 | 0.62 |
| toy4 | neural_none | 256 | 0.62 |
| toy4 | neural_none | 512 | 0.50 |
| toy4 | neural_output_average | 100 | 0.49 |
| toy4 | neural_output_average | 256 | 0.45 |
| toy4 | neural_output_average | 512 | 0.60 |
| toy5 | neural_none | 100 | 0.44 |
| toy5 | neural_none | 256 | 0.71 |
| toy5 | neural_none | 512 | 0.67 |
| toy5 | neural_output_average | 100 | 0.52 |
| toy5 | neural_output_average | 256 | 0.59 |
| toy5 | neural_output_average | 512 | 0.49 |

## Policy-Core Isolation

The policy-core microbenchmark confirms that cached batched readout removes the
old per-agent forward bottleneck. Relative to per-agent loop inference:

- CPU cached policy cache speedups: `23x`, `28x`, `93x` for 100/256/512 agents.
- CUDA cached policy cache speedups: `54x`, `170x`, `336x` for 100/256/512 agents.
- Cache refresh remains material: refresh+readout is only about `1.0x-1.1x`
  CUDA-vs-CPU at these counts.

The end-to-end runner therefore needs a broader GPU move before CUDA should be
expected to win.

## Next Work

1. Move local neural updates from per-agent optimizer steps toward a functional
   batched update path for the common same-shaped MLP case.
2. Make social distillation compute targets in batch and reduce per-agent commit
   overhead.
3. Reduce refresh frequency or make refresh incremental where only revised
   agents changed.
4. Add CI-level CPU/CUDA invariant tests with tolerances, while keeping CPU
   golden metrics strict.
