# Toy4/Toy5 Torch-State Generalization Benchmark

Date: 2026-05-11

## Inputs

- Static peers: `experiments/results/toy45_torch_state_generalization_none_1024_2048_e5_r5.csv`
- Static peer stage timings: `experiments/results/toy45_torch_state_generalization_none_1024_2048_e5_r5_stage_timings.csv`
- Output similarity: `experiments/results/toy45_torch_state_generalization_output_similarity_1024_2048_e3_r2.csv`
- Output similarity stage timings: `experiments/results/toy45_torch_state_generalization_output_similarity_1024_2048_e3_r2_stage_timings.csv`

## Parity

Tensor-state `tensor_batched` matched the existing `batched` backend for all Toy4/Toy5 CPU/CUDA smoke cases.

- Static peers max diff vs `batched`:
  - `final_action_rate`: `0.0`
  - `final_mean_policy_action_probability`: `2.9802322387695312e-08`
  - `final_mean_reputation`: `0.0`
- Output similarity max diff vs `batched`:
  - `final_action_rate`: `0.0`
  - `final_mean_policy_action_probability`: `0.0`
  - `final_mean_reputation`: `0.0`

## Timing Summary

The values below are `batched_seconds / tensor_batched_seconds`; values above `1.0` favor tensor-state `tensor_batched`.

### Static Peers, Epochs 5, Repeats 5

- Toy4 CPU 1024: `1.659x`
- Toy4 CUDA 1024: `1.421x`
- Toy4 CPU 2048: `1.366x`
- Toy4 CUDA 2048: `1.270x`
- Toy5 CPU 1024: `1.178x`
- Toy5 CUDA 1024: `1.004x`
- Toy5 CPU 2048: `1.411x`
- Toy5 CUDA 2048: `1.055x`

Range: `1.004x` to `1.659x`; mean `1.296x`.

### Output Similarity, Epochs 3, Repeats 2

- Toy4 CPU 1024: `2.169x`
- Toy4 CUDA 1024: `1.847x`
- Toy4 CPU 2048: `1.258x`
- Toy4 CUDA 2048: `1.271x`
- Toy5 CPU 1024: `1.290x`
- Toy5 CUDA 1024: `1.098x`
- Toy5 CPU 2048: `1.246x`
- Toy5 CUDA 2048: `1.164x`

Range: `1.098x` to `2.169x`; mean `1.418x`.
