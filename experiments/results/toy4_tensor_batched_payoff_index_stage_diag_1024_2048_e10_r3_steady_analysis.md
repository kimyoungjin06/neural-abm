# Toy steady-state timing analysis

Input: `experiments/results/toy4_tensor_batched_payoff_index_stage_diag_1024_2048_e10_r3_stage_timings.csv`

`steady_state_total` is `sample_revision_mask + hooked_step_total`; setup, initial aggregate, per-epoch aggregate, and write stages are excluded. Child stages are nested diagnostics and should not be summed.

## Steady-state backend speedup

| toy | device | agents | batched ms/epoch | tensor ms/epoch | speedup |
|---|---:|---:|---:|---:|---:|

## Top tensor_batched focus stages

| toy | device | agents | stage | tensor ms/epoch | hook share | stage speedup |
|---|---:|---:|---|---:|---:|---:|
| toy4 | cpu | 1024 | local_loss_update | 5.068 | 38.9% | nanx |
| toy4 | cpu | 1024 | local_optimizer_update | 4.293 | 32.9% | nanx |
| toy4 | cpu | 1024 | social_step | 3.754 | 29.2% | nanx |
| toy4 | cpu | 1024 | social_distillation | 3.742 | 29.1% | nanx |
| toy4 | cpu | 1024 | local_adam_update | 2.967 | 22.7% | nanx |
| toy4 | cpu | 1024 | social_optimizer_update | 2.444 | 18.9% | nanx |
| toy4 | cpu | 2048 | local_loss_update | 5.761 | 31.9% | nanx |
| toy4 | cpu | 2048 | social_step | 5.410 | 30.0% | nanx |
| toy4 | cpu | 2048 | social_distillation | 5.397 | 29.9% | nanx |
| toy4 | cpu | 2048 | local_optimizer_update | 4.629 | 25.7% | nanx |
| toy4 | cpu | 2048 | social_optimizer_update | 3.591 | 19.9% | nanx |
| toy4 | cpu | 2048 | local_adam_update | 3.167 | 17.6% | nanx |
| toy4 | cuda | 1024 | local_loss_update | 14.521 | 39.0% | nanx |
| toy4 | cuda | 1024 | local_optimizer_update | 9.534 | 27.0% | nanx |
| toy4 | cuda | 1024 | local_adam_update | 7.675 | 18.1% | nanx |
| toy4 | cuda | 1024 | social_step | 4.558 | 23.5% | nanx |
| toy4 | cuda | 1024 | social_distillation | 4.533 | 23.4% | nanx |
| toy4 | cuda | 1024 | social_mix | 2.279 | 7.6% | nanx |
| toy4 | cuda | 2048 | social_step | 3.089 | 23.5% | nanx |
| toy4 | cuda | 2048 | social_distillation | 3.065 | 23.3% | nanx |
| toy4 | cuda | 2048 | local_loss_update | 2.914 | 22.1% | nanx |
| toy4 | cuda | 2048 | local_optimizer_update | 2.130 | 16.2% | nanx |
| toy4 | cuda | 2048 | social_optimizer_update | 1.676 | 12.7% | nanx |
| toy4 | cuda | 2048 | policy_readout | 1.062 | 8.0% | nanx |

## Tensor stages slower than batched

| toy | device | agents | stage | delta ms/epoch | stage speedup |
|---|---:|---:|---|---:|---:|
