# Toy steady-state timing analysis

Input: `experiments/results/toy4_tensor_batched_observation_index_stage_diag_1024_2048_e10_r3_stage_timings.csv`

`steady_state_total` is `sample_revision_mask + hooked_step_total`; setup, initial aggregate, per-epoch aggregate, and write stages are excluded. Child stages are nested diagnostics and should not be summed.

## Steady-state backend speedup

| toy | device | agents | batched ms/epoch | tensor ms/epoch | speedup |
|---|---:|---:|---:|---:|---:|

## Top tensor_batched focus stages

| toy | device | agents | stage | tensor ms/epoch | hook share | stage speedup |
|---|---:|---:|---|---:|---:|---:|
| toy4 | cpu | 1024 | local_loss_update | 5.508 | 27.9% | nanx |
| toy4 | cpu | 1024 | local_optimizer_update | 4.656 | 23.6% | nanx |
| toy4 | cpu | 1024 | social_step | 4.215 | 21.3% | nanx |
| toy4 | cpu | 1024 | social_distillation | 4.203 | 21.3% | nanx |
| toy4 | cpu | 1024 | local_adam_update | 3.198 | 16.2% | nanx |
| toy4 | cpu | 1024 | social_optimizer_update | 2.885 | 14.6% | nanx |
| toy4 | cpu | 2048 | local_loss_update | 5.969 | 21.3% | nanx |
| toy4 | cpu | 2048 | social_step | 5.454 | 19.5% | nanx |
| toy4 | cpu | 2048 | social_distillation | 5.442 | 19.4% | nanx |
| toy4 | cpu | 2048 | local_optimizer_update | 4.723 | 16.8% | nanx |
| toy4 | cpu | 2048 | social_optimizer_update | 3.694 | 13.2% | nanx |
| toy4 | cpu | 2048 | local_adam_update | 3.166 | 11.3% | nanx |
| toy4 | cuda | 1024 | local_loss_update | 13.177 | 32.4% | nanx |
| toy4 | cuda | 1024 | local_optimizer_update | 8.892 | 22.2% | nanx |
| toy4 | cuda | 1024 | local_adam_update | 6.926 | 15.4% | nanx |
| toy4 | cuda | 1024 | social_step | 5.039 | 18.4% | nanx |
| toy4 | cuda | 1024 | social_distillation | 5.012 | 18.2% | nanx |
| toy4 | cuda | 1024 | social_mix | 2.641 | 6.9% | nanx |
| toy4 | cuda | 2048 | social_step | 3.679 | 15.9% | nanx |
| toy4 | cuda | 2048 | social_distillation | 3.653 | 15.8% | nanx |
| toy4 | cuda | 2048 | local_loss_update | 3.150 | 13.6% | nanx |
| toy4 | cuda | 2048 | local_optimizer_update | 2.271 | 9.8% | nanx |
| toy4 | cuda | 2048 | social_optimizer_update | 1.861 | 8.1% | nanx |
| toy4 | cuda | 2048 | local_adam_update | 1.117 | 4.8% | nanx |

## Tensor stages slower than batched

| toy | device | agents | stage | delta ms/epoch | stage speedup |
|---|---:|---:|---|---:|---:|
