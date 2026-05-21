# Toy steady-state timing analysis

Input: `experiments/results/toy45_tensor_batched_shared_step_stage_diag_1024_2048_e10_r3_stage_timings.csv`

`steady_state_total` is `sample_revision_mask + hooked_step_total`; setup, initial aggregate, per-epoch aggregate, and write stages are excluded. Child stages are nested diagnostics and should not be summed.

## Steady-state backend speedup

| toy | device | agents | batched ms/epoch | tensor ms/epoch | speedup |
|---|---:|---:|---:|---:|---:|

## Top tensor_batched focus stages

| toy | device | agents | stage | tensor ms/epoch | hook share | stage speedup |
|---|---:|---:|---|---:|---:|---:|
| toy4 | cpu | 1024 | local_loss_update | 5.202 | 39.1% | nanx |
| toy4 | cpu | 1024 | local_optimizer_update | 4.306 | 32.4% | nanx |
| toy4 | cpu | 1024 | social_step | 3.838 | 28.9% | nanx |
| toy4 | cpu | 1024 | social_distillation | 3.826 | 28.8% | nanx |
| toy4 | cpu | 1024 | local_adam_update | 2.871 | 21.6% | nanx |
| toy4 | cpu | 1024 | social_optimizer_update | 2.483 | 18.7% | nanx |
| toy4 | cpu | 2048 | local_loss_update | 5.880 | 32.4% | nanx |
| toy4 | cpu | 2048 | social_step | 5.332 | 29.4% | nanx |
| toy4 | cpu | 2048 | social_distillation | 5.319 | 29.3% | nanx |
| toy4 | cpu | 2048 | local_optimizer_update | 4.621 | 25.5% | nanx |
| toy4 | cpu | 2048 | social_optimizer_update | 3.513 | 19.4% | nanx |
| toy4 | cpu | 2048 | local_adam_update | 3.158 | 17.4% | nanx |
| toy4 | cuda | 1024 | local_loss_update | 14.972 | 41.4% | nanx |
| toy4 | cuda | 1024 | local_optimizer_update | 8.348 | 24.9% | nanx |
| toy4 | cuda | 1024 | local_adam_update | 5.936 | 15.0% | nanx |
| toy4 | cuda | 1024 | social_step | 4.648 | 22.4% | nanx |
| toy4 | cuda | 1024 | social_distillation | 4.621 | 22.2% | nanx |
| toy4 | cuda | 1024 | social_mix | 2.329 | 7.6% | nanx |
| toy4 | cuda | 2048 | social_step | 3.209 | 23.5% | nanx |
| toy4 | cuda | 2048 | social_distillation | 3.185 | 23.3% | nanx |
| toy4 | cuda | 2048 | local_loss_update | 3.039 | 22.1% | nanx |
| toy4 | cuda | 2048 | local_optimizer_update | 2.077 | 15.1% | nanx |
| toy4 | cuda | 2048 | social_optimizer_update | 1.662 | 12.2% | nanx |
| toy4 | cuda | 2048 | policy_readout | 1.049 | 7.6% | nanx |
| toy5 | cpu | 1024 | social_step | 3.406 | 21.0% | nanx |
| toy5 | cpu | 1024 | social_distillation | 3.388 | 20.9% | nanx |
| toy5 | cpu | 1024 | local_loss_update | 3.141 | 19.0% | nanx |
| toy5 | cpu | 1024 | local_optimizer_update | 2.623 | 15.9% | nanx |
| toy5 | cpu | 1024 | social_optimizer_update | 2.144 | 13.3% | nanx |
| toy5 | cpu | 1024 | local_adam_update | 1.598 | 9.6% | nanx |
| toy5 | cpu | 2048 | social_step | 5.500 | 19.1% | nanx |
| toy5 | cpu | 2048 | social_distillation | 5.482 | 19.0% | nanx |
| toy5 | cpu | 2048 | local_loss_update | 5.317 | 18.4% | nanx |
| toy5 | cpu | 2048 | local_optimizer_update | 4.363 | 15.1% | nanx |
| toy5 | cpu | 2048 | social_optimizer_update | 3.425 | 11.9% | nanx |
| toy5 | cpu | 2048 | local_adam_update | 2.998 | 10.3% | nanx |
| toy5 | cuda | 1024 | social_step | 9.057 | 34.9% | nanx |
| toy5 | cuda | 1024 | social_distillation | 9.023 | 34.7% | nanx |
| toy5 | cuda | 1024 | social_mix | 6.675 | 23.1% | nanx |
| toy5 | cuda | 1024 | local_loss_update | 2.791 | 14.3% | nanx |
| toy5 | cuda | 1024 | local_optimizer_update | 2.014 | 10.4% | nanx |
| toy5 | cuda | 1024 | social_optimizer_update | 1.525 | 7.4% | nanx |
| toy5 | cuda | 2048 | social_step | 3.381 | 14.5% | nanx |
| toy5 | cuda | 2048 | social_distillation | 3.351 | 14.4% | nanx |
| toy5 | cuda | 2048 | local_loss_update | 2.516 | 10.8% | nanx |
| toy5 | cuda | 2048 | local_optimizer_update | 1.755 | 7.6% | nanx |
| toy5 | cuda | 2048 | social_optimizer_update | 1.660 | 7.1% | nanx |
| toy5 | cuda | 2048 | social_mix | 0.971 | 4.2% | nanx |

## Tensor stages slower than batched

| toy | device | agents | stage | delta ms/epoch | stage speedup |
|---|---:|---:|---|---:|---:|
