# Toy steady-state timing analysis

Input: `experiments/results/toy45_tensor_batched_optimizer_patch_stage_diag_1024_2048_e10_r3_stage_timings.csv`

`steady_state_total` is `sample_revision_mask + hooked_step_total`; setup, initial aggregate, per-epoch aggregate, and write stages are excluded. Child stages are nested diagnostics and should not be summed.

## Steady-state backend speedup

| toy | device | agents | batched ms/epoch | tensor ms/epoch | speedup |
|---|---:|---:|---:|---:|---:|

## Top tensor_batched focus stages

| toy | device | agents | stage | tensor ms/epoch | hook share | stage speedup |
|---|---:|---:|---|---:|---:|---:|
| toy4 | cpu | 1024 | local_loss_update | 18.185 | 27.9% | nanx |
| toy4 | cpu | 1024 | local_optimizer_update | 13.470 | 20.6% | nanx |
| toy4 | cpu | 1024 | social_step | 11.843 | 18.1% | nanx |
| toy4 | cpu | 1024 | social_distillation | 11.801 | 18.0% | nanx |
| toy4 | cpu | 1024 | local_adam_update | 7.795 | 11.9% | nanx |
| toy4 | cpu | 1024 | social_optimizer_update | 6.198 | 9.5% | nanx |
| toy4 | cpu | 2048 | local_loss_update | 19.624 | 20.5% | nanx |
| toy4 | cpu | 2048 | social_step | 19.197 | 19.8% | nanx |
| toy4 | cpu | 2048 | social_distillation | 19.146 | 19.7% | nanx |
| toy4 | cpu | 2048 | local_optimizer_update | 14.828 | 15.5% | nanx |
| toy4 | cpu | 2048 | social_optimizer_update | 10.913 | 11.2% | nanx |
| toy4 | cpu | 2048 | local_adam_update | 10.056 | 10.5% | nanx |
| toy4 | cuda | 1024 | local_loss_update | 27.215 | 29.2% | nanx |
| toy4 | cuda | 1024 | local_optimizer_update | 19.194 | 20.9% | nanx |
| toy4 | cuda | 1024 | social_step | 15.135 | 17.6% | nanx |
| toy4 | cuda | 1024 | social_distillation | 15.067 | 17.5% | nanx |
| toy4 | cuda | 1024 | local_adam_update | 13.008 | 12.1% | nanx |
| toy4 | cuda | 1024 | social_mix | 9.062 | 8.6% | nanx |
| toy4 | cuda | 2048 | local_loss_update | 12.759 | 17.2% | nanx |
| toy4 | cuda | 2048 | social_step | 9.501 | 12.9% | nanx |
| toy4 | cuda | 2048 | social_distillation | 9.442 | 12.8% | nanx |
| toy4 | cuda | 2048 | local_optimizer_update | 9.242 | 12.5% | nanx |
| toy4 | cuda | 2048 | social_optimizer_update | 4.690 | 6.4% | nanx |
| toy4 | cuda | 2048 | policy_readout | 4.429 | 5.9% | nanx |
| toy5 | cpu | 1024 | local_loss_update | 10.074 | 28.0% | nanx |
| toy5 | cpu | 1024 | social_step | 9.079 | 25.1% | nanx |
| toy5 | cpu | 1024 | social_distillation | 9.038 | 25.0% | nanx |
| toy5 | cpu | 1024 | local_optimizer_update | 8.281 | 23.0% | nanx |
| toy5 | cpu | 1024 | local_adam_update | 5.414 | 15.0% | nanx |
| toy5 | cpu | 1024 | social_optimizer_update | 4.974 | 13.8% | nanx |
| toy5 | cpu | 2048 | local_loss_update | 14.481 | 25.3% | nanx |
| toy5 | cpu | 2048 | social_step | 14.048 | 24.0% | nanx |
| toy5 | cpu | 2048 | social_distillation | 14.005 | 23.9% | nanx |
| toy5 | cpu | 2048 | local_optimizer_update | 11.519 | 20.1% | nanx |
| toy5 | cpu | 2048 | local_adam_update | 7.996 | 14.0% | nanx |
| toy5 | cpu | 2048 | social_optimizer_update | 7.558 | 12.9% | nanx |
| toy5 | cuda | 1024 | social_step | 24.649 | 36.6% | nanx |
| toy5 | cuda | 1024 | social_distillation | 24.574 | 36.4% | nanx |
| toy5 | cuda | 1024 | social_mix | 17.867 | 22.9% | nanx |
| toy5 | cuda | 1024 | local_loss_update | 11.296 | 23.6% | nanx |
| toy5 | cuda | 1024 | local_optimizer_update | 8.861 | 18.4% | nanx |
| toy5 | cuda | 1024 | social_optimizer_update | 4.278 | 8.6% | nanx |
| toy5 | cuda | 2048 | local_loss_update | 10.538 | 20.7% | nanx |
| toy5 | cuda | 2048 | social_step | 10.145 | 19.8% | nanx |
| toy5 | cuda | 2048 | social_distillation | 10.068 | 19.7% | nanx |
| toy5 | cuda | 2048 | local_optimizer_update | 8.018 | 15.8% | nanx |
| toy5 | cuda | 2048 | social_optimizer_update | 4.229 | 8.3% | nanx |
| toy5 | cuda | 2048 | policy_readout | 3.454 | 6.8% | nanx |

## Tensor stages slower than batched

| toy | device | agents | stage | delta ms/epoch | stage speedup |
|---|---:|---:|---|---:|---:|
