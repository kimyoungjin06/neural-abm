# Toy steady-state timing analysis

Input: `experiments/results/toy45_tensor_batched_stage_diag_1024_e5_r2_stage_timings.csv`

`steady_state_total` is `sample_revision_mask + hooked_step_total`; setup, initial aggregate, per-epoch aggregate, and write stages are excluded. Child stages are nested diagnostics and should not be summed.

## Steady-state backend speedup

| toy | device | agents | batched ms/epoch | tensor ms/epoch | speedup |
|---|---:|---:|---:|---:|---:|
| toy4 | cpu | 1024 | 318.771 | 75.984 | 4.195x |
| toy4 | cuda | 1024 | 470.081 | 104.101 | 4.516x |
| toy5 | cpu | 1024 | 551.741 | 98.015 | 5.629x |
| toy5 | cuda | 1024 | 713.912 | 80.736 | 8.843x |

## Top tensor_batched focus stages

| toy | device | agents | stage | tensor ms/epoch | hook share | stage speedup |
|---|---:|---:|---|---:|---:|---:|
| toy4 | cpu | 1024 | local_loss_update | 22.748 | 30.1% | 7.196x |
| toy4 | cpu | 1024 | local_optimizer_update | 18.355 | 24.3% | 3.059x |
| toy4 | cpu | 1024 | social_step | 13.692 | 17.8% | 8.796x |
| toy4 | cpu | 1024 | social_distillation | 13.647 | 17.8% | 8.821x |
| toy4 | cpu | 1024 | local_adam_update | 10.326 | 13.7% | 4.966x |
| toy4 | cpu | 1024 | social_optimizer_update | 7.763 | 10.1% | 14.451x |
| toy4 | cuda | 1024 | local_loss_update | 27.317 | 26.4% | 5.939x |
| toy4 | cuda | 1024 | social_step | 22.499 | 21.7% | 11.496x |
| toy4 | cuda | 1024 | social_distillation | 22.356 | 21.5% | 11.566x |
| toy4 | cuda | 1024 | local_optimizer_update | 21.974 | 21.3% | 3.761x |
| toy4 | cuda | 1024 | social_optimizer_update | 12.739 | 12.3% | 18.577x |
| toy4 | cuda | 1024 | policy_readout | 8.753 | 8.4% | 0.394x |
| toy5 | cpu | 1024 | local_loss_update | 31.948 | 33.2% | 6.781x |
| toy5 | cpu | 1024 | social_step | 27.656 | 28.0% | 10.380x |
| toy5 | cpu | 1024 | social_distillation | 27.532 | 27.8% | 10.420x |
| toy5 | cpu | 1024 | local_optimizer_update | 26.444 | 27.6% | 4.907x |
| toy5 | cpu | 1024 | local_adam_update | 15.963 | 16.8% | 7.451x |
| toy5 | cpu | 1024 | social_optimizer_update | 13.685 | 13.9% | 19.271x |
| toy5 | cuda | 1024 | local_loss_update | 22.389 | 27.8% | 9.389x |
| toy5 | cuda | 1024 | social_step | 20.447 | 25.4% | 22.437x |
| toy5 | cuda | 1024 | social_distillation | 20.289 | 25.2% | 22.598x |
| toy5 | cuda | 1024 | local_optimizer_update | 17.211 | 21.4% | 5.681x |
| toy5 | cuda | 1024 | social_optimizer_update | 7.876 | 9.7% | 41.484x |
| toy5 | cuda | 1024 | social_mix | 7.578 | 9.5% | 14.979x |

## Tensor stages slower than batched

| toy | device | agents | stage | delta ms/epoch | stage speedup |
|---|---:|---:|---|---:|---:|
| toy4 | cuda | 1024 | policy_readout | 5.300 | 0.394x |
| toy4 | cpu | 1024 | policy_readout | 3.015 | 0.590x |
| toy5 | cuda | 1024 | policy_readout | 1.663 | 0.768x |
| toy4 | cpu | 1024 | social_mix | 0.840 | 0.616x |
| toy5 | cpu | 1024 | social_mix | 0.297 | 0.953x |
| toy5 | cpu | 1024 | select_peers | 0.101 | 0.868x |
| toy4 | cuda | 1024 | select_peers | 0.077 | 0.877x |
| toy4 | cpu | 1024 | select_peers | 0.020 | 0.922x |
