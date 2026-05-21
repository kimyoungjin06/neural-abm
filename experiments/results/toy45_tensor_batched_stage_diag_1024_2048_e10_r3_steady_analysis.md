# Toy steady-state timing analysis

Input: `experiments/results/toy45_tensor_batched_stage_diag_1024_2048_e10_r3_stage_timings.csv`

`steady_state_total` is `sample_revision_mask + hooked_step_total`; setup, initial aggregate, per-epoch aggregate, and write stages are excluded. Child stages are nested diagnostics and should not be summed.

## Steady-state backend speedup

| toy | device | agents | batched ms/epoch | tensor ms/epoch | speedup |
|---|---:|---:|---:|---:|---:|
| toy4 | cpu | 1024 | 317.768 | 61.135 | 5.198x |
| toy4 | cpu | 2048 | 703.323 | 105.615 | 6.659x |
| toy4 | cuda | 1024 | 365.332 | 54.306 | 6.727x |
| toy4 | cuda | 2048 | 1119.666 | 71.775 | 15.600x |
| toy5 | cpu | 1024 | 286.665 | 73.566 | 3.897x |
| toy5 | cpu | 2048 | 765.668 | 51.220 | 14.949x |
| toy5 | cuda | 1024 | 496.326 | 55.780 | 8.898x |
| toy5 | cuda | 2048 | 468.909 | 35.254 | 13.301x |

## Top tensor_batched focus stages

| toy | device | agents | stage | tensor ms/epoch | hook share | stage speedup |
|---|---:|---:|---|---:|---:|---:|
| toy4 | cpu | 1024 | local_loss_update | 18.282 | 30.1% | 6.308x |
| toy4 | cpu | 1024 | local_optimizer_update | 15.035 | 24.8% | 4.412x |
| toy4 | cpu | 1024 | social_step | 10.639 | 17.5% | 15.075x |
| toy4 | cpu | 1024 | social_distillation | 10.605 | 17.4% | 15.116x |
| toy4 | cpu | 1024 | local_adam_update | 9.852 | 16.3% | 6.108x |
| toy4 | cpu | 1024 | social_optimizer_update | 6.424 | 10.6% | 23.027x |
| toy4 | cpu | 2048 | local_loss_update | 25.423 | 24.1% | 6.995x |
| toy4 | cpu | 2048 | local_optimizer_update | 20.792 | 19.7% | 5.340x |
| toy4 | cpu | 2048 | social_step | 20.450 | 19.4% | 22.747x |
| toy4 | cpu | 2048 | social_distillation | 20.393 | 19.3% | 22.807x |
| toy4 | cpu | 2048 | local_adam_update | 13.931 | 13.2% | 7.490x |
| toy4 | cpu | 2048 | social_optimizer_update | 11.845 | 11.2% | 38.016x |
| toy4 | cuda | 1024 | local_loss_update | 13.274 | 24.6% | 9.196x |
| toy4 | cuda | 1024 | local_optimizer_update | 10.185 | 18.9% | 7.834x |
| toy4 | cuda | 1024 | social_step | 8.918 | 16.4% | 22.378x |
| toy4 | cuda | 1024 | social_distillation | 8.858 | 16.3% | 22.521x |
| toy4 | cuda | 1024 | social_optimizer_update | 4.882 | 9.0% | 38.187x |
| toy4 | cuda | 1024 | policy_readout | 4.355 | 8.0% | 0.963x |
| toy4 | cuda | 2048 | local_loss_update | 11.925 | 16.1% | 19.061x |
| toy4 | cuda | 2048 | social_step | 9.536 | 12.3% | 86.837x |
| toy4 | cuda | 2048 | social_distillation | 9.475 | 12.2% | 87.384x |
| toy4 | cuda | 2048 | local_optimizer_update | 8.962 | 12.0% | 13.724x |
| toy4 | cuda | 2048 | social_optimizer_update | 4.828 | 6.1% | 168.359x |
| toy4 | cuda | 2048 | policy_readout | 3.621 | 4.9% | 1.233x |
| toy5 | cpu | 1024 | local_loss_update | 23.061 | 31.7% | 3.791x |
| toy5 | cpu | 1024 | social_step | 22.969 | 30.9% | 7.470x |
| toy5 | cpu | 1024 | social_distillation | 22.905 | 30.8% | 7.486x |
| toy5 | cpu | 1024 | local_optimizer_update | 18.328 | 25.2% | 3.623x |
| toy5 | cpu | 1024 | social_optimizer_update | 11.114 | 15.0% | 14.244x |
| toy5 | cpu | 1024 | local_adam_update | 10.814 | 14.8% | 5.654x |
| toy5 | cpu | 2048 | local_loss_update | 14.762 | 28.5% | 12.327x |
| toy5 | cpu | 2048 | local_optimizer_update | 12.188 | 23.4% | 10.951x |
| toy5 | cpu | 2048 | social_step | 11.260 | 21.9% | 47.947x |
| toy5 | cpu | 2048 | social_distillation | 11.228 | 21.8% | 48.075x |
| toy5 | cpu | 2048 | local_adam_update | 7.641 | 14.7% | 16.033x |
| toy5 | cpu | 2048 | social_optimizer_update | 6.622 | 12.9% | 78.368x |
| toy5 | cuda | 1024 | social_step | 15.367 | 27.4% | 23.295x |
| toy5 | cuda | 1024 | social_distillation | 15.177 | 27.1% | 23.577x |
| toy5 | cuda | 1024 | local_loss_update | 14.940 | 26.9% | 7.140x |
| toy5 | cuda | 1024 | local_optimizer_update | 12.268 | 22.0% | 5.525x |
| toy5 | cuda | 1024 | social_optimizer_update | 6.746 | 12.1% | 44.774x |
| toy5 | cuda | 1024 | social_mix | 4.805 | 8.8% | 9.420x |
| toy5 | cuda | 2048 | local_loss_update | 6.070 | 17.2% | 21.081x |
| toy5 | cuda | 2048 | social_step | 5.763 | 16.3% | 54.593x |
| toy5 | cuda | 2048 | social_distillation | 5.718 | 16.2% | 55.015x |
| toy5 | cuda | 2048 | local_optimizer_update | 4.548 | 12.9% | 13.424x |
| toy5 | cuda | 2048 | social_optimizer_update | 2.829 | 8.0% | 109.024x |
| toy5 | cuda | 2048 | social_mix | 1.790 | 5.1% | 1.058x |

## Tensor stages slower than batched

| toy | device | agents | stage | delta ms/epoch | stage speedup |
|---|---:|---:|---|---:|---:|
| toy5 | cpu | 1024 | policy_readout | 2.266 | 0.661x |
| toy4 | cpu | 2048 | policy_readout | 1.097 | 0.847x |
| toy5 | cpu | 1024 | social_mix | 0.906 | 0.772x |
| toy5 | cuda | 1024 | select_peers | 0.414 | 0.551x |
| toy4 | cpu | 2048 | social_mix | 0.342 | 0.863x |
| toy4 | cuda | 1024 | policy_readout | 0.162 | 0.963x |
| toy4 | cpu | 2048 | select_peers | 0.096 | 0.730x |
| toy5 | cuda | 2048 | select_peers | 0.019 | 0.928x |
