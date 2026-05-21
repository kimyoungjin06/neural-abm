# Toy 1 Ablation Summary: toy1_first_ablation_seed01

| Case | Seed | Accuracy | Consensus | Fragmentation | Run |
| --- | ---: | ---: | ---: | ---: | --- |
| none_none_same_init | 1 | 0.896592 | 0.960864 | 50 | `experiments/runs/20260429_144808_toy1_first_ablation_seed01_none_none_same_init_seed01` |
| output_average_output_similarity_same_init | 1 | 0.897808 | 0.967374 | 1 | `experiments/runs/20260429_144816_toy1_first_ablation_seed01_output_average_output_similarity_same_init_seed01` |
| latent_average_state_similarity_same_init | 1 | 0.897748 | 0.964694 | 1 | `experiments/runs/20260429_144830_toy1_first_ablation_seed01_latent_average_state_similarity_same_init_seed01` |
| parameter_average_state_similarity_same_init | 1 | 0.902180 | 0.982808 | 1 | `experiments/runs/20260429_144840_toy1_first_ablation_seed01_parameter_average_state_similarity_same_init_seed01` |
| parameter_average_state_similarity_independent_init | 1 | 0.891972 | 0.949434 | 50 | `experiments/runs/20260429_144848_toy1_first_ablation_seed01_parameter_average_state_similarity_independent_init_seed01` |

This is an automated first ablation summary.

## Initial Readout

This is a single-seed result, so it should not be treated as evidence yet.
Still, it validates that the ablation workflow can separate several expected
behaviors:

- `output_average` and `latent_average` slightly improve accuracy and consensus
  over no-social in this seed.
- `parameter_average` with `same_init` gives the strongest accuracy and
  consensus in this seed.
- `parameter_average` with `independent_init` fails the state-similarity peer
  threshold for most agents, producing 50 filtered components and lower
  accuracy.

The next check should run seeds `1 2 3 4 5` and summarize means and standard
deviations before making claims.
