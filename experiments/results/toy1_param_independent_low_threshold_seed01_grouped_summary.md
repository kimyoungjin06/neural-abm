# Toy 1 Alpha/Threshold Sweep: toy1_param_independent_low_threshold_seed01

| Case | Alpha | Threshold | Seeds | Accuracy Mean | Consensus Mean | Fragmentation Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `parameter_average_state_similarity_independent_init` | 0 | -0.2 | 1 | 0.891972 | 0.949434 | 1.00 |
| `parameter_average_state_similarity_independent_init` | 0 | 0 | 1 | 0.891972 | 0.949434 | 4.00 |
| `parameter_average_state_similarity_independent_init` | 0 | 0.2 | 1 | 0.891972 | 0.949434 | 32.00 |
| `parameter_average_state_similarity_independent_init` | 0 | 0.4 | 1 | 0.891972 | 0.949434 | 49.00 |
| `parameter_average_state_similarity_independent_init` | 0 | 0.6 | 1 | 0.891972 | 0.949434 | 50.00 |
| `parameter_average_state_similarity_independent_init` | 0.1 | -0.2 | 1 | 0.895616 | 0.966932 | 1.00 |
| `parameter_average_state_similarity_independent_init` | 0.1 | 0 | 1 | 0.895828 | 0.967744 | 1.00 |
| `parameter_average_state_similarity_independent_init` | 0.1 | 0.2 | 1 | 0.892396 | 0.956676 | 31.00 |
| `parameter_average_state_similarity_independent_init` | 0.1 | 0.4 | 1 | 0.891940 | 0.949753 | 49.00 |
| `parameter_average_state_similarity_independent_init` | 0.1 | 0.6 | 1 | 0.891972 | 0.949434 | 50.00 |
| `parameter_average_state_similarity_independent_init` | 0.25 | -0.2 | 1 | 0.897608 | 0.980263 | 1.00 |
| `parameter_average_state_similarity_independent_init` | 0.25 | 0 | 1 | 0.897956 | 0.979239 | 1.00 |
| `parameter_average_state_similarity_independent_init` | 0.25 | 0.2 | 1 | 0.894148 | 0.959692 | 29.00 |
| `parameter_average_state_similarity_independent_init` | 0.25 | 0.4 | 1 | 0.891976 | 0.949869 | 49.00 |
| `parameter_average_state_similarity_independent_init` | 0.25 | 0.6 | 1 | 0.891972 | 0.949434 | 50.00 |
| `parameter_average_state_similarity_independent_init` | 0.5 | -0.2 | 1 | 0.897308 | 0.987669 | 1.00 |
| `parameter_average_state_similarity_independent_init` | 0.5 | 0 | 1 | 0.897012 | 0.989509 | 1.00 |
| `parameter_average_state_similarity_independent_init` | 0.5 | 0.2 | 1 | 0.893944 | 0.960872 | 28.00 |
| `parameter_average_state_similarity_independent_init` | 0.5 | 0.4 | 1 | 0.892040 | 0.949853 | 49.00 |
| `parameter_average_state_similarity_independent_init` | 0.5 | 0.6 | 1 | 0.891972 | 0.949434 | 50.00 |

## Initial Readout

This sweep is intended to inspect phase behavior across social influence strength and peer threshold. Treat single-seed sweeps as pilot results.

## Pilot Interpretation

- Independent initialization does not make parameter averaging impossible by
  itself; the previous `0.6+` thresholds were simply too strict for raw
  parameter cosine similarity.
- The effective transition is sharp in this seed: fragmentation is connected at
  threshold `-0.2`, mostly connected at `0.0`, partially fragmented around
  `0.2`, and effectively dead by `0.4`.
- Best accuracy in this pilot is `0.897956` at `alpha = 0.25`,
  `threshold = 0.0`.
- Higher `alpha` increases consensus strongly, but does not improve accuracy
  beyond the `alpha = 0.25` setting in this seed.
- This supports two next tests: multi-seed low-threshold validation and a proper
  parameter-alignment ablation.
