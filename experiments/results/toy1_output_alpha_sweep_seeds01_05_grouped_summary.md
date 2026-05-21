# Toy 1 Alpha/Threshold Sweep: toy1_output_alpha_sweep_seeds01_05

Figure candidate:

`paper/figures/toy1_output_alpha_accuracy_consensus.png`

| Case | Alpha | Threshold | Seeds | Accuracy Mean | Consensus Mean | Fragmentation Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `output_average_output_similarity_same_init` | 0 | 0.8 | 5 | 0.888605 | 0.946704 | 1.00 |
| `output_average_output_similarity_same_init` | 0.1 | 0.8 | 5 | 0.890332 | 0.955479 | 1.00 |
| `output_average_output_similarity_same_init` | 0.25 | 0.8 | 5 | 0.891126 | 0.959370 | 1.00 |
| `output_average_output_similarity_same_init` | 0.5 | 0.8 | 5 | 0.891625 | 0.963434 | 1.00 |

## Initial Readout

This sweep is intended to inspect phase behavior across social influence strength and peer threshold. Treat single-seed sweeps as pilot results.

## Multi-Seed Interpretation

- Output averaging shows a monotonic alpha response over seeds `1-5`.
- Mean accuracy rises from `0.888605` at `alpha = 0.0` to `0.891625` at
  `alpha = 0.5`.
- Mean consensus rises from `0.946704` at `alpha = 0.0` to `0.963434` at
  `alpha = 0.5`.
- Fragmentation stays at `1.0`, so this result reflects influence strength
  rather than graph connectivity changes.
- This is the cleanest current candidate for a first paper figure: output social
  mixing gives a small but stable task gain and a clear consensus gain.
