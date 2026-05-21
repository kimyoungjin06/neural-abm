# Toy 1 Parameter Alignment Diagnostic

Source grouped CSV:

`experiments/results/toy1_param_alignment_diagnostic_seeds01_05_grouped_summary.csv`

Comparison baseline:

`experiments/results/toy1_param_independent_low_threshold_seeds01_05_grouped_summary.csv`

## Setup

- Toy model: Neural HK Classification.
- Init mode: `independent_init`.
- Alpha: `0.25`.
- Seeds: `1-5`.
- New mixer: `parameter_aligned_average`.
- New peer rule: `aligned_state_similarity`.

## Key Result

| Method | Threshold | Accuracy Mean | Consensus Mean | Fragmentation Mean |
| --- | ---: | ---: | ---: | ---: |
| Raw parameter average + raw peers | 0.2 | 0.890282 | 0.958260 | 26.00 |
| Aligned parameter average + raw peers | 0.2 | 0.890865 | 0.962082 | 26.80 |
| Aligned parameter average + aligned peers | 0.2 | 0.893686 | 0.983375 | 1.00 |
| Raw parameter average + raw peers | 0.6 | 0.887794 | 0.947597 | 50.00 |
| Aligned parameter average + raw peers | 0.6 | 0.887794 | 0.947597 | 50.00 |
| Aligned parameter average + aligned peers | 0.6 | 0.893695 | 0.984123 | 1.00 |
| Raw parameter average + raw peers | 0.8 | 0.887794 | 0.947597 | 50.00 |
| Aligned parameter average + raw peers | 0.8 | 0.887794 | 0.947597 | 50.00 |
| Aligned parameter average + aligned peers | 0.8 | 0.892145 | 0.972957 | 9.80 |

## Interpretation

- Aligning peer weights before averaging gives a small improvement when the raw
  peer graph is already connected.
- It does not fix fragmentation if peer selection still uses raw parameter
  cosine similarity.
- Aligning the peer-selection similarity itself shifts the fragmentation
  transition substantially upward: thresholds `0.2` and `0.6` remain connected,
  and even `0.8` is only partially fragmented.
- The independent-init parameter failure is therefore not just a poor averaging
  operator; raw parameter similarity is also a weak social-neighbor criterion.

Figure:

`paper/figures/toy1_parameter_alignment_diagnostic.png`
