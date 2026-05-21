# NABM Effect Matrix

## Grouped Effects

| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| toy5_threshold_aware_wavefront_no_seed_heterogeneous_safety | toy5 | domain_non_adoption_rate | maximize | 1 | 1 | 0 | 0 | ceiling_tie_equal_time | 1/1 | 0/0 | 0/0 | 1/1 |
| toy5_threshold_aware_lattice_k4_heterogeneous_h0p85_spread | toy5 | domain_cascade_size | maximize | 1 | 100 | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /36.2 | 0/0 | 0/1 |
| toy5_threshold_aware_lattice_k6_heterogeneous_h0p95_spread | toy5 | domain_cascade_size | maximize | 1 | 100 | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /25 | 0/0 | 0/1 |
| toy5_threshold_aware_rewired_p0p10_heterogeneous_h0p95_spread | toy5 | domain_cascade_size | maximize | 1 | 100 | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /10 | 0/0 | 0/1 |

## Pairwise Baseline Effects

| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| toy5_threshold_aware_wavefront_no_seed_heterogeneous_safety | toy5 | neural_threshold_aware_wavefront_no_seed_output_average | neural_threshold_aware_wavefront_no_seed_threshold_anchor | 0 | 0 | ceiling_tie_equal_time | 1/1 | 0/0 | 0/0 | 1/1 |
| toy5_threshold_aware_lattice_k4_heterogeneous_h0p85_spread | toy5 | neural_threshold_aware_lattice_k4_h0p85_output_average | neural_threshold_aware_lattice_k4_h0p85_threshold_anchor | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /36.2 | 0/0 | 0/1 |
| toy5_threshold_aware_lattice_k6_heterogeneous_h0p95_spread | toy5 | neural_threshold_aware_lattice_k6_h0p95_output_average | neural_threshold_aware_lattice_k6_h0p95_threshold_anchor | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /25 | 0/0 | 0/1 |
| toy5_threshold_aware_rewired_p0p10_heterogeneous_h0p95_spread | toy5 | neural_threshold_aware_rewired_p0p10_h0p95_output_average | neural_threshold_aware_rewired_p0p10_h0p95_threshold_anchor | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /10 | 0/0 | 0/1 |

Positive effect values favor the NABM group.
