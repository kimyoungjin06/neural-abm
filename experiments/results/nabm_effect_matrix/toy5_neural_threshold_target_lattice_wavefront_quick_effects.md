# NABM Effect Matrix

## Grouped Effects

| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| toy5_threshold_target_wavefront_no_seed_safety | toy5 | domain_non_adoption_rate | maximize | 1 | 1 | 0 | 0 | ceiling_tie_equal_time | 1/1 | 0/0 | 0/0 | 1/1 |
| toy5_threshold_target_lattice_wavefront_spread | toy5 | domain_cascade_size | maximize | 1 | 100 | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /23 | 0/0 | 0/1 |

## Pairwise Baseline Effects

| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| toy5_threshold_target_wavefront_no_seed_safety | toy5 | neural_threshold_target_wavefront_no_seed_output_average | neural_threshold_target_wavefront_no_seed_exposure_anchor | 0 | 0 | ceiling_tie_equal_time | 1/1 | 0/0 | 0/0 | 1/1 |
| toy5_threshold_target_lattice_wavefront_spread | toy5 | neural_threshold_target_lattice_output_average | neural_threshold_target_lattice_max_wavefront_anchor | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /23 | 0/0 | 0/1 |

Positive effect values favor the NABM group.
