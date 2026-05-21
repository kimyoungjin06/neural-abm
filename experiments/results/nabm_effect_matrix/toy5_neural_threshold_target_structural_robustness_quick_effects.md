# NABM Effect Matrix

## Grouped Effects

| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| toy5_threshold_target_no_seed_heterogeneous_safety | toy5 | domain_non_adoption_rate | maximize | 1 | 1 | 0 | 0 | ceiling_tie_equal_time | 1/1 | 0/0 | 0/0 | 1/1 |
| toy5_threshold_target_random_seed_frontier_spread | toy5 | domain_cascade_size | maximize | 1 | 100 | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /32.6 | 0/0 | 0/1 |
| toy5_threshold_target_heterogeneous_frontier_spread | toy5 | domain_cascade_size | maximize | 1 | 100 | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /31 | 0/0 | 0/1 |

## Pairwise Baseline Effects

| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| toy5_threshold_target_no_seed_heterogeneous_safety | toy5 | neural_threshold_target_no_seed_heterogeneous_output_average | neural_threshold_target_no_seed_heterogeneous_exposure_anchor | 0 | 0 | ceiling_tie_equal_time | 1/1 | 0/0 | 0/0 | 1/1 |
| toy5_threshold_target_random_seed_frontier_spread | toy5 | neural_threshold_target_random_seed_frontier_output_average | neural_threshold_target_random_seed_frontier_exposure_anchor | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /32.6 | 0/0 | 0/1 |
| toy5_threshold_target_heterogeneous_frontier_spread | toy5 | neural_threshold_target_heterogeneous_frontier_output_average | neural_threshold_target_heterogeneous_frontier_exposure_anchor | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /31 | 0/0 | 0/1 |

Positive effect values favor the NABM group.
