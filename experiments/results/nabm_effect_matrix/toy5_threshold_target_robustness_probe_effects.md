# NABM Effect Matrix

## Grouped Effects

| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| no_seed_heterogeneous_safety_probe | toy5 | domain_non_adoption_rate | maximize | 1 | 1 | 0 | 0 | ceiling_tie_equal_time | 1/1 | 0/0 | 0/0 | 1/1 |
| frontier_random_seed_probe | toy5 | domain_cascade_size | maximize | 1 | 100 | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /32.3333 | 0/0 | 0/1 |
| frontier_lattice_probe | toy5 | domain_cascade_size | maximize | 1 | 65 | 64 | 0 | ceiling_tie | 0/0 | / | 0/0 | 0/0 |
| frontier_heterogeneous_probe | toy5 | domain_cascade_size | maximize | 1 | 100 | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /31.6667 | 0/0 | 0/1 |

## Pairwise Baseline Effects

| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| no_seed_heterogeneous_safety_probe | toy5 | baseline | exposure_anchor | 0 | 0 | ceiling_tie_equal_time | 1/1 | 0/0 | 0/0 | 1/1 |
| frontier_random_seed_probe | toy5 | baseline | exposure_anchor | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /32.3333 | 0/0 | 0/1 |
| frontier_lattice_probe | toy5 | baseline | exposure_anchor | 64 | 0 | ceiling_tie | 0/0 | / | 0/0 | 0/0 |
| frontier_heterogeneous_probe | toy5 | baseline | exposure_anchor | 99 | 0 | nabm_more_final_ceiling_hits | 0/1 | /31.6667 | 0/0 | 0/1 |

Positive effect values favor the NABM group.
