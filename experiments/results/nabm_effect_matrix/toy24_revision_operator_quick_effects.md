# NABM Effect Matrix

## Grouped Effects

| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| toy2_revision_operator | toy2 | final_mean_payoff | maximize | 3 | 2.99667 | -0.00333333 | 0.00653333 | baseline_more_final_ceiling_hits | 1/0.666667 | 13.1667/19.3333 | 0/0.333333 | 0.933333/0.866667 |
| toy4_revision_operator | toy4 | final_mean_payoff | maximize | 0.598 | 0.596 | -0.002 | 0.00196 | baseline_more_final_ceiling_hits | 0.666667/0.333333 | 11.8333/19 | 0.333333/0.666667 | 0.866667/0.8 |

## Pairwise Baseline Effects

| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| toy2_revision_operator | toy2 | linear_welfare_heavy | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | -0.00333333 | 0.00653333 | baseline_more_final_ceiling_hits | 1/0.666667 | 23.6667/19.3333 | 0/0.333333 | 0.866667/0.866667 |
| toy2_revision_operator | toy2 | reputation_imitation | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | -0.00333333 | 0.00653333 | baseline_more_final_ceiling_hits | 1/0.666667 | 2.66667/19.3333 | 0/0.333333 | 1/0.866667 |
| toy4_revision_operator | toy4 | linear_welfare_heavy | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | 0 | 0 | nabm_faster_to_ceiling | 0.333333/0.333333 | 21/19 | 0.666667/0.666667 | 0.733333/0.8 |
| toy4_revision_operator | toy4 | reputation_imitation | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | -0.004 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/19 | 0/0.666667 | 1/0.8 |

Positive effect values favor the NABM group.
