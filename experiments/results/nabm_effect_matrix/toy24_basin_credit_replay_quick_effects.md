# NABM Effect Matrix

## Grouped Effects

| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| toy2_basin_credit | toy2 | final_mean_payoff | maximize | 3 | 2.88917 | -0.110833 | 0.013143 | baseline_more_final_ceiling_hits | 1/0.75 | 13.1667/14.1111 |
| toy4_basin_credit | toy4 | final_mean_payoff | maximize | 0.598 | 0.5275 | -0.0705 | 0.0220663 | baseline_more_final_ceiling_hits | 0.666667/0.583333 | 11.8333/13.8889 |

## Pairwise Baseline Effects

| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| toy2_basin_credit | toy2 | linear_welfare_heavy | basin_credit_w1p0_h1_prototype | -0.443333 | 0.052572 | baseline_more_final_ceiling_hits | 1/0 | 23.6667/ |
| toy2_basin_credit | toy2 | linear_welfare_heavy | mixed_objective_basin_replay_all_p2_h1 | 0 | 0 | nabm_faster_to_ceiling | 1/1 | 23.6667/10.3333 |
| toy2_basin_credit | toy2 | linear_welfare_heavy | mixed_objective_basin_replay_all_p3_h1 | 0 | 0 | nabm_faster_to_ceiling | 1/1 | 23.6667/9.33333 |
| toy2_basin_credit | toy2 | linear_welfare_heavy | mixed_objective_basin_w0p5_0p5_h1 | 0 | 0 | nabm_faster_to_ceiling | 1/1 | 23.6667/22.6667 |
| toy2_basin_credit | toy2 | reputation_imitation | basin_credit_w1p0_h1_prototype | -0.443333 | 0.052572 | baseline_more_final_ceiling_hits | 1/0 | 2.66667/ |
| toy2_basin_credit | toy2 | reputation_imitation | mixed_objective_basin_replay_all_p2_h1 | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/10.3333 |
| toy2_basin_credit | toy2 | reputation_imitation | mixed_objective_basin_replay_all_p3_h1 | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/9.33333 |
| toy2_basin_credit | toy2 | reputation_imitation | mixed_objective_basin_w0p5_0p5_h1 | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/22.6667 |
| toy4_basin_credit | toy4 | linear_welfare_heavy | basin_credit_w1p0_h1_prototype | -0.282 | 0.0882653 | baseline_more_final_ceiling_hits | 0.333333/0 | 21/ |
| toy4_basin_credit | toy4 | linear_welfare_heavy | mixed_objective_basin_replay_all_p2_h1 | 0.004 | 0.00392 | nabm_more_final_ceiling_hits | 0.333333/1 | 21/13.6667 |
| toy4_basin_credit | toy4 | linear_welfare_heavy | mixed_objective_basin_replay_all_p3_h1 | 0.002 | 0.00392 | nabm_more_final_ceiling_hits | 0.333333/0.666667 | 21/11.6667 |
| toy4_basin_credit | toy4 | linear_welfare_heavy | mixed_objective_basin_w0p5_0p5_h1 | 0.002 | 0.00392 | nabm_more_final_ceiling_hits | 0.333333/0.666667 | 21/16.3333 |
| toy4_basin_credit | toy4 | reputation_imitation | basin_credit_w1p0_h1_prototype | -0.286 | 0.0843483 | baseline_more_final_ceiling_hits | 1/0 | 2.66667/ |
| toy4_basin_credit | toy4 | reputation_imitation | mixed_objective_basin_replay_all_p2_h1 | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/13.6667 |
| toy4_basin_credit | toy4 | reputation_imitation | mixed_objective_basin_replay_all_p3_h1 | -0.002 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.666667 | 2.66667/11.6667 |
| toy4_basin_credit | toy4 | reputation_imitation | mixed_objective_basin_w0p5_0p5_h1 | -0.002 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.666667 | 2.66667/16.3333 |

Positive effect values favor the NABM group.
