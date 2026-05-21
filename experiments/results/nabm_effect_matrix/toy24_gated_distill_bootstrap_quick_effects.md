# NABM Effect Matrix

## Grouped Effects

| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| toy2_gated_distill_bootstrap | toy2 | final_mean_payoff | maximize | 3 | 3 | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/27.6111 |
| toy4_gated_distill_bootstrap | toy4 | final_mean_payoff | maximize | 0.6 | 0.596 | -0.004 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/25 |

## Pairwise Baseline Effects

| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| toy2_gated_distill_bootstrap | toy2 | reputation_imitation | distill_bootstrap_w1p0_e5_linear_welfare | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/26.6667 |
| toy2_gated_distill_bootstrap | toy2 | reputation_imitation | gated_distill_bootstrap_w1p0_e5_linear_welfare | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/26.6667 |
| toy2_gated_distill_bootstrap | toy2 | reputation_imitation | linear_welfare_heavy | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/23.6667 |
| toy2_gated_distill_bootstrap | toy2 | reputation_imitation | stable_distill_bootstrap_w1p0_e5_linear_welfare | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/26.6667 |
| toy2_gated_distill_bootstrap | toy2 | reputation_imitation | stable_gated_distill_bootstrap_w1p0_e5_linear_welfare | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/26.6667 |
| toy2_gated_distill_bootstrap | toy2 | reputation_imitation | target_stable_gated_distill_w1p0_e5_linear_welfare | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/35.3333 |
| toy4_gated_distill_bootstrap | toy4 | reputation_imitation | distill_bootstrap_w1p0_e5_linear_welfare | -0.004 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/24.3333 |
| toy4_gated_distill_bootstrap | toy4 | reputation_imitation | gated_distill_bootstrap_w1p0_e5_linear_welfare | -0.004 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/22.3333 |
| toy4_gated_distill_bootstrap | toy4 | reputation_imitation | linear_welfare_heavy | -0.004 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/21 |
| toy4_gated_distill_bootstrap | toy4 | reputation_imitation | stable_distill_bootstrap_w1p0_e5_linear_welfare | -0.004 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/22.3333 |
| toy4_gated_distill_bootstrap | toy4 | reputation_imitation | stable_gated_distill_bootstrap_w1p0_e5_linear_welfare | -0.004 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/22.3333 |
| toy4_gated_distill_bootstrap | toy4 | reputation_imitation | target_stable_gated_distill_w1p0_e5_linear_welfare | -0.004 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/37.6667 |

Positive effect values favor the NABM group.
