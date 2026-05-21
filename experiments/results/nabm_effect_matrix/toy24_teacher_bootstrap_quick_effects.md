# NABM Effect Matrix

## Grouped Effects

| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| toy2_teacher_bootstrap | toy2 | final_mean_payoff | maximize | 3 | 2.99722 | -0.00277778 | 0.00392604 | baseline_more_final_ceiling_hits | 1/0.833333 | 2.66667/29.6111 |
| toy4_teacher_bootstrap | toy4 | final_mean_payoff | maximize | 0.6 | 0.595667 | -0.00433333 | 0.00326667 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/26.8667 |

## Pairwise Baseline Effects

| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| toy2_teacher_bootstrap | toy2 | reputation_imitation | linear_welfare_heavy | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/23.6667 |
| toy2_teacher_bootstrap | toy2 | reputation_imitation | nonlinear_interaction | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/28.6667 |
| toy2_teacher_bootstrap | toy2 | reputation_imitation | teacher_bootstrap_w0p5_e5_linear_welfare | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/23.6667 |
| toy2_teacher_bootstrap | toy2 | reputation_imitation | teacher_bootstrap_w0p5_e5_nonlinear_interaction | -0.00666667 | 0.0130667 | baseline_more_final_ceiling_hits | 1/0.666667 | 2.66667/31 |
| toy2_teacher_bootstrap | toy2 | reputation_imitation | teacher_bootstrap_w1p0_e3_linear_welfare | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/31 |
| toy2_teacher_bootstrap | toy2 | reputation_imitation | teacher_bootstrap_w1p0_e3_nonlinear_interaction | -0.01 | 0.0113161 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/39.6667 |
| toy4_teacher_bootstrap | toy4 | reputation_imitation | linear_welfare_heavy | -0.004 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/21 |
| toy4_teacher_bootstrap | toy4 | reputation_imitation | nonlinear_interaction | -0.004 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/19.3333 |
| toy4_teacher_bootstrap | toy4 | reputation_imitation | teacher_bootstrap_w0p5_e5_linear_welfare | -0.004 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/24.3333 |
| toy4_teacher_bootstrap | toy4 | reputation_imitation | teacher_bootstrap_w0p5_e5_nonlinear_interaction | -0.004 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.333333 | 2.66667/42.5 |
| toy4_teacher_bootstrap | toy4 | reputation_imitation | teacher_bootstrap_w1p0_e3_linear_welfare | -0.002 | 0.00392 | baseline_more_final_ceiling_hits | 1/0.666667 | 2.66667/25.3333 |
| toy4_teacher_bootstrap | toy4 | reputation_imitation | teacher_bootstrap_w1p0_e3_nonlinear_interaction | -0.008 | 0.00392 | baseline_more_final_ceiling_hits | 1/0 | 2.66667/48 |

Positive effect values favor the NABM group.
