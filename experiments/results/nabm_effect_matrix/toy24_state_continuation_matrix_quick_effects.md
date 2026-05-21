# NABM Effect Matrix

## Grouped Effects

| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| toy2_state_continuation_vs_reputation | toy2 | final_mean_payoff | maximize | 3 | 2.49125 | -0.50875 | 0.0224546 |
| toy4_state_continuation_vs_reputation | toy4 | final_mean_payoff | maximize | 0.6 | 0.302 | -0.298 | 0.00706688 |

## Pairwise Baseline Effects

| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI |
| --- | --- | --- | --- | ---: | ---: |
| toy2_state_continuation_vs_reputation | toy2 | reputation_imitation | neural_continuation_balanced | -0.00333333 | 0.00653333 |
| toy2_state_continuation_vs_reputation | toy2 | reputation_imitation | neural_continuation_social_heavy | -0.0916667 | 0.0397407 |
| toy2_state_continuation_vs_reputation | toy2 | reputation_imitation | neural_continuation_welfare_heavy | 0 | 0 |
| toy2_state_continuation_vs_reputation | toy2 | reputation_imitation | neural_material_output_average | -1.94 | 0.0898185 |
| toy4_state_continuation_vs_reputation | toy4 | reputation_imitation | neural_continuation_balanced | -0.004 | 0.00392 |
| toy4_state_continuation_vs_reputation | toy4 | reputation_imitation | neural_continuation_social_heavy | -0.592 | 0.01568 |
| toy4_state_continuation_vs_reputation | toy4 | reputation_imitation | neural_continuation_welfare_heavy | -0.004 | 0.00392 |
| toy4_state_continuation_vs_reputation | toy4 | reputation_imitation | neural_material_output_average | -0.592 | 0.01568 |

Positive effect values favor the NABM group.
