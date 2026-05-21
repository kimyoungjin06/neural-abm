# NABM Effect Matrix

## Grouped Effects

| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| toy2_basin_learned_credit_replay_floor | toy2 | final_mean_payoff | maximize | 3 | 3 | 0 | 0 | nabm_faster_to_ceiling | 1/1 | 23.6667/16 |
| toy4_basin_learned_credit_replay_floor | toy4 | final_mean_payoff | maximize | 0.596 | 0.6 | 0.004 | 0.00392 | nabm_more_final_ceiling_hits | 0.333333/1 | 21/14.6667 |

## Pairwise Baseline Effects

| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| toy2_basin_learned_credit_replay_floor | toy2 | linear_welfare_heavy | learned_candidate_context_confident_agreement_floor50_replay | 0 | 0 | nabm_faster_to_ceiling | 1/1 | 23.6667/16 |
| toy4_basin_learned_credit_replay_floor | toy4 | linear_welfare_heavy | learned_candidate_context_confident_agreement_floor50_replay | 0.004 | 0.00392 | nabm_more_final_ceiling_hits | 0.333333/1 | 21/14.6667 |

Positive effect values favor the NABM group.
