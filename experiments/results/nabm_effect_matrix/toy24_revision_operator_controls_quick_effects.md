# NABM Effect Matrix

## Grouped Effects

| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| toy2_revision_operator_controls | toy2 | final_mean_payoff | maximize | 3 | 3 | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/17.8889 | 0/0 | 1/0.977778 |
| toy4_revision_operator_controls | toy4 | final_mean_payoff | maximize | 0.6 | 0.6 | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/17.2222 | 0/0 | 1/0.955556 |

## Pairwise Baseline Effects

| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| toy2_revision_operator_controls | toy2 | reputation_imitation | revision_operator_commitment_hysteresis | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/15 | 0/0 | 1/1 |
| toy2_revision_operator_controls | toy2 | reputation_imitation | revision_operator_terminal_argmax_k1 | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/19.3333 | 0/0 | 1/0.933333 |
| toy2_revision_operator_controls | toy2 | reputation_imitation | revision_operator_terminal_argmax_k5 | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/19.3333 | 0/0 | 1/1 |
| toy4_revision_operator_controls | toy4 | reputation_imitation | revision_operator_commitment_hysteresis | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/13.6667 | 0/0 | 1/0.933333 |
| toy4_revision_operator_controls | toy4 | reputation_imitation | revision_operator_terminal_argmax_k1 | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/19 | 0/0 | 1/0.933333 |
| toy4_revision_operator_controls | toy4 | reputation_imitation | revision_operator_terminal_argmax_k5 | 0 | 0 | baseline_faster_to_ceiling | 1/1 | 2.66667/19 | 0/0 | 1/1 |

Positive effect values favor the NABM group.
