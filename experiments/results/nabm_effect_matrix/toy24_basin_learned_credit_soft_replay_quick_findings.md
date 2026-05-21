# Toy2/Toy4 Learned Basin-Credit Soft Replay Findings

## Scope

This slice replaces the hard learned-credit replay include/drop decision with a
soft per-agent replay loss weight:

```text
learned_credit_replay_mode: soft_attention
learned_credit_replay_selection: confident_agreement
learned_credit_replay_soft_min_weight: 0.50
learned_credit_replay_soft_disagreement_weight: 0.25
```

The learned critic and fallback semantics are unchanged. The point of this
slice is narrower: keep every eligible replay candidate trainable while letting
critic confidence and prototype/learned relation scale the update.

Manifest:

```text
experiments/evidence/toy24_basin_learned_credit_soft_replay_quick.yaml
```

Diagnostic summary:

```text
experiments/results/basin_critic/toy24_basin_learned_credit_soft_replay_quick_learned_diagnostic_summary.md
```

## Evidence Gate Result

The evidence gate passed.

```text
experiments/evidence/results/toy24_basin_learned_credit_soft_replay_quick.summary.json
experiments/evidence/results/toy24_basin_learned_credit_soft_replay_quick.summary.md
```

Main `soft_min50` results:

| Toy | Final ceiling hits | Mean TtC | Gate threshold | Replay selected | Replay weight | Training learned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | 3/3 | 9.33 | < 10 | 1.0000 | 0.5416 | 0.8038 |
| Toy4 | 3/3 | 11.67 | < 12 | 1.0000 | 0.8006 | 0.8667 |

## Variant Comparison

| Toy | Variant | Group | Final hits | Mean TtC | Final metric | Replay selected | Replay weight |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Toy2 | linear_welfare_heavy | baseline | 3/3 | 23.67 | 3.000 |  |  |
| Toy2 | prototype escalation | diagnostic | 3/3 | 9.33 | 3.000 | 1.0000 | 1.0000 |
| Toy2 | learned candidate-context all replay | diagnostic | 3/3 | 9.33 | 3.000 | 1.0000 | 1.0000 |
| Toy2 | learned confident agreement | diagnostic | 0/3 |  | 2.328 | 0.0000 | 0.0000 |
| Toy2 | learned confident agreement curriculum floor50 d30 | diagnostic | 3/3 | 9.33 | 3.000 | 0.6528 | 0.6528 |
| Toy2 | learned confident agreement soft min25 | diagnostic | 3/3 | 9.33 | 3.000 | 1.0000 | 0.3100 |
| Toy2 | learned confident agreement soft min50 | nabm | 3/3 | 9.33 | 3.000 | 1.0000 | 0.5416 |
| Toy4 | linear_welfare_heavy | baseline | 1/3 | 21.00 | 0.596 |  |  |
| Toy4 | prototype escalation | diagnostic | 3/3 | 11.67 | 0.600 | 1.0000 | 1.0000 |
| Toy4 | learned candidate-context all replay | diagnostic | 3/3 | 11.67 | 0.600 | 1.0000 | 1.0000 |
| Toy4 | learned confident agreement | diagnostic | 0/3 |  | 0.298 | 0.0039 | 0.0039 |
| Toy4 | learned confident agreement curriculum floor50 d30 | diagnostic | 3/3 | 11.67 | 0.600 | 0.9929 | 0.9929 |
| Toy4 | learned confident agreement soft min25 | diagnostic | 3/3 | 12.33 | 0.600 | 1.0000 | 0.6969 |
| Toy4 | learned confident agreement soft min50 | nabm | 3/3 | 11.67 | 0.600 | 1.0000 | 0.8006 |

## Interpretation

The soft replay result supports the current structural direction, but it should
not be overstated.

- Hard `confident_agreement` replay still fails by starving the basin update.
- Soft weighting avoids starvation without reverting to full-strength all
  replay: selected rate is `1.0`, but mean replay weight is lower than `1.0`.
- The main `soft_min50` candidate matches the prototype/all-replay quick gate on
  both toys, including Toy4 mean time-to-ceiling `11.67`.
- The lower `soft_min25` diagnostic is viable in Toy2 but too slow for Toy4
  under the existing `< 12` time-to-ceiling criterion.

The defensible conclusion is:

> For this slice, learned-credit replay should be controlled as a soft loss
> weight rather than a hard candidate gate. This is closer to a credit-attention
> mechanism than the previous replay floor, but it is still a lightweight
> policy around the critic, not a Transformer-style learned attention module.

The next structural step is to move the replay weight computation into a learned
or calibrated scorer over richer basin-phase state, instead of only combining
margin, uncertainty, and agreement with fixed coefficients.
