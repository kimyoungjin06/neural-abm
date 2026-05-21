# Toy2/Toy4 Candidate-Context Learned Basin-Credit Replay Findings

## Scope

This slice changes the learned critic representation rather than a replay
threshold. The learned critic now receives candidate-conditioned phase features:

- `candidate_action_delta`
- `candidate_policy_delta`
- `candidate_phase_action_rate`
- `candidate_phase_policy_rate`
- `candidate_phase_consensus`

These features are recomputed for action-0 and action-1 candidate scoring, so
the critic can see how a candidate action changes the population-level phase
context instead of only seeing a flipped `candidate_action` scalar.

Critic-quality manifest:

```text
experiments/evidence/toy24_basin_phase_critic_candidate_context_quick.yaml
```

Replay manifest:

```text
experiments/evidence/toy24_basin_learned_credit_candidate_context_replay_quick.yaml
```

## Critic Quality

The candidate-context critic passed held-out quality checks.

| Toy | Eval AUC | Pairwise rank | Abstention | Uncertainty |
| --- | ---: | ---: | ---: | ---: |
| Toy2 | 0.9078 | 0.9078 | 0.1814 | 0.0303 |
| Toy4 | 0.9759 | 0.9759 | 0.1402 | 0.0136 |

Compared with the previous learned critic quality slice, Toy2 AUC increased
slightly, while Toy4 stayed essentially tied.

## Replay Gate Result

The evidence gate passed.

```text
experiments/evidence/results/toy24_basin_learned_credit_candidate_context_replay_quick.summary.json
experiments/evidence/results/toy24_basin_learned_credit_candidate_context_replay_quick.summary.md
```

Main learned candidate-context prototype-fallback results:

| Toy | Final ceiling hits | Mean time-to-ceiling | Baseline mean TtC |
| --- | ---: | ---: | ---: |
| Toy2 | 3/3 | 9.33 | 23.67 |
| Toy4 | 3/3 | 11.67 | 21.00 |

## Variant Comparison

| Toy | Variant | Group | Final hits | Mean TtC | Final metric |
| --- | --- | --- | ---: | ---: | ---: |
| Toy2 | linear_welfare_heavy | baseline | 3/3 | 23.67 | 3.000 |
| Toy2 | prototype escalation | diagnostic | 3/3 | 9.33 | 3.000 |
| Toy2 | learned candidate-context prototype fallback | nabm | 3/3 | 9.33 | 3.000 |
| Toy2 | learned candidate-context zero fallback | diagnostic | 3/3 | 9.33 | 3.000 |
| Toy4 | linear_welfare_heavy | baseline | 1/3 | 21.00 | 0.596 |
| Toy4 | prototype escalation | diagnostic | 3/3 | 11.67 | 0.600 |
| Toy4 | learned candidate-context prototype fallback | nabm | 3/3 | 11.67 | 0.600 |
| Toy4 | learned candidate-context zero fallback | diagnostic | 3/3 | 11.67 | 0.600 |

## Learned Source Usage

Average over all aggregate rows:

| Toy | Variant | Learned source rate | Abstention rate | Learned uncertainty mean | Learned advantage mean |
| --- | --- | ---: | ---: | ---: | ---: |
| Toy2 | prototype fallback | 0.8013 | 0.1987 | 0.0316 | -0.0067 |
| Toy2 | zero fallback | 0.8014 | 0.1986 | 0.0316 | -0.0067 |
| Toy4 | prototype fallback | 0.8665 | 0.1335 | 0.0141 | 0.0125 |
| Toy4 | zero fallback | 0.8667 | 0.1333 | 0.0145 | 0.0129 |

Learned/prototype diagnostic relation:

| Toy | Variant | Sign agreement | Non-abstain sign agreement | Correlation |
| --- | --- | ---: | ---: | ---: |
| Toy2 | prototype fallback | 0.8061 | 1.0000 | -0.1945 |
| Toy2 | zero fallback | 0.9865 | 1.0000 | -0.2237 |
| Toy4 | prototype fallback | 0.9782 | 1.0000 | 0.9821 |
| Toy4 | zero fallback | 0.8667 | 1.0000 | 0.9959 |

## Interpretation

This slice achieved the intended structural change: the learned critic now has
candidate-conditioned phase context and can produce a credit signal that is not
just a near-copy of the prototype signal.

The result is still not a performance improvement over prototype escalation.
All learned candidate-context variants matched the prototype diagnostic on final
hits and time-to-ceiling. The important research finding is narrower:

- Toy2 now shows learned/prototype distinctiveness, including negative
  prototype correlation, without policy collapse.
- Toy4 remains highly aligned with the prototype but uses learned credit more
  often than the previous learned-credit replay slice.
- The candidate-context representation is a viable mechanism change, but it has
  not yet become a superior policy mechanism.

The next structural step should focus on replay selection or critic target
definition. The clearest option is to use learned/prototype disagreement as a
selection diagnostic: replay only where learned credit is confident and either
agrees with prototype on direction or exposes high-value disagreement.
