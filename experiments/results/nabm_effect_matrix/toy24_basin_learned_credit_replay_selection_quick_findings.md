# Toy2/Toy4 Learned Basin-Credit Replay Selection Findings

## Scope

This slice changes replay candidate selection, not the critic representation or
a scalar replay threshold. Learned basin credit can now choose which active
basin-training agents enter replay using:

- `all`
- `confident`
- `confident_agreement`
- `confident_disagreement`

The tested manifest uses the candidate-context learned critic from the previous
slice and compares all-replay against learned/prototype agreement and
disagreement selection.

Manifest:

```text
experiments/evidence/toy24_basin_learned_credit_replay_selection_quick.yaml
```

Diagnostic summary:

```text
experiments/results/basin_critic/toy24_basin_learned_credit_replay_selection_quick_learned_diagnostic_summary.md
```

## Evidence Gate Result

The evidence gate failed.

```text
experiments/evidence/results/toy24_basin_learned_credit_replay_selection_quick.summary.json
experiments/evidence/results/toy24_basin_learned_credit_replay_selection_quick.summary.md
```

Main confident-agreement replay results:

| Toy | Final ceiling hits | Mean final metric | Replay selected | Training learned |
| --- | ---: | ---: | ---: | ---: |
| Toy2 | 0/3 | 2.328 | 0.0000 | 0.4745 |
| Toy4 | 0/3 | 0.298 | 0.0039 | 0.0039 |

The failure mode is replay starvation. The confident-agreement policy is too
restrictive early in training, so it removes nearly all candidates from replay.
Toy2 selects no replay agents on average, and Toy4 selects only about `0.39%`.

## Variant Comparison

| Toy | Variant | Group | Final hits | Mean TtC | Final metric | Replay selected |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Toy2 | linear_welfare_heavy | baseline | 3/3 | 23.67 | 3.000 |  |
| Toy2 | prototype escalation | diagnostic | 3/3 | 9.33 | 3.000 | 1.0000 |
| Toy2 | learned candidate-context all replay | diagnostic | 3/3 | 9.33 | 3.000 | 1.0000 |
| Toy2 | learned confident agreement replay | nabm | 0/3 |  | 2.328 | 0.0000 |
| Toy2 | learned confident disagreement replay | diagnostic | 3/3 | 24.67 | 3.000 | 0.5761 |
| Toy4 | linear_welfare_heavy | baseline | 1/3 | 21.00 | 0.596 |  |
| Toy4 | prototype escalation | diagnostic | 3/3 | 11.67 | 0.600 | 1.0000 |
| Toy4 | learned candidate-context all replay | diagnostic | 3/3 | 11.67 | 0.600 | 1.0000 |
| Toy4 | learned confident agreement replay | nabm | 0/3 |  | 0.298 | 0.0039 |
| Toy4 | learned confident disagreement replay | diagnostic | 0/3 |  | 0.276 | 0.0000 |

## Learned Diagnostic Relation

| Toy | Variant | Sign agree | Non-abstain agree | Abstain | Uncertainty |
| --- | --- | ---: | ---: | ---: | ---: |
| Toy2 | all replay | 0.8061 | 1.0000 | 0.1987 | 0.0316 |
| Toy2 | confident disagreement | 1.0000 | 1.0000 | 0.0000 | 0.0292 |
| Toy4 | all replay | 0.9782 | 1.0000 | 0.1335 | 0.0141 |
| Toy4 | confident agreement | 1.0000 | 1.0000 | 0.0000 | 0.0061 |

The all-replay candidate-context variant still matches prototype escalation on
both toys. The negative result is specific to selection: filtering replay down
to confident agreement, before the policy has reached useful basin states,
removes the training pressure that made the escalation slice work.

## Interpretation

This is a structural replay-policy experiment, not a threshold sweep: the
candidate set used for replay is changed by learned/prototype relation and
critic confidence. The result is negative for the tested main candidate.

The useful conclusion is narrow:

- Learned credit remains viable when all active basin candidates are replayed.
- Agreement-only replay is not viable as an early-training policy because it
  self-starves.
- Disagreement-only replay is not a general replacement: it recovers Toy2 only
  slowly and collapses Toy4.
- The next mechanism should add a curriculum or fallback replay floor before
  using learned/prototype relation as a selector, or change the critic target so
  confident selected sets are non-empty in early training.

Do not promote `confident_agreement` as the next main variant without a
warm-start, replay-floor, or target-definition change.
