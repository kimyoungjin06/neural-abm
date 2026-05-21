# Toy2/Toy4 Learned Basin-Credit Replay Floor Findings

## Scope

This slice tests a starvation-safe replay selector. The previous
`confident_agreement` selector failed because it selected almost no replay
candidates. This implementation keeps the learned/prototype agreement selector
but adds a minimum selected-rate floor:

```text
learned_credit_replay_min_selected_rate
learned_credit_replay_floor_source: prototype_abs
```

When the selected set is below the floor, replay is filled from active
basin-training candidates with the largest absolute prototype basin advantage.

Manifest:

```text
experiments/evidence/toy24_basin_learned_credit_replay_floor_quick.yaml
```

Diagnostic summary:

```text
experiments/results/basin_critic/toy24_basin_learned_credit_replay_floor_quick_learned_diagnostic_summary.md
```

## Evidence Gate Result

The evidence gate failed.

```text
experiments/evidence/results/toy24_basin_learned_credit_replay_floor_quick.summary.json
experiments/evidence/results/toy24_basin_learned_credit_replay_floor_quick.summary.md
```

Main `floor50` results:

| Toy | Final ceiling hits | Mean TtC | Gate threshold | Replay selected | Training learned |
| --- | ---: | ---: | ---: | ---: | ---: |
| Toy2 | 3/3 | 16.00 | < 10 | 0.5000 | 0.6321 |
| Toy4 | 3/3 | 14.67 | < 12 | 0.8912 | 0.7873 |

The floor solved the collapse-to-no-replay failure: both toys recovered `3/3`
final ceiling hits. It did not recover the prototype/all-replay speed.

## Variant Comparison

| Toy | Variant | Group | Final hits | Mean TtC | Final metric | Replay selected |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Toy2 | linear_welfare_heavy | baseline | 3/3 | 23.67 | 3.000 |  |
| Toy2 | prototype escalation | diagnostic | 3/3 | 9.33 | 3.000 | 1.0000 |
| Toy2 | learned candidate-context all replay | diagnostic | 3/3 | 9.33 | 3.000 | 1.0000 |
| Toy2 | learned confident agreement | diagnostic | 0/3 |  | 2.328 | 0.0000 |
| Toy2 | learned confident agreement floor25 | diagnostic | 3/3 | 28.67 | 3.000 | 0.2500 |
| Toy2 | learned confident agreement floor50 | nabm | 3/3 | 16.00 | 3.000 | 0.5000 |
| Toy4 | linear_welfare_heavy | baseline | 1/3 | 21.00 | 0.596 |  |
| Toy4 | prototype escalation | diagnostic | 3/3 | 11.67 | 0.600 | 1.0000 |
| Toy4 | learned candidate-context all replay | diagnostic | 3/3 | 11.67 | 0.600 | 1.0000 |
| Toy4 | learned confident agreement | diagnostic | 0/3 |  | 0.298 | 0.0039 |
| Toy4 | learned confident agreement floor25 | diagnostic | 3/3 | 19.33 | 0.600 | 0.7739 |
| Toy4 | learned confident agreement floor50 | nabm | 3/3 | 14.67 | 0.600 | 0.8912 |

## Interpretation

This slice confirms the failure diagnosis from the replay-selection experiment:
the hard agreement selector was failing by starvation, not because learned
credit necessarily points in the wrong direction. Adding a replay floor restores
final ceiling outcomes.

The negative part is equally important:

- `floor25` is too sparse for speed in both toys.
- `floor50` reaches final ceilings but remains slower than prototype/all replay.
- Toy4 selected far above the nominal floor because agreement becomes common,
  yet it still lagged prototype/all replay.

So a static floor is not enough. The next structural mechanism should avoid a
fixed selected-rate floor and instead use either:

- a replay curriculum that starts broad and narrows only after basin motion is
  established, or
- a critic target that produces useful confident selections earlier.

Do not claim this as a successful replacement for all-replay learned credit.
Claim it only as evidence that replay starvation is removable and that the next
selection mechanism needs temporal/curriculum structure.
