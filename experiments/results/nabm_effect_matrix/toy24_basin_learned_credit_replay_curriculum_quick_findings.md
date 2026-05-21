# Toy2/Toy4 Learned Basin-Credit Replay Curriculum Findings

## Scope

This slice replaces the static replay floor with a temporal curriculum. The
selector still uses learned/prototype confident agreement, but the minimum
selected rate starts broad and linearly decays toward the target floor:

```text
learned_credit_replay_floor_schedule: linear_decay
learned_credit_replay_floor_start_rate: 1.0
learned_credit_replay_min_selected_rate: 0.50
```

The main candidate uses `decay_epochs=30`. A `decay_epochs=20` variant is kept
as a diagnostic because it passed Toy2 but missed Toy4 speed by a narrow margin.

Manifest:

```text
experiments/evidence/toy24_basin_learned_credit_replay_curriculum_quick.yaml
```

Diagnostic summary:

```text
experiments/results/basin_critic/toy24_basin_learned_credit_replay_curriculum_quick_learned_diagnostic_summary.md
```

## Evidence Gate Result

The evidence gate passed.

```text
experiments/evidence/results/toy24_basin_learned_credit_replay_curriculum_quick.summary.json
experiments/evidence/results/toy24_basin_learned_credit_replay_curriculum_quick.summary.md
```

Main curriculum `floor50_d30` results:

| Toy | Final ceiling hits | Mean TtC | Gate threshold | Replay floor | Replay selected | Training learned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | 3/3 | 9.33 | < 10 | 0.6500 | 0.6528 | 0.7428 |
| Toy4 | 3/3 | 11.67 | < 12 | 0.6500 | 0.9929 | 0.8665 |

## Variant Comparison

| Toy | Variant | Group | Final hits | Mean TtC | Final metric | Replay selected |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Toy2 | linear_welfare_heavy | baseline | 3/3 | 23.67 | 3.000 |  |
| Toy2 | prototype escalation | diagnostic | 3/3 | 9.33 | 3.000 | 1.0000 |
| Toy2 | learned candidate-context all replay | diagnostic | 3/3 | 9.33 | 3.000 | 1.0000 |
| Toy2 | learned confident agreement | diagnostic | 0/3 |  | 2.328 | 0.0000 |
| Toy2 | learned confident agreement floor50 | diagnostic | 3/3 | 16.00 | 3.000 | 0.5000 |
| Toy2 | learned confident agreement curriculum floor50 d20 | diagnostic | 3/3 | 9.33 | 3.000 | 0.6018 |
| Toy2 | learned confident agreement curriculum floor50 d30 | nabm | 3/3 | 9.33 | 3.000 | 0.6528 |
| Toy4 | linear_welfare_heavy | baseline | 1/3 | 21.00 | 0.596 |  |
| Toy4 | prototype escalation | diagnostic | 3/3 | 11.67 | 0.600 | 1.0000 |
| Toy4 | learned candidate-context all replay | diagnostic | 3/3 | 11.67 | 0.600 | 1.0000 |
| Toy4 | learned confident agreement | diagnostic | 0/3 |  | 0.298 | 0.0039 |
| Toy4 | learned confident agreement floor50 | diagnostic | 3/3 | 14.67 | 0.600 | 0.8912 |
| Toy4 | learned confident agreement curriculum floor50 d20 | diagnostic | 3/3 | 12.33 | 0.600 | 0.9893 |
| Toy4 | learned confident agreement curriculum floor50 d30 | nabm | 3/3 | 11.67 | 0.600 | 0.9929 |

## Interpretation

The curriculum result validates the structural diagnosis:

- Hard `confident_agreement` replay fails by starvation.
- A static floor restores final ceilings but is slower.
- A broad-to-narrow curriculum restores the quick-gate target on both toys.

The claim must stay narrow. The curriculum main candidate matched the
prototype/all-replay time-to-ceiling; it did not outperform them. Toy4 also
selected almost every active replay candidate on average (`0.9929`), so this is
not yet strong evidence that selective learned replay can replace all replay.

The defensible conclusion is:

> A learned/prototype relation selector needs a temporal broad-start curriculum
> to avoid early replay starvation. With a slow decay, it can match the existing
> prototype/all-replay quick evidence while preserving the learned critic as the
> credit source.

The next structural step should make the narrowing criterion state-dependent,
not just epoch-dependent. A better selector should reduce replay only after
basin motion, critic confidence, or agreement coverage crosses a logged
threshold.
