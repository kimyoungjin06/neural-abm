# Toy2/Toy4 Learned Basin-Credit Replay Findings

## Scope

Manifest:

```text
experiments/evidence/toy24_basin_learned_credit_replay_quick.yaml
```

This run tests the first policy-facing learned basin-credit source. The learned
phase critic is not read-only here: when the abstention and uncertainty gates
allow it, its action-1 advantage replaces the prototype action-1 advantage used
inside the basin replay objective. Abstaining samples fall back either to the
prototype advantage or to zero, depending on the variant.

## Gate Result

The evidence gate passed.

```text
experiments/evidence/results/toy24_basin_learned_credit_replay_quick.summary.json
experiments/evidence/results/toy24_basin_learned_credit_replay_quick.summary.md
```

Main learned-gated prototype fallback results:

| Toy | Final ceiling hits | Mean time-to-ceiling | Baseline mean TtC |
| --- | ---: | ---: | ---: |
| Toy2 | 3/3 | 9.33 | 23.67 |
| Toy4 | 3/3 | 11.67 | 21.00 |

## Variant Comparison

| Toy | Variant | Group | Final hits | Mean TtC | Final metric |
| --- | --- | --- | ---: | ---: | ---: |
| Toy2 | linear_welfare_heavy | baseline | 3/3 | 23.67 | 3.000 |
| Toy2 | prototype escalation | diagnostic | 3/3 | 9.33 | 3.000 |
| Toy2 | learned-gated prototype fallback | nabm | 3/3 | 9.33 | 3.000 |
| Toy2 | learned-gated zero fallback | diagnostic | 3/3 | 9.33 | 3.000 |
| Toy4 | linear_welfare_heavy | baseline | 1/3 | 21.00 | 0.596 |
| Toy4 | prototype escalation | diagnostic | 3/3 | 11.67 | 0.600 |
| Toy4 | learned-gated prototype fallback | nabm | 3/3 | 11.67 | 0.600 |
| Toy4 | learned-gated zero fallback | diagnostic | 3/3 | 11.67 | 0.600 |

## Learned Source Usage

Average over all aggregate rows:

| Toy | Variant | Learned source rate | Abstention rate | Learned uncertainty mean |
| --- | --- | ---: | ---: | ---: |
| Toy2 | learned-gated prototype fallback | 0.8192 | 0.1808 | 0.0271 |
| Toy2 | learned-gated zero fallback | 0.8192 | 0.1808 | 0.0270 |
| Toy4 | learned-gated prototype fallback | 0.7538 | 0.2462 | 0.0197 |
| Toy4 | learned-gated zero fallback | 0.7472 | 0.2528 | 0.0200 |

At the final epoch, both learned-gated variants used learned credit for all
agents in Toy2 and Toy4.

## Interpretation

This is a structural improvement over the previous read-only diagnostic slice:
the learned phase critic now participates in the replay objective, and the CSV
contract records which source actually supplied the training credit.

The evidence does not yet show a performance gain over the prototype escalation
control. The learned-gated variants matched the prototype diagnostic on final
hits and mean time-to-ceiling. The conservative conclusion is:

- learned basin-credit replay is wired end-to-end and does not collapse on
  Toy2/Toy4 quick evidence;
- the uncertainty/abstention gate is active and inspectable;
- current learned critic quality is sufficient to replace prototype credit in
  these runs, but not yet sufficient to claim a better policy mechanism.

The next meaningful improvement should target critic distinctiveness or replay
selection, not more threshold-only sweeps.
