# Toy2/Toy4 Direction-Pressure Basin Credit Findings

Date: 2026-05-15

## Question

Can we stop treating replay weight as the direction signal itself, and instead
separate:

- direction: which action points toward faster basin entry
- pressure: how strongly that direction should be replayed in this phase

This slice keeps the candidate-context learned basin critic as the direction
critic. It adds a new replay scorer target mode:

```text
target_mode: intervention_pressure
target = max(
  scaled_abs(training_effective_advantage),
  scaled_positive(future_basin_score_delta)
)
```

The intent is to avoid the future-motion scorer failure mode where the target is
future-grounded but too weak for Toy4.

## Artifacts

- Pressure scorer:
  `experiments/results/basin_critic/toy24_basin_replay_pressure_scorer_h5_q99_quick_summary.md`
- Evidence manifest:
  `experiments/evidence/toy24_basin_direction_pressure_quick.yaml`
- Evidence gate:
  `experiments/evidence/results/toy24_basin_direction_pressure_quick.summary.md`
- Runs:
  `experiments/results/nabm_effect_matrix/toy24_basin_direction_pressure_quick_runs.csv`

## Offline Scorer

| Toy | Target | Horizon | Eval MSE | Eval Corr | Eval Weight Mean | Eval Target Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Toy2 | intervention_pressure | 5 | 0.000462 | 0.885082 | 0.762534 | 0.763424 |
| Toy4 | intervention_pressure | 5 | 0.010451 | -0.159526 | 0.962346 | 0.992892 |

Toy2 has a meaningful pressure fit. Toy4 does not have a useful ranking
correlation because the pressure target is almost saturated. This is still
useful operationally, but it is not evidence that the pressure scorer has
learned a fine-grained Toy4 curriculum.

## Evidence Result

Gate status: pass.

| Toy | Variant | Final Hits | Mean TtC | Final Payoff | Mean Replay Weight |
| --- | --- | ---: | ---: | ---: | ---: |
| Toy2 | prototype escalation | 3/3 | 9.33 | 3.0 | 1.000 |
| Toy2 | magnitude scorer | 3/3 | 9.33 | 3.0 | 0.749 |
| Toy2 | future-motion scorer | 3/3 | 9.33 | 3.0 | 0.514 |
| Toy2 | direction-pressure scorer | 3/3 | 9.33 | 3.0 | 0.749 |
| Toy4 | prototype escalation | 3/3 | 11.67 | 0.6 | 1.000 |
| Toy4 | magnitude scorer | 3/3 | 11.67 | 0.6 | 0.985 |
| Toy4 | future-motion scorer | 3/3 | 12.00 | 0.6 | 0.514 |
| Toy4 | direction-pressure scorer | 3/3 | 11.67 | 0.6 | 0.983 |

The main behavioral outcome is that the direction-pressure scorer avoids the
future-motion under-pressure failure in Toy4 while preserving Toy2 attenuation.

## Interpretation

This is a better structural framing than choosing between magnitude and
future-motion scorers:

- The learned basin phase critic remains responsible for action direction.
- The pressure scorer decides training strength.
- Future motion is now part of pressure supervision, but not allowed to erase
  the intervention pressure needed by Toy4.

The result should still be stated carefully:

- Toy2 supports the direction-pressure split: lower pressure than all-replay is
  sufficient.
- Toy4 supports the need for high intervention pressure: the successful scorer
  is still near all-replay.
- The Toy4 pressure model is not yet a fine-grained learned curriculum; its
  target is mostly saturated.

## Next Step

The next structural target should make direction supervision explicitly
pairwise:

```text
score(candidate_action=1) - score(candidate_action=0)
  should predict faster basin entry than the opposite action
```

That would move the learned critic closer to counterfactual basin entry instead
of only phase-level target-reaching probability. The pressure scorer can then be
trained to preserve the intervention strength needed for the domain geometry
without carrying the burden of action direction.
