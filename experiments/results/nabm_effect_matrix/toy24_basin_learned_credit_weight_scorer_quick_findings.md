# Toy2/Toy4 Learned Basin-Credit Replay Weight Scorer Findings

## Scope

This slice adds a frozen learned replay-weight scorer:

```text
learned_credit_replay_mode: learned_weight
learned_credit_replay_weight_model_path: <scorer>.npz
```

The scorer is trained offline from `basin_transition_samples.parquet` rows and
the candidate-context learned basin critic. At runtime it predicts a continuous
per-agent replay loss weight from critic margin, uncertainty, learned/prototype
agreement, candidate scores, and phase-state features.

Training artifact:

```text
experiments/results/basin_critic/toy24_basin_replay_weight_scorer_q99_quick_summary.md
```

Manifest:

```text
experiments/evidence/toy24_basin_learned_credit_weight_scorer_quick.yaml
```

Diagnostic summary:

```text
experiments/results/basin_critic/toy24_basin_learned_credit_weight_scorer_quick_learned_diagnostic_summary.md
```

## Evidence Gate Result

The evidence gate passed.

```text
experiments/evidence/results/toy24_basin_learned_credit_weight_scorer_quick.summary.json
experiments/evidence/results/toy24_basin_learned_credit_weight_scorer_quick.summary.md
```

Main `weight_scorer` results:

| Toy | Final ceiling hits | Mean TtC | Gate threshold | Replay selected | Replay weight | Training learned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | 3/3 | 9.33 | < 10 | 1.0000 | 0.7636 | 0.7721 |
| Toy4 | 3/3 | 11.67 | < 12 | 1.0000 | 0.9745 | 0.8666 |

Offline scorer quality:

| Toy | Train N | Eval N | Eval MSE | Eval corr | Eval weight | Eval target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | 10000 | 5000 | 0.000584 | 0.8436 | 0.7629 | 0.7627 |
| Toy4 | 10000 | 5000 | 0.009975 | 0.9395 | 0.9645 | 0.9924 |

## Variant Comparison

| Toy | Variant | Group | Final hits | Mean TtC | Final metric | Replay selected | Replay weight |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Toy2 | linear_welfare_heavy | baseline | 3/3 | 23.67 | 3.000 |  |  |
| Toy2 | prototype escalation | diagnostic | 3/3 | 9.33 | 3.000 | 1.0000 | 1.0000 |
| Toy2 | learned candidate-context all replay | diagnostic | 3/3 | 9.33 | 3.000 | 1.0000 | 1.0000 |
| Toy2 | learned soft min50 | diagnostic | 3/3 | 9.33 | 3.000 | 1.0000 | 0.5416 |
| Toy2 | learned replay weight scorer | nabm | 3/3 | 9.33 | 3.000 | 1.0000 | 0.7636 |
| Toy4 | linear_welfare_heavy | baseline | 1/3 | 21.00 | 0.596 |  |  |
| Toy4 | prototype escalation | diagnostic | 3/3 | 11.67 | 0.600 | 1.0000 | 1.0000 |
| Toy4 | learned candidate-context all replay | diagnostic | 3/3 | 11.67 | 0.600 | 1.0000 | 1.0000 |
| Toy4 | learned soft min50 | diagnostic | 3/3 | 11.67 | 0.600 | 1.0000 | 0.8006 |
| Toy4 | learned replay weight scorer | nabm | 3/3 | 11.67 | 0.600 | 1.0000 | 0.9745 |

## Interpretation

This is a real structural step relative to fixed soft replay:

- Replay weight is now a frozen learned module loaded at runtime.
- The fixed `soft_attention` formula is no longer the main replay mechanism in
  this manifest.
- The scorer uses phase-state and critic diagnostics as input, not just a
  hand-coded selected-rate floor.

The claim still needs to remain narrow:

- The scorer target is derived from offline prototype/all-replay transition
  samples. It is not yet trained end-to-end on policy improvement.
- Toy4 learned a replay weight very close to all-replay (`0.9745`), so this
  does not prove that sparse or strongly attenuated replay is sufficient.
- Toy2 shows more meaningful attenuation (`0.7636`) while matching the gate.

The defensible conclusion is:

> A frozen learned replay-weight scorer can replace the fixed soft replay policy
> without breaking the Toy2/Toy4 quick gate. The current target still teaches a
> conservative near-all-replay policy in Toy4, so the next structural step is to
> change the scorer target from magnitude imitation to counterfactual policy
> improvement or held-out basin-motion prediction.
