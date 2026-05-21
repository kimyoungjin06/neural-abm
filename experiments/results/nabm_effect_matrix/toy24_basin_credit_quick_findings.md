# Toy2/4 Basin-Credit Quick Findings

This report summarizes `toy24_basin_credit_quick` from
`experiments/evidence/toy24_basin_credit_quick.yaml`.

Run scope:

- Seeds: 1, 2, 3.
- Epochs: 50.
- Run rows: `toy24_basin_credit_quick_runs.csv`.
- Grouped effects: `toy24_basin_credit_quick_effects.csv`.
- Pairwise effects: `toy24_basin_credit_quick_pairwise_effects.csv`.
- Gate summary: `experiments/evidence/results/toy24_basin_credit_quick.summary.json`.

## Gate Verdict

Overall gate status: **fail**.

Input validation was clean: no missing runs, malformed rows, duplicate rows, or
unknown rows were reported. The failure is therefore a substantive result for
the current quick-run evidence contract, not an artifact-completeness issue.

## Case Results

| Case | Gate Criterion | Best Main Variant | Main Final Hits | Main Mean TtC | Main Metric Mean | Best Baseline | Baseline Final Hits | Baseline Mean TtC | Baseline Metric Mean |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `toy2_basin_credit` | final hits >= 3, mean TtC < 10 | `basin_credit_w1p0_h1_prototype` | 0/3 | n/a | 2.5567 | `reputation_imitation` | 3/3 | 2.6667 | 3.0000 |
| `toy4_basin_credit` | final hits >= 2, mean TtC < 12 | `basin_credit_w1p0_h1_prototype` | 0/3 | n/a | 0.3140 | `reputation_imitation` | 3/3 | 2.6667 | 0.6000 |

The best eligible main variant in both cases was
`basin_credit_w1p0_h1_prototype`, but it produced zero final ceiling hits.
`mixed_individual_basin_w0p5_0p5_h1` also produced zero final ceiling hits in
both cases, with lower mean final payoff than the pure basin-credit prototype.

Grouped effects were negative for the NABM group:

- Toy2: NABM mean `1.8083` versus baseline mean `3.0000`, effect `-1.1917`.
- Toy4: NABM mean `0.1590` versus baseline mean `0.5980`, effect `-0.4390`.

Pairwise comparisons also favored the baselines for every baseline/main pair.
Against `reputation_imitation`, the best main variant was lower by `-0.4433` in
Toy2 and `-0.2860` in Toy4.

## Diagnostic Variants

Teacher/bootstrap/replay paths remain excluded from the main claim by the gate.
`decision_bootstrap_w1p0_e5_linear_welfare` is therefore diagnostic-only even
when it reaches the ceiling.

Diagnostic outcome:

- Toy2 `decision_bootstrap_w1p0_e5_linear_welfare`: 3/3 final ceiling hits,
  mean TtC `22.6667`, final payoff mean `3.0000`.
- Toy4 `decision_bootstrap_w1p0_e5_linear_welfare`: 1/3 final ceiling hits,
  mean TtC `19.6667`, final payoff mean `0.5960`.

This keeps the evidence interpretation narrow: the scaffolded/bootstrap path
can still be useful for diagnosis, but it does not rescue the basin-credit main
claim under the hardened gate.

## Interpretation

The current `prototype_phase` basin-credit scaffold does not show a positive
Toy2/Toy4 signal in this quick evidence run. The result is especially clear
because the baselines hit or nearly hit the ceiling while both eligible
basin-credit variants miss the ceiling entirely.

This should not be treated as a reason to move directly into a contrastive
critic implementation. The immediate question is why the current basin-credit
objective is suppressing ceiling convergence relative to the simpler welfare
and reputation paths.

## Branch Rule

- If both cases pass: proceed to a shared `PostSocialBasinState` abstraction or
  contrastive critic preparation.
- If any case is inconclusive: rerun the missing or malformed seeds first.
- If any case fails with clean inputs: inspect aggregate trajectories and
  basin-credit diagnostics before adding a learned contrastive critic.

Current branch: **fail with clean inputs**. The next action is trajectory-level
diagnosis of the basin-credit variants, not another threshold/source sweep and
not a contrastive critic implementation yet.
