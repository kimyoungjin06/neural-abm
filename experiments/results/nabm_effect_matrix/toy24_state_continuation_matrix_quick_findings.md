# Toy2/4 State-Continuation Matrix Findings

## Verdict

The state-continuation objective was implemented and produced the intended
directional mechanism, but the pre-registered strict "beat reputation_imitation"
criterion was not satisfied.

The strict criterion is not attainable on `final_mean_payoff` for the current
Toy2/4 reference settings because `reputation_imitation` already reaches the
domain payoff ceiling:

- Toy2 all-cooperation payoff ceiling: `3.0`
- Toy4 all-contribution payoff ceiling: `0.6`

## Observed Results

- Toy2 `neural_material_output_average`: mean payoff `1.06`
- Toy2 `neural_continuation_welfare_heavy`: mean payoff `3.0`
- Toy2 `reputation_imitation`: mean payoff `3.0`

- Toy4 `neural_material_output_average`: mean payoff `0.008`
- Toy4 `neural_continuation_balanced`: mean payoff `0.596`
- Toy4 `neural_continuation_welfare_heavy`: mean payoff `0.596`
- Toy4 `reputation_imitation`: mean payoff `0.6`

## Diagnostic Interpretation

Toy2 reached the same ceiling as the reputation baseline under the welfare-heavy
continuation objective. Toy4 moved from near-zero material neural payoff to
near-ceiling payoff, and diagnostics showed positive effective contribution
advantage for the successful continuation variants.

This supports the mechanism direction, but not a strict superiority claim over
the current deterministic reputation baseline. A defensible next acceptance
criterion should distinguish ceiling-tie cases from true failure, or compare
against a harder non-ceiling reference regime.
