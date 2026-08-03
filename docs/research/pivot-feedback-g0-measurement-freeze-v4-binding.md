# Pivot-feedback G0 measurement-freeze v4 binding successor

Date: 2026-07-31

Status:
`PASS_G0_V4_EXACT_SIX_CHOICE_BINDING_SEVEN_CHOICES_REMAIN`

## Result

G0 v4 implements exactly the six-choice binding authorized by gate v60. It
binds G0 v3, the risk-set/weight/support precommitment, and the authorizing gate
by byte hash. Ten G0 v3 top-level subtrees remain identical, including the
earlier four-choice measurement/proxy binding.

No production discovery rule application was run. The successor is a
versioned metadata contract, not an analytic sample, fitted model, or effect
result.

## Newly bound choices

Six additional choices are serialized:

1. `negative_sampling_ratio`: retain all cases and sample at most ten controls
   per case without replacement within the exact risk set using deterministic
   SHA-256 priority;
2. `inverse_probability_weight_truncation`: nearest-rank P01/P99
   winsorization of positive finite stabilized weights followed by mean-one
   normalization, with mandatory raw-weight sensitivity;
3. `minimum_cohort_size`: at least 500 full-risk-set opportunities, 100 entry
   events, and 100 non-entry opportunities per partition, field, and episode;
4. `small_cell_suppression_threshold`: suppress counts 1 through 10 and
   derivable complements, with 11 as the minimum directly reportable nonzero
   count;
5. `capacity_cell_support_threshold`: at least 50 opportunities, 11 events,
   and 11 non-events in every one of the four signal-by-capacity cells; and
6. `collinearity_threshold`: every centered weighted VIF for signal, capacity,
   and their interaction must be strictly below 5.

The full rationale, literature boundary, formulae, failure actions, and closed
fixture remain in
`docs/research/pivot-feedback-g0-risk-set-weight-support-precommitment-v1.md`.

## Preservation and failure action

Independent replay verifies the exact G0 v3 predecessor hash, exact v60 gate
hash, exact precommitment hash, and all ten unauthorized-to-change top-level
subtrees. Mutation tests reject both preserved-subtree drift and filling a
seventh new quantitative choice.

Capacity support and VIF failure share one registered action: collapse the
three-column channel block to an any-signal-or-capacity indicator and forfeit
the separate information-versus-capacity claim. No post-hoc selective variable
deletion is authorized.

Twelve choices are now resolved and seven remain null. All fifteen required
sections remain not freeze-ready. The holdout receipt remains absent,
`pass_g0` remains false, and the contract remains unsealable.

The verifier reads four repository YAML documents. It opens no operator JSON,
Parquet, row-level, confirmatory, or holdout data and fits no model. G4 remains
logically reserved, not byte-sealed, and unopened.
