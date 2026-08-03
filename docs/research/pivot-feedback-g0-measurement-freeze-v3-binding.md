# Pivot-feedback G0 measurement-freeze v3 binding successor

Date: 2026-07-31

Status:
`PASS_G0_V3_EXACT_FOUR_CHOICE_BINDING_THIRTEEN_CHOICES_REMAIN`

## Result

G0 v3 implements exactly the four-choice binding authorized by gate v58. It
binds G0 v2, the measurement/proxy precommitment, and the authorizing gate by
byte hash. Nine G0 v2 top-level subtrees remain identical.

No production discovery rule application was run. The successor is a
versioned metadata contract, not a production classification or effect result.

## Newly bound choices

The two inherited D0 choices remain unchanged. Four additional choices are
serialized:

1. `attention_surge_threshold`: incumbent-only recent-versus-reference
   log-attention acceleration, standardized within field and decision year by
   median/MAD, with an inclusive modified z-score threshold of 3.0 and a
   fail-closed MAD-zero rule;
2. `public_success_threshold`: field-by-publication-year age-3 citation
   top-decile target, inclusive cutoff ties, explicit realized-fraction
   reporting, and no classification for cohorts smaller than ten;
3. `citation_maturation_lag_years`: three complete calendar years after
   publication year; and
4. `negative_control_equivalence_margins`: average marginal binary-outcome
   probability-difference practical null of `[-0.01, 0.01]`, evaluated by TOST
   at one-sided alpha 0.05, equivalently a two-sided 90% confidence interval.

These values are outcome-blind policy choices rather than literature-mandated
truths. The rationale, references, exact formulae, and closed-fixture edge
cases remain in
`docs/research/pivot-feedback-g0-measurement-proxy-precommitment-v1.md`.

## Preservation and remaining boundary

Independent replay verifies the exact G0 v2 predecessor hash, exact v58 gate
hash, exact precommitment hash, and all nine unauthorized-to-change top-level
subtrees. Mutation tests reject both preserved-subtree drift and filling a
fifth quantitative choice.

Six choices are now resolved and thirteen remain null. All fifteen required
sections remain not freeze-ready. The holdout receipt remains absent,
`pass_g0` remains false, and the contract remains unsealable.

Citation availability is still a current-snapshot bibliographic-time proxy. It
does not recover historical platform observation time or establish researcher
visibility or information receipt.

The verifier reads four repository YAML documents. It opens no operator JSON,
Parquet, row-level, confirmatory, or holdout data and fits no model. G4 remains
logically reserved, not byte-sealed, and unopened.
