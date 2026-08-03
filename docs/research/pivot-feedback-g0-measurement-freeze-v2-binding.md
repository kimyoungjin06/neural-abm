# Pivot-feedback G0 measurement-freeze v2 binding successor

Date: 2026-07-31

Status:
`PASS_G0_V2_BINDING_ONLY_G_MINUS_1A_SATISFIED_THREE_BLOCKERS_REMAIN_NO_SEAL`

## Result

G0 v2 implements exactly the authorized G-1A binding change.

It preserves the immutable G0 v1 foundation parents, holdout binding, fixed
design choices, resolution, inherited digests, completion contract, failure
policy, and authorization subtrees. The only semantic changes are:

1. retaining the old G-1 binding as pending immutable lineage; and
2. adding the exact G-1A bounded-evidence binding.

## Binding semantics

The legacy binding remains
`PENDING_NOT_PASSED_IMMUTABLE_LINEAGE`. It is not an active prerequisite and
no legacy pass is inferred.

The new `g_minus_1a_binding` points to the exact existing receipt, config, and
independent verifier hashes. It is marked
`SATISFIED_G_MINUS_1A_BOUNDED_EVIDENCE_ONLY`. All 14 unresolved receipt
boundaries and the exact receipt claim boundary remain in force.

This binding establishes bounded current-snapshot primitive and source
fidelity only. It does not establish natural-person identity, historical
OpenAlex visibility, final eligibility, an empirical effect, or a causal
effect.

## Remaining G0 blockers

Only `G_MINUS_1_RECEIPT_ABSENT` is removed. Three blockers remain:

- `HOLDOUT_ACCESS_SEAL_RECEIPT_ABSENT`;
- `SEVENTEEN_QUANTITATIVE_CHOICES_UNRESOLVED`; and
- `FIFTEEN_REQUIRED_SECTIONS_NOT_FREEZE_READY`.

Consequently `pass_g0` remains false and the contract remains unsealable.

Independent validation read seven repository documents. It opened no operator
JSON, Parquet, row-level, confirmatory, or holdout data and made no API
request. Model fitting and G4 remain unauthorized.
