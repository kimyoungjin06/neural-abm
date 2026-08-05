# Pivot-feedback current-status and evidence ledger v16

Date: 2026-08-05

Status:
`CURRENT_HUMAN_FACING_INDEX_E_G1_INPUT_COMPLETENESS_VERIFIED_EXECUTION_NOT_RUN`

## The v1 preflight was necessary but incomplete

The v1 E/G1 preflight correctly froze the registered 2013-2016 scope, fields
16, 32, and 11, 28 episodes, 13 families, the primary normalized work and author
history sources, scientific rules, and seven output schemas. Before any
confirmatory row was read, implementation review found that it did not bind
three sources already required by those frozen rules: citation chronology, the
full topic-to-subfield taxonomy, and raw authorship ORCID/affiliation data.

That finding invalidated v1 as a complete execution authority but did not
invalidate its provenance or alter the registered science. Preflight v1 remains
immutable; v1 alone is superseded for execution.

## One boundary, corrected additively

The independently verified
[`preflight v2 correction`](../../experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v2_20260805.summary.md)
adds exactly the three missing input roles and one aggregate sensitivity output
schema. It is not a new scientific gate. The single E entry boundary is the
conjunction of parent v1 plus independently verified v2.

| Added input | Exact binding |
| --- | --- |
| Citation chronology | 4,096 artifacts; 22,721,388,007 bytes |
| Topic taxonomy | 4,516 rows; exact file SHA-256 |
| Raw authorship | 1,363 files; 103,412,497,044 bytes; content-freeze lineage |

The effective contract now has seven input groups and eight output schemas.
ORCID-linked estimates are a mandatory sensitivity rather than a replacement
for the primary analysis. Affiliation is explicitly a current-snapshot proxy
with missingness retained, not a historical identity claim.

## Scope and exclusion remain unchanged

The authorized G1 execution remains restricted to entry years 2013-2016,
fields 16, 32, and 11 in registered slot order, 28 certified episodes, and 13
atomic families. Field or episode replacement, threshold revision, control
revision, and estimand revision are forbidden. G4 field/year data, the private
G4 root or manifest, model-selection scores, and seal opening remain outside E.

## Current boundary and next action

No confirmatory read occurred while diagnosing or correcting the input
contract: confirmatory rows read remain zero, as do holdout rows. No label,
negative-control value, estimate, or decision has been materialized. The active
state therefore remains `empirical_g1_status=NOT_RUN`, `pass_e=false`,
`pass_g1=false`, `g4_open=false`, and `scientific_effect=NOT_ESTABLISHED`.

The next artifact is now actual E/G1 execution under parent v1 plus
independently verified v2. That successor must apply the G1 partition and field
allowlist before label logic and accept PASS, registered-decision FAIL, or
input/support/estimator BLOCKED without post-access redesign.
