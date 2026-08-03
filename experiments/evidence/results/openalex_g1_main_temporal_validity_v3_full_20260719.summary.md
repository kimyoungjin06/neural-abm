# OpenAlex Main Temporal Field Validity v3 Full Audit

- Status: `PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_FIELD_VALIDITY_V3_WITH_EXACT_UTC_MILLISECONDS_AND_NONBLOCKING_ORDER_PROFILE`
- Scope: 493 files, 492,293,207 raw rows
- GPU used: `false`
- Component pass: `true`

## Exact results

The audit quarantined the one frozen `W-1` row and retained 492,293,206
grammar-valid work rows, including the unresolved `W0` row. All retained
`created_date` and `updated_date` values were non-null, parsed exactly as
`YYYY-MM-DDTHH:MM:SS.sssZ`, and fell before the exclusive
`2026-03-31T00:00:00.000Z` boundary. Metadata parse errors, null rows, and
future field values were all zero.

Publication dates and years were pairwise concordant for all 464,727,674
nonnull pairs. The 27,565,532 both-null rows and every frozen publication-year
bucket matched the prior aggregate regression exactly. All accounting
contracts passed.

## Ordering is descriptive

Exactly 492,293,203 parsed pairs had `created_date <= updated_date`, while
three had `created_date > updated_date`. OpenAlex documents the meanings of the
two fields in its [Work object reference](https://github.com/ourresearch/openalex-docs/blob/main/api-entities/works/work-object/README.md)
but does not define their relationship as a monotone validity invariant; the
[current API reference](https://developers.openalex.org/api-reference/works/get-a-single-work)
contains a reversed example. The three rows therefore remain a
fingerprint-bound descriptive count and do not enter the violation total,
component PASS, or an exception allowlist.

## Evidence integrity

- Report SHA-256: `de98be73fb0365672cec7ee770c40de090ae012d45a0422faebd34c0b9d19200`
- Receipt SHA-256: `2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe`
- Result fingerprint: `a844c624e81567bf89536218b0c488144fe2730f93c53e72193e6cfe4931bfb5`
- Task fingerprint: `5d86a797481a14e95d08ecbe333bb207ff476bf7dc49886aec1db7b5b366001d`
- Checkpoint tree: `725cf22ce96eca153fa1c9542bd09bd628a4ea1687a2fee3e5610a8cded95392`
- Receipt artifacts independently rehashed: `497 / 497`

The full run is bound to the current-content receipt
`082161a29bcd8fd79fd85e0bf6814a5c7950b97b847ea08056e17eebcd369a12`
and its exact 493-file main-source rich-stat bridge.

## Claim boundary

This pass establishes the bounded main-work temporal-field component only. It
does not establish record-event order, citation-edge observation time or
`available_at`, historical content identity, author or candidate validity,
G-1 or G0, empirical G1, field selection, NABM effects, or a scientific
effect. Downstream products require a separate v3-only parent rebind and their
own audited execution.
