# OpenAlex Main Temporal Validity Audit Surface

Status: `PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_FIELD_VALIDITY_V3_WITH_EXACT_UTC_MILLISECONDS_AND_NONBLOCKING_ORDER_PROFILE`

The repository preserves three versioned surfaces:

- v1: `openalex_g1_main_temporal_validity_v0.yaml` and
  `audit_openalex_main_temporal_validity.py`;
- v2: `openalex_g1_main_temporal_validity_v1.yaml` and
  `audit_openalex_main_temporal_validity_v2.py`;
- v3: `openalex_g1_main_temporal_validity_v2.yaml` and
  `audit_openalex_main_temporal_validity_v3.py`.

Runtime checkpoints, plans, reports, receipts, and absolute source paths remain
excluded from packages and releases. Compact, claim-bounded evidence summaries
may be retained under `experiments/evidence/results/`.

## Execution history

The v1 full run scanned all 493 files and 492,293,207 raw rows. Publication
date/year parsing and aggregate regression were exact, but every retained
`created_date` and `updated_date` failed its incorrect date-only parser. The v1
output is therefore a terminal `BLOCKED_MAIN_TEMPORAL_VALIDITY_COMPONENT`; its
zero reversal/future counts are non-evidence because it parsed zero metadata
pairs.

The v2 parser corrected metadata timestamps to exact
`YYYY-MM-DDTHH:MM:SS.sssZ`. Its nonpromotable first-file pilot parsed all
1,000,000 retained pairs with zero parse, null, future, publication, or
accounting failures, but observed three `created_date > updated_date` rows. The
pilot is terminal, byte-preserved, and dirty under the v2 ordering gate. A full
v2 run was not started.

The v3 one-file preflight then parsed all 1,000,000 pairs, retained the same
three reversals as a nonblocking diagnostic, and passed its local contract.
The separately authorized v3 full run scanned all 493 files and 492,293,207
raw rows. After the frozen `W-1` quarantine, all 492,293,206 retained metadata
pairs parsed exactly, were non-null, and were before the snapshot boundary.
All 464,727,674 nonnull publication-date/year pairs matched, every accounting
contract passed, and the frozen aggregate regression was exact. The terminal
status is
`PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_FIELD_VALIDITY_V3_WITH_EXACT_UTC_MILLISECONDS_AND_NONBLOCKING_ORDER_PROFILE`.

The compact evidence is recorded in
`experiments/evidence/results/openalex_g1_main_temporal_validity_v3_full_20260719.summary.json`
and its Markdown companion. The runtime report SHA-256 is
`de98be73fb0365672cec7ee770c40de090ae012d45a0422faebd34c0b9d19200`;
the receipt SHA-256 is
`2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe`.
All 497 receipt artifacts were independently rehashed exactly.

## Why v3 changes the gate

OpenAlex documents `created_date` as the date a Work was created or added in
the OpenAlex dataset and `updated_date` as the last time the Work changed. It
does not document `created_date <= updated_date` as a vendor invariant; the
current API reference even contains a reversed example. See the
[Work object reference](https://github.com/ourresearch/openalex-docs/blob/main/api-entities/works/work-object/README.md#created_date)
and [current API reference](https://developers.openalex.org/api-reference/works/get-a-single-work).

Accordingly, v3 keeps the exact UTC-millisecond parser but separates two
contracts:

- metadata timestamp validity is blocking for nulls, lexical/round-trip parse
  failures, and timestamps at or after the next UTC day following the declared
  snapshot date;
- `created_date <= updated_date` versus `created_date > updated_date` is a
  descriptive, fingerprint-bound profile with no exception allowlist and no
  role in component PASS or pilot cleanliness.

The completed production projection was exactly `id`, `publication_date`,
`publication_year`, `created_date`, and `updated_date`. It required explicit
`--authorize-full`, the complete current-content receipt, resource preflight,
and a private operator-owned `0700` output directory. A max-files run remains
nonpromotable even when its local v3 parser contract is clean.

## Claim boundary

The v3 full pass establishes current-snapshot publication parsing and date/year
concordance, frozen publication buckets, and exact non-null UTC-millisecond
metadata timestamps that do not exceed the snapshot boundary. It does not
establish record-event ordering. Citation-edge `available_at` and edge
chronology remain `PENDING_NOT_IN_THIS_COMPONENT`; neither metadata timestamp
is used as an edge timestamp. Historical identity, maturation windows,
outcomes, author validity, candidate eligibility, G-1, G0, empirical G1, field
selection, NABM effects, and scientific effects remain unestablished.
