# OpenAlex G-1 Grammar-Valid Main-ID Recount

## Status and gate boundary

The immutable raw source remains `BLOCKED` on ID validity because it contains
one malformed `W-1` row. Under a separate, frozen analysis-view amendment, the
full recount returned
`PASS_GRAMMAR_VALID_MAIN_ID_RECOUNT_WITH_FROZEN_QUARANTINE`.

This component result means:

- body recount: pass;
- exact `W-1` quarantine contract: pass;
- grammar-valid main-ID distinct/duplicate recount: pass;
- raw source-key integrity: not clean;
- `W0` entity validity: unresolved;
- inventory-era byte identity: unverified until content SHA-256;
- G-1 data readiness: pending; and
- empirical G1: not run, so `pass_g1=false`.

## Exact counts

| Measure | Value |
| --- | ---: |
| Files scanned | 493 |
| Body rows | 492,293,207 |
| Null rows | 0 |
| Frozen malformed rows quarantined | 1 |
| Unexpected malformed rows | 0 |
| Grammar-valid retained rows | 492,293,206 |
| Exact distinct grammar-valid IDs | 492,293,206 |
| Duplicate row excess / groups | 0 / 0 |
| Maximum multiplicity | 1 |
| Minimum / maximum numeric key | 0 / 7,143,276,261 |
| Retained `W0` rows | 1 |

`W-1` was reproduced at `part-384.parquet`, row group 13, file row
428,949 and excluded before writing the compact equality spool. `W0` was
reproduced at `part-322.parquet`, row group 10, file row 340,597 and retained
as the real numeric key 0. No numeric sentinel was used.

## Independent verification

The completed run was reopened with `--require-pass`; all three artifact
checksums were recomputed before the same status was returned. Independently,
hashing the earlier body spool while skipping exactly the `W-1` row produced
`7a15e19dc9fc7696e7d6093879f05a13e272e46fb4f9a1395212dad9ca740534`,
identical to the new compact spool. A separate chunked scan of the sorted spool
confirmed 492,293,206 monotone rows, zero adjacent duplicates, and exactly one
numeric zero.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Grammar-valid unsorted IDs | 3,938,345,648 | `7a15e19dc9fc7696e7d6093879f05a13e272e46fb4f9a1395212dad9ca740534` |
| Grammar-valid sorted IDs | 3,938,345,648 | `889d4a54ffb41fbed3b8a657ca76a7887610275dd4b407b68a9233aaf2a08e55` |
| Quarantine observation | 1,679 | `1a8a978ae8473b5c7ccd8239394b1a297cd2a50f74166f6152f95534f995c6af` |

The final terminology-separated run took 122.72 seconds, including 89.24
seconds of file materialization and 21.63 seconds for compact sorting, exact
counting, quarantine output, and checksums. Peak RSS was 8,266,716 KiB.

## Relation incidence and remaining work

The companion targeted audit found 21 `W-1` source rows across six tables and
no `W-1` target in the declared referenced-work or related-work columns. It
found six retained `W0` source rows and no declared work-target edge. This
fixes the exact incidence to be handled, but it is not yet a row-conserved
filtered relation view and does not validate every other child identifier.

Remaining G-1 work includes full-root content hashing, row-conserved quarantine
propagation, complete child source membership and reference-target coverage,
authorship missingness and identity validity, date/censoring checks,
cutoff-aware panel construction, leakage tests, and a representative pilot.

The report must not be summarized as “492,293,207 valid OpenAlex works,” “the
raw source passed,” or “G-1 passed.” Exact results apply only to the declared
492,293,206-row grammar-valid analysis view, with `W0` retained and unresolved.
