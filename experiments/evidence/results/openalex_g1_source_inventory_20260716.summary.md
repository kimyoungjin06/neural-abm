# OpenAlex G-1 Source Inventory Summary

Date: 2026-07-16

Status: `INVENTORY_ONLY`

`PASS_METADATA=false`; `PASS_G1=false`

The footer-only audit completed in 15.25 seconds with eight workers and a peak
RSS of 287,248 KiB. It read Parquet metadata, not column data pages. The source
path is redacted; the public identifier is
`openalex_works_20260330_repairreplay_20260410_190630`.

| Measure | Observed |
| --- | ---: |
| Table directories | 23 |
| Parquet files | 21,019 |
| Bytes | 988,404,537,807 |
| Footer rows | 20,996,232,817 |
| Row groups | 650,939 |
| Serialized footer bytes | 808,479,308 |
| Blocking checks | 0 |
| Warnings | 3 |
| Pending content checksums | 23 |

The inventory fingerprint is
`1bee200c61e746ee774715de2521550cb2c7a6f2bf08e0f8b06f43a39c9ec473`.
The raw ignored-run manifest SHA-256 is
`6b5929a0155c033e818a816aad57095380a3907ab558c8d2cd58a89055f22b46`.

## Findings

- Main-work footer rows are 492,293,207 and complete footer statistics report
  zero null work IDs. This is not an exact distinct-ID recount.
- Authorship footer rows are 1,361,551,038. Author ID is null in 580,116,420
  rows (42.607%), matching the frozen source profile but remaining unresolved
  for eligible-author coverage and panel construction.
- Mesh has 1,212,666,618 rows after a historically documented pooled repair
  replaced `base-0` and `base-213` with two repartitioned repair files and
  removed 45 duplicate rows. The original files lack per-file checksums.
- Awards has one canonical names/types/nullability schema and two physical
  column orders. This is an order-drift warning, not semantic schema failure.
- The exception table is raw exception payload, not a validated abstract-text
  source, and is excluded from the primary text branch.

## Boundary

This inventory freezes a physical source profile only. It does not establish
distinct IDs, join coverage, author validity, temporal leakage safety,
representative pilot status, empirical effects, or analysis readiness. The next
step is the separate resumable CPU/I/O body recount defined in the prospective
protocol; full-root content SHA-256 is also still pending.
