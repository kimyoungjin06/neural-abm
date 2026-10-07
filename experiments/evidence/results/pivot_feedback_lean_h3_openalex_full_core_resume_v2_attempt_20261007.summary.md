# Lean-H3 OpenAlex full-core resume v2 attempt

Date: 2026-10-07

Status: `STOPPED_FAIL_CLOSED_THE_EXACT_2026_06_26_SNAPSHOT_IS_NO_LONGER_AVAILABLE_AT_THE_SOURCE`

## What happened

The v2 authorization (`625ed69`) changed only the runtime capacity rule, so the
transfer could resume with 898 GB free on `/ssd`. The network-free preflight
passed. It rebuilt the v1 plan byte for byte, validated the state, and showed a
margin of 821 GB under the v2 rule. Under the v1 rule the margin would have
been −159 GB.

The resume then stopped fail-closed at the first pending file. Its remote copy,
`works/updated_date=2026-06-26/part_0089.parquet`, returned HTTP 404. No bytes
were written. The only change to the state was its run counter, now 136.

## Why

OpenAlex replaces its public snapshot in place. A new snapshot dated
2026-09-23 has overwritten or removed the 2026-06-26 objects:

| Object | Now |
|---|---|
| `works/updated_date=2026-06-26/part_0089` and `part_0090` | 404 |
| `topics/updated_date=2026-06-26/part_0000` | 404 |
| `works/updated_date=2026-06-26/part_0000` | A different object: 17.7 MB written on 2026-09-23. The local copy is 501.6 MB, written on 2026-06-27 |

The local 2026-06-26 copy is therefore **permanently incomplete**. It holds
2,383 of 2,454 files and lacks 25,462,673 Works records (35.2 GB).

## The new snapshot

Its four manifests were captured, metadata only, so the option can still be
decided:

| | Value |
|---|---|
| Works records | 476,196,327 |
| Files | 2,046 |
| Bytes | 707,144,414,213 |
| Local files identical in path, size, and record count | 0 |

No local file can be reused, so taking this snapshot would mean a full 707 GB
transfer. That transfer is not authorized.

## Decision needed

- **Transfer the 2026-09-23 snapshot in full.** This would need a new
  exact-manifest authorization. The transfer must finish before OpenAlex
  replaces the snapshot again.
- **Use the local 2026-03-30 snapshot** on `/hdd`, with its G1-derived tables
  on `/ssd`, under an amended lean-H3 design.

No row value was decoded, and no data bytes were transferred.
