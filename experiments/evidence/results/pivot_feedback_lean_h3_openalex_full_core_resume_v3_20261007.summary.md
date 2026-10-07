# Lean-H3 OpenAlex full-core resume v3

Date: 2026-10-07

Status: `PASS_THE_EXACT_2026_06_26_FULL_CORE_TRANSFER_IS_COMPLETE_NO_ROW_DECODE`

## Why the partial download failed

The v2 resume stopped at HTTP 404 on the first missing file. The
[v2 attempt record](pivot_feedback_lean_h3_openalex_full_core_resume_v2_attempt_20261007.summary.md)
(`e7ae454`) concluded that the 2026-06-26 copy was permanently incomplete.
**That conclusion was wrong.**

The OpenAlex bucket has versioning enabled. The 2026-09-23 snapshot did not
delete the 2026-06-26 objects. It placed a delete marker over each one:

| `works/updated_date=2026-06-26/` | Count |
|---|---|
| Versions written on 2026-06-27 | 160 |
| Delete markers written on 2026-09-23 | 159 |
| New object written on 2026-09-23 (`part_0000`) | 1 |

An unversioned URL addresses the current object. For these files that is the
delete marker, so the v1 URLs returned 404. A URL with `?versionId=` still
returns the original object, and range requests on it work.

## What ran

The v3 authorization (`29de3e2`) changed only how the remote objects are
addressed. It bound one version to each of the 71 missing files. Each was the
unique version written on 2026-06-2x whose size equals the plan's. On the 89
files of the same partition that the v1 run had completed, this rule selects
the version id, ETag, size, and date the v1 run recorded, 89 of 89.

| | Value |
|---|---|
| Files | 71, Works `updated_date=2026-06-26` parts 0089–0159 |
| Bytes | 35,183,477,118 |
| Records, by the manifest | 25,462,673 |
| Run | 07:18–08:59 UTC, 100 minutes |

## Controls

- The 117,440,512 bytes already on disk for `part_0089` matched a hash of the
  same range of the bound version before the transfer continued.
- Every HEAD returned the bound version id, ETag, size, and date, and the
  state records that identity for all 71 files.
- After the transfer, the S3 multipart ETag of each file was recomputed
  locally with 64 MiB parts. All 71 equal the bound ETag.
- The v1 terminal receipt passed with every file's SHA-256.

## Result

The local copy at `/ssd/pivot-feedback-lean-h3-openalex-full-core-v1-20260806`
is now the exact 2026-06-26 manifest: 2,454 of 2,454 files and
724,973,106,144 bytes. This answers O18: the source is the complete 2026-06-26
snapshot.

No row value was decoded, no G1 label or G4 row was read, and no model was
fitted.

## Next

- **N10.** Preregister the empirical outcome selection trend before any row is
  read: episodes, years, the persistence definition and horizon, and the
  failure actions.
- **Capacity.** `/ssd` now has 929 GB free. The v1 transform-and-replay
  reserve of 1,087 GB, deferred by v2, exceeds that. Any transform stage must
  register its own budget and check it before it starts. Transform remains
  unauthorized.
