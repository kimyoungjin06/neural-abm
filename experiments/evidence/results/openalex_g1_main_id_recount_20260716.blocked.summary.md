# OpenAlex G-1 Main-ID Recount: Blocked Result

## Status and gate boundary

The full 493-file body scan is `BLOCKED`. It reproduced all 492,293,207
footer rows and found zero null IDs, but it also found one noncanonical ID:
`https://openalex.org/W-1`. Therefore `pass_main_id_recount=false` and
`pass_g1=false`. The latter means that the empirical G1-signature gate has not
been run; it is not evidence against the scientific hypothesis.

The original zero-invalid contract remains failed. The row is not silently
canonicalized, recoded, or removed in this result.

## Exact observations

| Measure | Value |
| --- | ---: |
| Main-table files scanned | 493 |
| Data-page rows scanned | 492,293,207 |
| Null IDs | 0 |
| Invalid IDs | 1 |
| Exact distinct canonical IDs | unavailable after the blocking check |
| Exact duplicate excess | unavailable after the blocking check |

The malformed value occurred once in
`openalex_works_20260330/part-384.parquet` (sorted file index 317). Its row has
the same value in `id` and `ids__openalex`, an empty title, no DOI, publication
date `2008-08-14`, one reported author, and no primary topic. Those fields make
it consistent with a placeholder-like source record, but do not turn it into a
valid OpenAlex work ID.

This is not a new parser artifact. A source-side key-health report generated on
2026-04-27 independently reported 492,293,207 rows, one malformed key, and the
same `W-1` value with count one. The later upstream dataset profile also
documented that observation.

Critical follow-up inspection also found one `https://openalex.org/W0` row in
`part-322.parquet` (row group 10, file row 340,597). The frozen grammar accepts
`W0`, so it is not folded into the `W-1` failure. It remains a separate
`WARN_ZERO_ID_UNRESOLVED`: its MAG value is `0`, its one authorship row has a
null author ID, and targeted probes found no topic or reference source rows and
no reference-target rows. A later grammar-valid recount must retain and expose
this diagnostic; it must not call `W0` entity-valid or remove it solely from a
current API response.

## Execution and provenance

| Commitment | Value |
| --- | --- |
| Parent inventory fingerprint | `1bee200c61e746ee774715de2521550cb2c7a6f2bf08e0f8b06f43a39c9ec473` |
| Parent manifest SHA-256 | `6b5929a0155c033e818a816aad57095380a3907ab558c8d2cd58a89055f22b46` |
| Plan SHA-256 | `2378b49ffe9523439aee7b26e36c769d9f31a0f6b41f65382ad0ad68baac7c91` |
| Tool SHA-256 | `fb22a03e89b672ae3790e7cad03f1f99ebc04a8c2df543bfda42434de2a427a9` |
| Raw report SHA-256 | `ceb318b378ba23a5418754f4e5c906d71f4c22ceb6e05c1f1c11cfd6f26e9483` |
| Unsorted ID artifact SHA-256 | `e1620784a27fdcfd33683112b94d8ebdeb913d54b914225986e9788709591db9` |
| Prior key-health report SHA-256 | `e75c5aa9b426bdc066647c621cee5c93a5f1ae1ff9f17187005077f842bfab4f` |

The scan took 300.8 seconds, read 5,137,957,807 compressed ID-column bytes,
and peaked at 4,181,056 KiB RSS. The parent inventory was unchanged at the end.
A second invocation revalidated the completed report and the 3,938,345,656-byte
raw artifact checksum before returning the same blocked status.

## Remediation boundary

A follow-up analysis view may quarantine this row only under a new frozen
contract, before any empirical outcome is inspected. That contract must:

- reproduce the exact quarantined value and count;
- keep the physical 492,293,207-row total visible;
- compute exact distinct and duplicate statistics on IDs satisfying the
  published frozen grammar only;
- propagate the same exclusion through authorship, topic, reference, and every
  other child relation; and
- keep the original blocked result as evidence rather than rewriting it.

Until that succeeds, main-ID integrity and G-1 data readiness remain blocked.
