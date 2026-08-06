# Lean H3 OpenAlex taxonomy download dry-run v1

Date: 2026-08-06

Status:
`PASS_INDEPENDENT_TAXONOMY_RANGE_RESUME_HASH_AND_FOOTER_CLOSURE_NO_ROW_DECODE`

## Outcome

The receipt-bound downloader completed a deliberately interrupted and resumed
transfer of the eight Topics, Subfields, and Fields files from the captured
OpenAlex 2026-06-26 Parquet manifests. It transferred 2,783,017 bytes in total,
exactly equal to the manifest sum, and produced a local SHA-256 inventory under
`/ssd/pivot-feedback-lean-h3-openalex-taxonomy-dry-run-v1-20260806`.

The first run requested only 4,096 bytes of the Fields file and stopped at
`INTENTIONALLY_PAUSED_MAX_NEW_BYTES`. The second run validated the `.part`
size, prefix SHA-256, content length, ETag, Last-Modified value, and S3 version
identifier before resuming. Completion required nine HTTP range requests: two
for the interrupted Fields file and one for each of the other seven files.
Total remote bytes remained 2,783,017, so the completed prefix was not
redownloaded.

A third terminal invocation returned
`PASS_TERMINAL_LOCAL_REVALIDATION_NO_REMOTE_ACCESS` after validating only the
local plan, state, receipt, exact file surface, sizes, modes, and SHA-256 values.

## Independent replay

The independent verifier did not import the downloader. It reconstructed the
eight exact URLs from the captured manifests, checked the download root and all
control/data hashes, then opened only local Parquet footers. The replay found:

| Entity | Files | Footer rows | Manifest bytes |
| --- | ---: | ---: | ---: |
| Topics | 6 | 4,516 | 2,589,362 |
| Subfields | 1 | 252 | 174,417 |
| Fields | 1 | 26 | 19,238 |
| Total | 8 | 4,794 | 2,783,017 |

Every data file is `0600`, every directory is `0700`, no `.part` file remains,
and the downloader plan/state/receipt hashes are bound in the JSON summary.
No Parquet row value was decoded.

## Fail-closed scope

The CLI accepts only `taxonomy-dry-run` and one exact output root. Its plan
explicitly marks Works transfer, source-row decoding, and confirmatory
execution as unauthorized. It has no executable full-core or Works scope.

Therefore this result establishes the transfer mechanism for a small bounded
sample only. It does not authorize or establish the 724.97 GB Works transfer,
full-file Works schema unanimity, historical completeness, H3 horizon maturity,
eligible discovery or confirmatory cohorts, estimator precision, an
association, null, mechanism, causal effect, or scientific effect.

## Next bounded decision

The next decision is whether to authorize a capacity-safe Works transfer. If
authorized, the successor receipt must add an explicit full-core scope while
preserving the captured URL list, 724,973,106,144-byte raw cap, 1.812 TB total
reserve, content-length/remote-identity resume checks, and receipt-last local
SHA-256 inventory. No Works byte has been materialized by this dry-run.
