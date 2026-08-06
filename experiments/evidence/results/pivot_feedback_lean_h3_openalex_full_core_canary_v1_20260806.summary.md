# Lean H3 OpenAlex full-core transfer canary v1

Date: 2026-08-06

Status:
`PASS_INDEPENDENT_FULL_CORE_CANARY_MANIFEST_PREFIX_FOOTER_AND_CAPACITY_CLOSURE_NO_ROW_DECODE`

## Outcome

The exact-manifest Works plus Topics/Subfields/Fields transfer is authorized
and its registered 64 MiB canary passed. The plan contains exactly 2,454 files
and 724,973,106,144 bytes. Authors is absent, mutable-prefix synchronization is
forbidden, and the runtime stops if available space falls below remaining
source bytes plus the 1,087,459,659,216-byte working/transformation reserve.

The canary intentionally paused at exactly 67,108,864 cumulative bytes:

| Entity | Prefix files | Canary bytes |
| --- | ---: | ---: |
| Works | 201 | 64,325,847 |
| Topics | 6 | 2,589,362 |
| Subfields | 1 | 174,417 |
| Fields | 1 | 19,238 |
| Total | 209 | 67,108,864 |

At the pause, 208 files were complete and one Works file had an 11,730,808-byte
verified prefix. The remaining capacity margin under the registered reserve
rule was 1,211,315,301,968 bytes.

## Interrupted-attempt disposition

The first canary attempt stopped after 12,576,018 bytes and 70 complete files
because of a temporary DNS name-resolution failure. No partial file was open
at the failed request. Review found that the initial implementation treated the
64 MiB cap as invocation-local, which could have exceeded the registered total
after a retry.

Before resuming, the implementation was corrected to enforce 64 MiB as a
cumulative cap across retries and to reject any uncapped resume before an
immutable canary receipt exists. Both downloader SHA-256 values and the exact
interruption disposition are retained in the canary lineage. The resumed run
reused the 70 complete files and stopped at the registered cumulative total.

## Independent verification

The verifier independently reconstructed all 2,454 URLs from the captured
manifests. It rehashed all 209 canary prefixes, opened only the 208 complete
local Parquet footers, and confirmed 24,814 manifest-matched footer rows. The
prefix inventory tree SHA-256 is
`92b4660a2e56969b575d7d94dfc797393ce17c03ba5453ec55541ce44521ca4a`.
No Parquet row value was decoded.

## Boundary and next action

This is transfer-canary passage, not full download completion or source
readiness. It does not establish historical completeness, H3 horizon maturity,
eligible cohorts, precision, an association, null, mechanism, causal effect,
or scientific effect. Current G1, model-selection, G4, and holdout boundaries
are unchanged.

The independently verified next action is background resumption of the same
exact plan. The process must retain single-file execution, remote identity and
segment-hash resume checks, the live capacity invariant, and receipt-last
completion. Row decoding remains unauthorized.
