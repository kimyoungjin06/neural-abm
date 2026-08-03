# OpenAlex Main Temporal v2 UTC-ms Parser Pilot

- Status: `BLOCKED_DIRTY_PARSER_PILOT_CREATED_UPDATED_REVERSALS`
- Scope: one file, 1,000,000 retained rows
- GPU used: `false`
- Component promoted: `false`
- Full v2 execution: `NOT_RUN`

## What passed

All 1,000,000 retained `created_date`/`updated_date` pairs parsed under the exact
`YYYY-MM-DDTHH:MM:SS.sssZ` contract. Metadata nulls, parse failures, future
timestamps, publication parse failures, publication year mismatches, and
accounting failures were all zero.

## What blocked promotion

Three rows had `created_date > updated_date`, so
`pilot_local_contract_clean=false`. They occur in
`openalex_works_20260330/part-0.parquet`, row group 25:

| OpenAlex work | File row | Created (UTC) | Updated (UTC) | Created minus updated |
| --- | ---: | --- | --- | ---: |
| `W2410776720` | 844,168 | `2025-10-10T00:00:00.000Z` | `2025-08-28T14:23:35.770Z` | 3,663,384,230 ms |
| `W3142421094` | 844,170 | `2025-10-10T00:00:00.000Z` | `2025-09-03T13:49:58.679Z` | 3,147,001,321 ms |
| `W2793775362` | 844,171 | `2025-10-10T00:00:00.000Z` | `2025-09-08T15:44:56.852Z` | 2,708,103,148 ms |

The exact UTC-ms parser therefore works on this file, but the assumed ordering
is not a clean local invariant. No exception allowlist is justified from this
pilot, and a full v2 run was not started.

## Claim boundary

This nonpromotable pilot does not estimate full-snapshot reversal prevalence,
prove that the three observations are source errors, establish the full
temporal component, pass G-1 or G0, run empirical G1, or establish a scientific
effect.

- Receipt SHA-256: `ec59f3c649baa4a6a5333daace92a5e46f6492fad62e4c1ed42a1702394425f4`
- Result fingerprint: `4f3dc7a31e1e2a1d4b92a021edfb544d237ef93c148754d9ecdf5f20d366df16`
