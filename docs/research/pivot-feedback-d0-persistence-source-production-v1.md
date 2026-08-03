# Pivot-feedback D0 persistence source production v1

Date: 2026-07-31

Status:
`PASS_INDEPENDENT_EXACT_4301_THREE_YEAR_PERSISTENCE_SOURCE_ALL_ROW_REPLAY`

## Result

The frozen persistence-source contract was applied to all 4,301 independently
verified `NEW_TOPIC_ENTRY` rows. Every row has a complete `t+1` through `t+3`
window.

| Episode | Field | Topic | Entry year | Rows | Persistent | No persistence | Descriptive rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 14,430 | 2003 | 950 | 289 | 661 | 30.42% |
| 2 | 17 | 10,579 | 2012 | 1,441 | 572 | 869 | 39.69% |
| 3 | 27 | 10,151 | 1993 | 1,442 | 529 | 913 | 36.69% |
| 4 | 27 | 12,421 | 1953 | 468 | 97 | 371 | 20.73% |
| Total | — | — | — | 4,301 | 1,487 | 2,814 | 34.57% |

The binary state is positive when the registered author token has at least one
eligible work in the same certified primary topic during `t+1..t+3`. The
source contains 3,689 distinct qualifying author-work pairs: 1,315 at `t+1`,
1,281 at `t+2`, and 1,093 at `t+3`.

## Source and denominator accounting

The producer rehashed all 4,096 cutoff-work partitions and all 4,096
author-event partitions before reading them. It accounted for 492,293,206
source work rows and 779,630,258 source author-event rows.

The output keeps the exact 4,301-row classification denominator:

- zero denominator rows were filtered or mutated;
- zero rows were right-censored;
- zero confirmatory entry-year rows were read;
- zero post-entry offsets outside `t+1..t+3` were read;
- zero API requests were created; and
- no citation, impact, peer-exposure, model, or G4 data were joined.

## Independent replay

The producer used DuckDB. A separate PyArrow/NumPy verifier neither imported
the producer nor used DuckDB. It independently rehashed all 8,192 source
partitions, reconstructed the `NEW_TOPIC_ENTRY` denominator, rebuilt the
eligible author-work pairs, and compared every key and value in all 4,301
output rows.

The replay found zero mismatches in keys, offset counts, total counts,
follow-up completeness, pivot state, persistence state, and persistence
boolean. Source metadata checks also found zero year, temporal-disposition,
declared-author-count, and primary-field mismatches among joined rows.

## Interpretation boundary

These are descriptive current-snapshot-annotated persistence counts. The four
episode rates are not adjusted comparisons: the topics, fields, entry years,
and opportunity structures differ. The table therefore cannot rank fields or
episodes, identify an exposure association, establish causality, or support a
scientific-effect claim.

The safe successor may summarize only this independently verified 13-column
artifact for D0 measurement calibration. It must not join exposure, citation,
impact, peer, confirmatory, or holdout data; use the episode rates to select
fields; freeze G0; fit a model; or open G4.
