# Pivot-feedback D0 persistence calibration readout v1

Date: 2026-07-31

Status:
`PASS_INDEPENDENT_EXACT_4301_DESCRIPTIVE_PERSISTENCE_CALIBRATION_ALL_VALUE_REPLAY`

## Result

The aggregate-only readout rehashed and read the independently verified
4,301-row persistence source. It read no raw work/event partition and wrote no
author identifier.

Of 4,301 observable-history `NEW_TOPIC_ENTRY` rows, 1,487 have at least one
eligible target-topic work during `t+1..t+3` and 2,814 have none. The
unadjusted descriptive rate is 34.57%.

Among the 1,487 positive rows, the first observed subsequent target-topic work
appears:

| First observed offset | Rows | Share of positive rows |
|---|---:|---:|
| `t+1` | 837 | 56.29% |
| `t+2` | 401 | 26.97% |
| `t+3` | 249 | 16.75% |

The three-year eligible target-topic work-count distribution is:

| Works in `t+1..t+3` | Rows |
|---|---:|
| 0 | 2,814 |
| 1 | 722 |
| 2 | 325 |
| 3 | 174 |
| 4 | 90 |
| 5 or more | 176 |

Of the positive rows, 722 (48.55%) have exactly one qualifying work, while 765
have at least two. A total of 190 rows have target-topic work in each of the
three follow-up years. These are author-token/work-accounting descriptions,
not researcher-level productivity or quality measures.

## Episode readout

| Episode | Rows | Persistent | Rate | First at t+1 | First at t+2 | First at t+3 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 950 | 289 | 30.42% | 115 | 98 | 76 |
| 2 | 1,441 | 572 | 39.69% | 394 | 127 | 51 |
| 3 | 1,442 | 529 | 36.69% | 285 | 147 | 97 |
| 4 | 468 | 97 | 20.73% | 43 | 29 | 25 |

The rates are intentionally unadjusted. The four rows refer to different
topics, fields, entry years, and opportunity structures, so the table cannot
be used to rank or select them.

## Independent replay

The producer used vectorized PyArrow/NumPy aggregation. The verifier did not
import that implementation and independently iterated through every source
row using Python counters. It recomputed the overall and four episode
summaries, six work-count bins, four first-offset categories, and eight yearly
presence patterns.

All overall, episode, distribution, and invariant mismatch counts are zero.
The source receipt, source artifact, prior independent verification, readout
report, Markdown, and receipt are hash-bound.

## Boundary and successor

No exposure, peer, citation, impact, affiliation, confirmatory, holdout, API,
model, or G4 data was read. No adjusted comparison, inferential interval,
outcome-based ranking, or scientific effect was computed.

This completes the bounded D0 entry-and-persistence measurement calibration
path. It does not complete or freeze G0. The safe successor is a
metadata-and-contract-only G0 readiness inventory that maps the independently
verified D0 decisions to every unresolved G0 section without opening
confirmatory rows, choosing values by outcome, fitting a model, or opening G4.
