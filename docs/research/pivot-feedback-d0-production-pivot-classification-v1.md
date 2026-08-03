# Pivot-feedback D0 production pivot classification v1

Date: 2026-07-31

Status:
`PASS_INDEPENDENT_EXACT_FOUR_ALL_ROW_OBSERVABLE_HISTORY_STATE_REPLAY`

## Result

The frozen threshold-free four-state definition was applied to every one of
the 3,256,271 registered author-episode rows in the independently verified
three-count source. No source row was filtered and no source feature value was
changed.

| Observable-history state | Rows |
|---|---:|
| `NO_FOCAL_ENTRY` | 3,246,984 |
| `NEW_TOPIC_ENTRY` | 4,301 |
| `RETURN` | 125 |
| `CONTINUATION` | 4,861 |
| Total | 3,256,271 |

The four states form an exhaustive and mutually exclusive partition. There are
9,287 focal-positive rows, equal to the sum of the three focal-positive
states.

## Exact definition

The classifier uses only three nonnegative counts already present in the
verified source:

- `NO_FOCAL_ENTRY`: focal count is zero;
- `NEW_TOPIC_ENTRY`: focal count is positive and the pre-entry count is zero;
- `RETURN`: focal and pre-entry counts are positive while the trailing
  three-year count is zero; and
- `CONTINUATION`: focal and trailing three-year counts are positive.

There is no minimum-history threshold or graph/team cap. `NEW_TOPIC_ENTRY` is
the primary pivot state, but it means first entry within the observed
current-snapshot history under the frozen work/topic ABI. It is not a
true-lifetime-first claim.

## Episode accounting

| Episode | Field | Topic | Entry year | Rows | No focal | New-topic entry | Return | Continuation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 14,430 | 2003 | 549,629 | 547,540 | 950 | 15 | 1,124 |
| 2 | 17 | 10,579 | 2012 | 1,215,043 | 1,211,416 | 1,441 | 36 | 2,150 |
| 3 | 27 | 10,151 | 1993 | 1,330,658 | 1,327,869 | 1,442 | 51 | 1,296 |
| 4 | 27 | 12,421 | 1953 | 160,941 | 160,159 | 468 | 23 | 291 |

## Independent replay

The production classifier used DuckDB. A separate PyArrow/NumPy verifier did
not import the producer and did not use DuckDB. It compared all eight source
columns and independently recomputed both output columns on all 3,256,271
rows.

The replay found:

- zero source-column mismatches;
- zero state mismatches;
- zero primary-boolean mismatches;
- zero filtered source rows; and
- zero source feature mutations.

## Boundary and next gate

This result establishes a deterministic event classification, not persistence,
impact, association, causality, or scientific effect. The build and verifier
read no raw work or author-event partition and no post-entry outcome. They
created no API request, fit no model, and did not open G4.

The safe successor is a design-and-fixtures-only contract for the D0
post-entry persistence outcome source. That contract must resolve the exact
`t+1..t+3` discovery-only outcome, work/topic ABI, temporal maturation and
right-censoring rules, missingness, immutable source bindings, and an
independent replay plan before any production post-entry row is read.
