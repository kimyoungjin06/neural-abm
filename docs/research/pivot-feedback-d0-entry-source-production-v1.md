# Pivot-feedback D0 production entry source v1

Date: 2026-07-31

Status:
`PASS_INDEPENDENT_EXACT_FOUR_THREE_COUNT_SOURCE_REPLAY_NO_STATE_LABELS`

## Result

The exact-four production feature source is complete and independently
verified. It retains all 3,256,271 registered author-episode rows and contains
only:

- the five private numeric key columns;
- `pre_entry_target_topic_work_count`;
- `trailing_3y_target_topic_work_count`; and
- `focal_confirmed_eligible_target_topic_work_count`.

No production classifier state or pivot label is present.

## Full source replay

The DuckDB producer and the independent PyArrow/NumPy verifier both processed
all 492,293,206 receipt-bound work rows and all 779,630,258 receipt-bound
author-event rows. The verifier rehashed every one of the 4,096 work and 4,096
event partitions before replay.

Within the four bounded `t-3..t` windows, 36,138 works passed the exact
topic/field and frozen scholarly-work ABI. Of these, 35,057 had a valid author
token and 1,081 did not. The linked works yielded 70,008 distinct author-work
pairs: 61,030 belonged to the registered measurement population and 8,978 did
not.

There were zero source metadata mismatches for publication year, temporal
disposition, declared author count, or field, and zero duplicate
author-work-pair excess rows.

## Feature accounting

| Episode | Registered author rows | Trailing pairs | Focal pairs | Focal-positive authors |
|---:|---:|---:|---:|---:|
| 1 | 549,629 | 12,957 | 2,542 | 2,089 |
| 2 | 1,215,043 | 18,432 | 6,534 | 3,627 |
| 3 | 1,330,658 | 11,469 | 4,105 | 2,789 |
| 4 | 160,941 | 4,053 | 938 | 782 |
| Total | 3,256,271 | 46,911 | 14,119 | 9,287 |

The independent replay found:

- zero pre-entry value mismatches;
- zero trailing-count mismatches;
- zero focal-count mismatches;
- zero `trailing > pre-entry` violations;
- zero positive-trailing rows with zero pre-entry count; and
- a globally sorted, unique author-episode key.

## Boundary

`content-confirmed` remains an episode-topic certificate plus exact
current-snapshot primary-topic membership. It is not an individual-work API
label, semantic ground truth, or historical topic observation.

The build created no API request and read no post-entry persistence, impact,
model, or G4 outcome. A successor gate may authorize deterministic application
of the already frozen four-state classifier to these three counts. That later
classification must remain separate from persistence and scientific-effect
claims.
