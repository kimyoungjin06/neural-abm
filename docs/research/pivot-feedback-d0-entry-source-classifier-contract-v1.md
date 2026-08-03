# Pivot-feedback D0 entry-source and classifier contract v1

Date: 2026-07-31

Status:
`FROZEN_EXACT_FOUR_ENTRY_SOURCE_CLASSIFIER_CONTRACT_FIXTURES_ONLY`

## Decision

The three inputs to the threshold-free pivot definition now have an exact,
receipt-bound source contract:

| Input | Source and window |
|---|---|
| `pre_entry_target_topic_work_count` | Verified predecision author measurement, through `t-1` |
| `trailing_3y_target_topic_work_count` | Distinct eligible author-linked target-topic works in `t-3..t-1` |
| `focal_confirmed_eligible_target_topic_work_count` | Distinct eligible author-linked target-topic works in `t` |

All 3,256,271 registered author-episode rows must be retained by a left join.
The trailing and focal counts use the receipt-bound normalized work metadata
and author-work event partitions. They inherit the frozen scholarly-work ABI,
deduplicate exact author-work pairs, apply no team-size cap, and do not use any
post-entry work.

## Meaning of content-confirmed

`content-confirmed` is a topic-level certification, not a new label attached to
each work. Each of the four selected episode topics already has a frozen
`CONTENT_PASS` global selection certificate. A focal work is therefore counted
when:

1. its current-snapshot primary topic and field exactly match the certified
   episode;
2. it passes the frozen work-type, paratext, retraction, and temporal ABI; and
3. at least one valid current-snapshot author-token event links the work.

No additional OpenAI, Claude, OpenAlex, or human review call is needed for this
feature. The earlier provider-assisted evidence helped certify episode topics;
it is not individual-work semantic ground truth. Current-snapshot topic
membership also does not establish historical topic observability.

## Exact temporal and missingness rules

The four entry years range from 1953 to 2012, before the 2026 snapshot year, so
their work rows must have temporal disposition `2`. The general ABI still
allows dispositions `2` and `3`, but disposition `3` cannot occur for these
four focal windows.

Works outside `t-3..t`, with a missing or nonmatching primary topic, with a
nonmatching field, with an excluded work type, marked paratext, or marked
retracted are excluded and reported. A null `is_retracted` value is retained
when all other eligibility rules pass and is separately reported. A work with
no valid author token is excluded from author counts and reported.

## Independent fixture replay

The verifier reads and hashes five operator receipts but opens no production
Parquet file. It independently replays a 13-work, 13-event fixture covering:

- duplicate author-work events;
- outside-window, wrong-topic, paratext/work-type, retracted, and invalid
  temporal rows;
- a work with no valid author token;
- a retained null-retraction row; and
- all four classifier states: `NO_FOCAL_ENTRY`, `NEW_TOPIC_ENTRY`, `RETURN`,
  and `CONTINUATION`.

The replay reconstructs all five expected author rows and every accounting
total with zero mismatch.

## Current boundary

This freeze resolves the sources and classifier semantics only. It reads no
production Parquet rows, creates no API requests, and opens no persistence,
impact, model, or G4 outcome. A successor gate may authorize construction of
the three-count production source table. Production pivot labels remain a
separate later step.
