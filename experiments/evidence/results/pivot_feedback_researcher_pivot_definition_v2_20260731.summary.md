# Pivot-feedback researcher-pivot definition v2

Status:
`PASS_THRESHOLD_FREE_PAST_ONLY_OBSERVABLE_HISTORY_PIVOT_DEFINITION_V2`

The v2 primary event is `NEW_TOPIC_ENTRY`: an author token has at least one
content-confirmed eligible focal-year target-topic work and zero eligible
target-topic works through `t-1`.

The complete decision table is:

| State | Rule |
|---|---|
| `NO_FOCAL_ENTRY` | focal count = 0 |
| `NEW_TOPIC_ENTRY` | focal count >= 1 and pre-entry topic count = 0 |
| `RETURN` | focal count >= 1, prior topic count > 0, trailing-three-year count = 0 |
| `CONTINUATION` | focal count >= 1 and trailing-three-year count > 0 |

The old v1 minimum of three prior observable years and three prior eligible
works is removed. Those two counts are sensitivity descriptors only. Three
sensitivity profiles ranging from `0/0` to `40/500` produced zero state
changes.

The validator passed seven valid boundary fixtures and rejected six invalid
fixtures. It also enumerated 45 valid bounded count tuples with all four states
represented and zero mismatches.

Only the pre-entry topic count has a verified production source at this stage.
The trailing-three-year count and focal-year content-confirmed count remain
uncontracted. The definition and fixture validation read no production author,
focal-year, future, outcome, API, model, or G4 rows and do not authorize
production pivot classification.
