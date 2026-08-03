# Pivot-feedback threshold-free researcher-pivot definition v2

Date: 2026-07-31

Status:
`FROZEN_THRESHOLD_FREE_OBSERVABLE_HISTORY_PIVOT_DEFINITION_V2_FIXTURES_ONLY`

## Primary decision

The primary pivot event is an observable-history new-topic entry:

> An author token has at least one content-confirmed eligible focal-year work
> in the target primary topic and has no eligible work in that topic through
> `t-1`.

This is not a true lifetime-first claim. Topic membership and author identity
remain bounded by the current OpenAlex snapshot and the registered measurement
ABI.

## Threshold-free four-state table

| State | Exact rule |
|---|---|
| `NO_FOCAL_ENTRY` | No content-confirmed eligible focal-year target-topic work |
| `NEW_TOPIC_ENTRY` | Focal entry exists and the pre-entry topic count is zero |
| `RETURN` | Prior topic work exists, but none in `t-3..t-1` |
| `CONTINUATION` | Target-topic work exists in `t-3..t-1` |

The classifier uses only three nonnegative counts:

1. pre-entry target-topic work count through `t-1`;
2. target-topic work count during `t-3..t-1`; and
3. content-confirmed eligible focal-year target-topic work count.

There is no minimum prior-year or prior-work threshold. Observable span and
prior eligible-work counts remain sensitivity descriptors and cannot change a
primary state.

## Why this supersedes v1

The old v1 decision table used fewer than three observable years or fewer than
three prior works as `INSUFFICIENT_HISTORY`. The rebuilt predecision
measurement surface instead registers every structurally observed
author-opportunity row and selects no history threshold. Applying the old
three-year/three-work rule would silently add an unregistered eligibility
choice.

The v2 definition preserves v1's valid boundaries: history stops at `t-1`,
future persistence and impact cannot define entry, and the claim concerns
observable history rather than a researcher's true lifetime.

## Current boundary

The machine-readable contract and verifier execute synthetic boundary fixtures
only. The trailing-three-year and focal-year production sources are not yet
contracted. No production pivot labels, outcome rows, API requests, model
inputs, G0 decision, G4 result, association, or scientific effect are created
by this definition freeze.
