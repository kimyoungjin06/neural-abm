# Pivot Feedback D0 API-Assisted Content Review Protocol v2

Status: `BATCH_REQUESTS_NOT_SUBMITTED_HUMAN_CALIBRATION_NOT_RUN`

This protocol replaces the planned execution path, not the evidence already
created under v1. The empty v1 human-review packets remain immutable evidence
that packet generation occurred and review did not. They are not relabeled as
completed work.

## Purpose

The v2 path uses two pinned API model snapshots to screen every one of the 384
outcome-blind title records. Humans then review a deterministic calibration
sample and every escalated record. This reduces routine human labeling while
retaining an empirical check on correlated model error.

Neither API judge is a human reviewer. API agreement alone cannot complete D0.

## Information boundary

Each API request and each human packet contains only:

- target topic name;
- work title;
- language; and
- work type.

Episode identity, field/topic IDs, calendar year and window, OpenAlex ID,
topic score, source, source-row location, authors, citations, treatment,
entry, persistence, and all downstream outcomes are excluded. API tools and
web search are disabled, and Responses API storage is requested as `false`.

## API screen

Every title is independently sent to:

1. `gpt-5-mini-2025-08-07`; and
2. `gpt-4.1-mini-2025-04-14`.

The exact model snapshots, prompt, JSON schema, request bytes, opaque request
IDs, and response request IDs are retained. No temperature field is sent; the
pinned snapshot and provider default are part of the frozen contract. Each
judge returns one of `ON_TOPIC`, `AMBIGUOUS`, `OFF_TOPIC`, or `UNREVIEWABLE`,
a short rationale, and `HIGH`, `MEDIUM`, or `LOW` confidence.

The prepared JSONL uses the OpenAI Batch API `/v1/responses` format. Batch
preparation does not submit a request and incurs no API execution.

## Human calibration

The base calibration set contains 96 titles: eight deterministic records from
each of the 12 episode-by-window strata. Two distinct humans review this same
set independently while blind to episode metadata and API outputs. A third
human adjudicates their disagreements after both files are frozen.

The human set expands to include the union of records where:

- the API judges disagree;
- either API label is not `ON_TOPIC`;
- either confidence is not `HIGH`; or
- a request or response is invalid.

The base and escalation sets are deduplicated. Reviewer discussion is not
allowed before independent files are frozen.

## Fail-closed rules

The API-assisted path fails and reverts to double-human review of all 384
titles if any of the following occurs:

- both APIs say `ON_TOPIC` but the adjudicated base label is not `ON_TOPIC`;
- any API response is missing or invalid;
- any required human label is missing;
- human exact agreement is below 0.80; or
- human Cohen's kappa is below 0.60 when kappa is defined.

These rules prioritize detecting false-clean decisions. They do not claim that
two model outputs are independent human evidence.

## What this stage cannot establish

A successful 384-title calibration does not apply a contamination rule to the
70,277-row candidate universe. That requires a separately frozen successor,
followed by deterministic episode reselection and independent replay. Until
then, episode 3 stays quarantined, the other episodes are not automatically
promoted, the old author panel is not current, and D0, outcomes, G0, model
fitting, G4, and scientific-effect claims remain closed.
