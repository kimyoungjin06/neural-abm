# Pivot-feedback D0 blinded content-review protocol v1

Date: 2026-07-29

Status: `PACKETS_PREPARED_HUMAN_DOUBLE_REVIEW_NOT_RUN`

## Purpose

The automated content/indexing audit recovered all 384 predeclared title
samples but identified a strong indexing-discontinuity threat in the
provisional Medicine episode `F27/T12505/1991`. This protocol prepares the
required human review without exposing reviewers to episode identity, calendar
window, source concentration, topic score, or downstream outcomes.

Packet generation is not review completion. A language model or other
artificial agent may help maintain the packet tooling but may not be counted as
either required human reviewer.

## Reviewer task

Each of two distinct human reviewers receives all 384 sampled titles in a
different deterministic order. Reviewers see only an opaque item ID, target
topic name, title, language, and work type.

For every row, select exactly one label:

| Label | Use when |
| --- | --- |
| `ON_TOPIC` | The title directly supports the target topic or a clear synonym. |
| `AMBIGUOUS` | The title is plausibly related but does not support a clear decision. |
| `OFF_TOPIC` | The title is clearly unrelated or is an archival/inventory object. |
| `UNREVIEWABLE` | The title or language does not support a meaningful assessment. |

Notes are required for `AMBIGUOUS`, `OFF_TOPIC`, and `UNREVIEWABLE`. Reviewers
must complete their packets independently and must not compare decisions before
both files are frozen and hashed.

## Adjudication

After both complete reviewer files are frozen:

1. verify that all 384 rows have one allowed label;
2. verify that reviewer identities are distinct humans;
3. compute exact agreement and Cohen's kappa;
4. send every disagreement to a third adjudicator who is not either original
   reviewer;
5. freeze the adjudicated label and notes; and
6. write a versioned review receipt before defining an indexing-contamination
   exclusion.

Agreement statistics are diagnostics, not thresholds for promoting an episode.
The adjudicated labels must be interpreted together with the predeclared
topic-score, source-concentration, and annual-count diagnostics.

## Blinding and claim boundary

Reviewer packets exclude episode number, field/topic identifiers, entry year,
window, work ID, publication year, topic score, source, and source row
location. The private adjudication key retains these fields solely to merge
completed packets after freeze.

The review must not inspect entry, persistence, realized citation, predictive,
model-selection, or G4 outcomes. Completing it does not itself complete D0 or
authorize leave-out, network, bridge, outcome, G0, model, or G4 work. A
candidate-level, outcome-blind contamination rule must still be applied to the
complete candidate universe, followed by deterministic reselection and
independent replay.
