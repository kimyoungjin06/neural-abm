# Pivot-feedback D0 API-only topic routing v1

## Purpose

This successor reduces routine API use without treating rules or API agreement
as ground truth. It routes the frozen 384-title diagnostic sample into:

- `NO_API_RULE_ACCEPT`;
- `NO_API_RULE_REJECT`;
- `API_REQUIRED_AMBIGUOUS`; or
- `QUARANTINE_UNRESOLVED`.

The executable topic cards are frozen in
[`pivot_feedback_d0_api_only_topic_routing_v1.yaml`](../../experiments/configs/pivot_feedback_d0_api_only_topic_routing_v1.yaml).

## Topic reference boundary

The four OpenAlex topic descriptions and keywords were retrieved from the
official topic API on 2026-07-29 and stored in
[`pivot_feedback_topic_cards_openalex_snapshot_20260729.json`](../../experiments/evidence/reference/pivot_feedback_topic_cards_openalex_snapshot_20260729.json).
They describe the current OpenAlex clusters. They do not reconstruct topic
meaning when historical papers were published or when an earlier OpenAlex
snapshot assigned them.

This distinction is material for `T12505`. Its current description emphasizes
fibroblast activation protein, tumor stroma, and oncology, while the 1991
diagnostic sample contains broad foundational peptidase biochemistry. The card
therefore auto-accepts explicit peptidase or protease anchors but routes mixed
ontology-versus-cluster cases to the API exception queue.

## Routing logic

Titles are HTML-stripped, ASCII-folded, lowercased, tokenized, and normalized.
The precedence is:

1. opaque identifiers and uninformative titles are quarantined without an API
   request;
2. a positive card match without a negative match is rule-accepted;
3. a negative card match without a positive match is rule-rejected; and
4. every mixed or unmatched case requires API review.

Current-snapshot topic score and source name cannot cause rule acceptance.
Prior API labels are joined only after routing to report descriptive agreement
and a counterfactual request count. They do not execute or tune a row's route.

## Why this is a dry-run

The 70,277 rows in the existing candidate universe are topic-year candidates,
not title-bearing candidate works. A complete production request manifest
therefore requires a separately frozen work-level materialization across those
topic-year windows. The 384 titles cannot stand in for that universe.

This stage makes no network request, reads no credential, and incurs no API
cost. It cannot authorize paid submission, production rule dispositions,
episode reselection, author-panel rebuilding, pivot classification, outcomes,
model fitting, or G4.

## Frozen dry-run result

The independent replay matched all 384 rows and every artifact hash:

| Route | Rows |
|---|---:|
| `NO_API_RULE_ACCEPT` | 229 |
| `NO_API_RULE_REJECT` | 24 |
| `API_REQUIRED_AMBIGUOUS` | 94 |
| `QUARANTINE_UNRESOLVED` | 37 |

The router therefore avoids 290 counterfactual requests (75.5208%) and sends
94 rows (24.4792%) to the API exception queue. The compact repository evidence
is frozen in
[`pivot_feedback_d0_api_only_topic_routing_dry_run_20260729.summary.json`](../../experiments/evidence/results/pivot_feedback_d0_api_only_topic_routing_dry_run_20260729.summary.json).
The v12 ledger authorizes only complete title-bearing candidate-work manifest
construction and local routing before any paid request is considered.

## Interpretation

The dry-run reports routing coverage and how the frozen API diagnostic labels
fall within each route. Those comparisons are engineering diagnostics, not
accuracy, error-rate, calibration, or scientific-effect estimates.
