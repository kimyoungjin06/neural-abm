# OpenAlex Work-Side Author-History v1 Audit Design

Status: `DESIGN_ONLY`

Production status: `PRODUCTION_NOT_RUN`

This is the temporal-v3-only successor to the v0 author-history design. Its
machine-readable contract is
`experiments/configs/openalex_g1_work_side_author_history_v1.yaml`, with schema
`openalex_g1_work_side_author_history_design.v2`. The successor is deliberately
outside the public researcher-pivot subtree: it does not silently change the
published case-study allowlist or rebind the byte-exact v0 record.

No executable, synthetic result, production result, or receipt exists in this
tranche. The present production output ABI is
`NOT_IMPLEMENTED_REQUIRES_SEPARATE_REVIEW`. This note records design
requirements, not completed computation and not a component pass.

## Question and bounded unit

The future component would ask how publication histories attached to author
identifiers in the current OpenAlex work-side authorship table can be counted
exactly enough to support later candidate and `t-1` analyses. Its unit is the
current-snapshot OpenAlex author-ID token: **tokens, not persons**. A token
history is an observed retrospective publication span, not a verified career.

A future bounded pass could establish exact author-work-token accounting,
duplicate-pair conservation, annual unique-work counts, token-span summaries,
and frozen sensitivity grids. It could not establish natural-person identity,
correct ID splits or merges, historical observability, candidate eligibility,
G-1, G0, empirical G1, or a scientific effect.

## Exact parent order and current stop

A future author runner must bind and revalidate this order:

1. the current-content baseline;
2. the full main temporal-v3 receipt;
3. a real production canonical work-time spine bound to the same content and
   temporal commitments;
4. the author-history task.

The content parent is fixed to its v0 config and tool, plan SHA, task
fingerprint, 11-key receipt ABI, and exact three-artifact set. The author design
has not itself reopened or verified that receipt, so the future runner must do
so again.

The canonical design binding is exact:

- config: `openalex_g1_canonical_work_time_spine_v1.yaml`, schema v2;
- synthetic tool: `audit_openalex_canonical_work_time_spine_v2.py`;
- synthetic status:
  `PASS_SYNTHETIC_CANONICAL_WORK_TIME_SPINE_TEMPORAL_V3_PARENT_CONTRACT_ONLY`;
- production result and production receipt: absent;
- production-parent eligibility: false.

Therefore author production is currently
`BLOCKED_CANONICAL_PRODUCTION_RECEIPT_NOT_AVAILABLE`. The design does not invent
a canonical production receipt schema or hash. Both the canonical production
surface and the author production surface require new versions and review.

## Temporal v3: aggregate evidence, not a reusable row product

The exact parent is the full temporal-v3 result with status
`PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_FIELD_VALIDITY_V3_WITH_EXACT_UTC_MILLISECONDS_AND_NONBLOCKING_ORDER_PROFILE`.
Its config, tool, receipt, report, task fingerprint, result fingerprint,
checkpoint-tree hash, and source-tree hash are all bound as eight exact SHA-256
anchors. Temporal v1 and v2 schemas, paths, statuses, and hashes are rejected;
there is no fallback.

Temporal v3 audited all declared main-table rows. What it persisted for reuse
is aggregate/checkpoint evidence, not a reusable row-level disposition product.
Its receipt has an exact seven-key ABI, four fixed control/report artifacts,
493 per-file checkpoints, and **497** receipt artifacts in total. Every receipt
path, byte length, and SHA-256 must be revalidated.

The checkpoint metric ABI has 35 keys. The first **24** are the identifier,
accounting, and publication-side keys that a future author/canonical
publication scan may recompute. The final **11** `created`/`updated` keys are
inherited-only from the exact temporal-v3 receipt; an author runner may not
recompute or reinterpret them.

The canonical spine remains the only allowed row-level time input. Source
`publication_date` reparsing is forbidden. History usability comes only from
canonical temporal disposition codes 2 and 3, and the historical decision
cutoff is evaluated as:

```text
publication_day <= days_since_epoch(t_minus_1_december_31)
```

This avoids accidentally making a fresh source date parse into a substitute
temporal parent.

## Author identity and the incomplete master

The separate 2026-03-30 author delta contains 44,099,961 rows, while the
work-side table contains 113,748,578 distinct grammar-valid author tokens. The
ratio is **38.76968115%**, but its explicit semantics are
`UNVERIFIED_ROW_COUNT_UPPER_BOUND`. It assumes every delta row is unique and
overlaps a different work-side token. Actual coverage is `NOT_MEASURED`.

The delta may later support a separately lineage-bound linked-subset
diagnostic. It cannot be a full author parent and cannot validate person
entities. ORCID checksum categories and ORCID/name one-to-many or many-to-one
patterns remain anomaly proxies, not split/merge truth or entity-resolution
ground truth. Names, ORCIDs, positions, and current primary fields are not
past-observable model features.

## Event counting, cutoffs, caps, and thresholds

The exact publication unit is the distinct
`(author_numeric_id, work_numeric_id)` pair. Duplicate rows count as one
publication while retaining the union of diagnostic sets; selecting a
representative duplicate row is forbidden. Full-snapshot token summaries are
audit-only and cannot enter candidate selection, model features, `t-1` graphs,
or exposure construction.

Sensitivity predicates are frozen rather than inferred from labels:

- cap 10, 50, or 100 includes a pair iff `declared_authors_count <= cap`;
- `ALL` includes every history-usable unique author-work pair;
- a hyper-prolific threshold cell uses
  `maximum_annual_unique_work_count > threshold` for thresholds 50, 100, 250,
  500, and 1000;
- every cell reduces hidden records from the same view and same cap; cross-cap
  source substitution is forbidden.

Thresholds are diagnostics. They cannot silently become candidate-exclusion
rules.

## Future closure evidence is not a new parent

The canonical spine is still the future runner's row-level time input. A new
production version must separately choose exactly one way to prove or reopen
author-work event closure at finalization and terminal reentry:

1. a **receipt-bound author-work event artifact**; or
2. a **same-parent full rescan**.

These are closure-evidence strategies, not alternate temporal parents. The
choice and its parent commitments must enter the task fingerprint and cannot
change on resume.

If the artifact strategy is selected, the new version must bind the exact
retained-main-to-canonical-spine one-row join closure, event row count, event
schema digest, event content-tree or equivalent digest, canonical result
fingerprint, canonical spine-data-tree SHA, and an exact terminal revalidation
method. If the rescan strategy is selected, it must revalidate the same parent
receipts and source identities, reproduce the exact join closure, and
recompute the event row/schema digests. None of those production schemas,
hashes, or methods is frozen by v1.

## Hidden reductions and terminal safety

Hidden cross-record closures are future-runner obligations, not claims about
work performed here. Before aggregate digests and metric hashes, a future
runner must verify within each base/positive-author view that:

- author-year and author-token keys are unique;
- the two surfaces contain the same author keyset;
- each token summary exactly reduces the same author's yearly records;
- positive-ID records are the stable `author_numeric_id > 0` projection;
- each positive capped array is the same-cap projection of its base array;
- each cap count and threshold cell reduces same-view, same-cap hidden records.

The exact per-token reductions are minimum year, maximum year, distinct active
year count, inclusive span, inactive gap, summed annual unique-work count, and
maximum annual unique-work count.

A production version also needs **terminal resume revalidation**: it must
rehash the receipt and every bound artifact, revalidate all parent receipts and
cross-bindings, and revalidate plan, parents, source identity, and completed
checkpoints before continuing from failure. It may not claim an independent
hidden-array reduction on terminal reentry unless those arrays are reopened or
their receipt-bound proof is revalidated. V1 implements none of this runtime
behavior.

## Siblings and downstream gates

The work-topic relation is a parallel sibling, not an author-history parent. It
joins only at later field-coverage, candidate, or `t-1` gates.

The temporal-v3-only citation successor is also not a parent. The citation `available_at`
surface is a **downstream sibling blocker** for G-1/G0. Its design config
and synthetic tool are bound exactly, but its production receipt is absent,
historical edge observation time is not established, and outcome maturation is
not defined. The legacy citation v0 config/tool/schema/status cannot be
substituted.

Accordingly, G-1 and G0 remain `PENDING`; citation chronology is fail-closed as
`BLOCKED_CITATION_PRODUCTION_RECEIPT_NOT_AVAILABLE`; empirical G1 remains `NOT_RUN`;
`pass_g1` remains false; and the scientific effect remains
`NOT_ESTABLISHED`.

## Current claim ledger

| Item | Current state |
|---|---|
| Author-history component | `DESIGN_ONLY` |
| Production execution | unauthorized / `PRODUCTION_NOT_RUN` |
| Temporal v3 | exact aggregate/checkpoint parent evidence bound |
| Reusable temporal row product | not persisted |
| Canonical production spine | receipt absent; author production blocked |
| Author entity validity | unresolved, work-side token only |
| Author-master coverage | actual `NOT_MEASURED`; upper bound unverified |
| Historical content identity | `UNVERIFIED` |
| Token/field past observability | current-snapshot retrospective proxy |
| ORCID validation | diagnostic, not ground truth |
| Candidate eligibility and field selection | `PENDING` |
| Citation chronology | production receipt blocked |
| Outcome maturation | `PENDING` / not defined |
| G-1 / G0 | `PENDING` |
| Empirical G1 | `NOT_RUN` |
| Scientific/NABM effect | `NOT_ESTABLISHED` |

The long hypothetical component-pass string in the YAML is only a future
bounded status name. It is explicitly not a current result and could promote
only token-linked accounting—not author entities, candidates, causal effects,
or the paper's empirical signature.
