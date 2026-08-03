# OpenAlex Citation `available_at` v1 Audit Design

Status: `IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN`

Production status: `PRODUCTION_NOT_RUN`

This note defines the temporal-v3-parent-only successor to the frozen citation
chronology v0 contract. It is paired with
`experiments/configs/openalex_g1_citation_available_at_v1.yaml` and
`scripts/audit_openalex_citation_available_at_v2.py`. The earlier v0 config,
runner, runtime tests, design tests, and note remain byte-exact historical
surfaces.

The executable is a closed synthetic diagnostic. It has no production source
root, no production execution mode, no production receipt, and no production
result. It exercises only direct reconstruction of publication time from a
synthetic temporal-v3 parent substitute. Canonical-spine reuse is explicitly
`BLOCKED_CANONICAL_PRODUCTION_RECEIPT_NOT_AVAILABLE`.

## Scientific interpretation boundary

The edge direction is citing work to cited work. The bounded `available_at`
value is the citing work's publication date under the
`BIBLIOGRAPHIC_EVENT_TIME_CONVENTION`. It is not an OpenAlex first-observation
timestamp, evidence that a researcher saw an edge by a historical cutoff, or a
historical graph snapshot.

A future production component may establish exact current-byte pair
accounting, directed-pair deduplication, frozen chronology buckets, open-world
target accounting, and a retrospectively publication-dated citation proxy. It
cannot establish historical content identity, external-target chronology,
fixed-horizon maturation, author or candidate validity, G-1, G0, empirical G1,
an NABM effect, or a scientific effect.

## Exact temporal-v3 parent

There is no v1-or-v2 fallback. The production contract requires these exact
schema and status values:

```text
openalex_g1_main_temporal_validity_outputs.v3
openalex_g1_main_temporal_validity_plan.v3
openalex_g1_main_temporal_validity_checkpoint.v3
openalex_g1_main_temporal_validity_report.v3
openalex_g1_main_temporal_validity_state.v3
PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_FIELD_VALIDITY_V3_WITH_EXACT_UTC_MILLISECONDS_AND_NONBLOCKING_ORDER_PROFILE
```

The completion state remains `COMPLETE_PENDING_RECEIPT`; completion requires a
receipt written last under
`RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED`.

The exact production evidence anchors are:

| Commitment | SHA-256 |
| --- | --- |
| config | `672940908090fedd0b562dd938ad4583f5e8238830fc1ff5b613be9618f033f7` |
| tool | `76e25ffb2bb45977f70ac089db1011b4219d6215e546f16cbe11ce1efdb2cc1d` |
| outputs receipt | `2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe` |
| JSON report | `de98be73fb0365672cec7ee770c40de090ae012d45a0422faebd34c0b9d19200` |
| task fingerprint | `5d86a797481a14e95d08ecbe333bb207ff476bf7dc49886aec1db7b5b366001d` |
| result fingerprint | `a844c624e81567bf89536218b0c488144fe2730f93c53e72193e6cfe4931bfb5` |
| checkpoint tree | `725cf22ce96eca153fa1c9542bd09bd628a4ea1687a2fee3e5610a8cded95392` |
| source tree | `da3fa570a5fb49a36668dffdee008c2fea4d4b6fd5a3f7f17873df78b603b2a0` |

The parent surface contains exactly 497 receipt-bound artifacts, including 493
main checkpoints. Each checkpoint and the aggregate report use the exact
35-key temporal-v3 metric ABI. The 24 publication-related metrics are the
portion relevant to deterministic direct reconstruction. The 11
created/updated metrics remain inherited parent validity and descriptive
evidence; they are not recomputed into an edge timestamp.

Temporal v3 publishes no row-level disposition product or digest. Therefore a
future direct reconstruction may claim only deterministic recomputation under
the frozen v3 policy and the same current bytes. It may not claim direct
row-by-row artifact identity to the temporal parent.

Temporal v1 and v2 schemas, statuses, config/tool paths, config/tool/receipt
hashes, and arbitrary replacement hashes are rejected without fallback.

## Canonical successor binding and blocked reuse

The repository binds the reviewed canonical successor design exactly:

| Commitment | Value |
| --- | --- |
| design config | `experiments/configs/openalex_g1_canonical_work_time_spine_v1.yaml` |
| design config SHA-256 | `2caef77c8b3dc2ffcb0a35920c9e3527096ec3a9f31d20166eec3a3486f4196b` |
| synthetic runner | `scripts/audit_openalex_canonical_work_time_spine_v2.py` |
| synthetic runner SHA-256 | `6a2d1d87dd4a3e7d3c97056fabbe951a1528cfd62d8448acc74ae337f344b7b1` |
| design schema | `openalex_g1_canonical_work_time_spine_design.v2` |
| synthetic status | `PASS_SYNTHETIC_CANONICAL_WORK_TIME_SPINE_TEMPORAL_V3_PARENT_CONTRACT_ONLY` |

That canonical component remains `PRODUCTION_NOT_RUN`. Its synthetic output is
not a production parent, no production receipt exists, and no fake production
receipt hash or invented receipt schema is recorded. Consequently the citation
successor treats canonical reuse as
`BLOCKED_CANONICAL_PRODUCTION_RECEIPT_NOT_AVAILABLE` and
`production_parent_eligible=false`.

Enabling canonical reuse requires another version after a separately reviewed
production runner and a valid receipt exist. The v1 citation contract must not
be silently edited or supplied with an old canonical v0 receipt.

## Closed synthetic executable

The only executable mode is:

```text
exact_temporal_reconstruction_from_current_source
```

Its execution ABI is
`CITATION_DIRECT_RECONSTRUCTION_TEMPORAL_V3_V2`. The JSON fixture must declare
`SYNTHETIC_FIXTURE_ONLY`, deny production input, contain only the direct-mode
temporal payload, and use the exact ordered synthetic parent substitutes for
content, temporal v3, and work-source membership. A synthetic temporal parent
substitute is marked `production_parent_established=false` and
`production_evidence_consumed=false`.

The hard safety caps remain:

- fixture bytes: 1,048,576;
- work-time rows: 10,000;
- reference-pair rows: 10,000.

The CLI exposes no production source-root argument. It requires
`--execute-synthetic`, and its `--mode` choice contains only the direct mode.
Calling the Python API with canonical reuse fails before the fixture is opened.

All synthetic schemas are versioned `.v2`. The report and receipt status is:

```text
PASS_SYNTHETIC_CITATION_EDGE_CHRONOLOGY_TEMPORAL_V3_DIRECT_CONTRACT_ONLY
```

Reports and receipts bind `SYNTHETIC_FIXTURE_ONLY`, `PRODUCTION_NOT_RUN`,
`production_component_status_emitted=false`,
`canonical_production_receipt_consumed=false`, and
`canonical_production_parent_established=false`. They cannot satisfy the future
production receipt ABI.

## Accounting mechanics retained from v0

The successor preserves the exact directed uint64 pair hash, duplicate
multiplicity accounting, seven mutually exclusive chronology buckets,
cross-tab closure, receipt-last publication, terminal no-touch resume, and
receipt-first invalidation behavior from v0. A self-loop remains the highest
precedence bucket. External targets remain open-world and are not assigned a
fabricated target publication time.

The synthetic runner emits only aggregate and checksum control artifacts. It
does not emit detailed production edge partitions or a canonical spine.

## Claim boundary

This is a versioned design and closed synthetic mechanics result. No production
run or result exists. The future production status string in the config is a
contract name, not evidence. Canonical production reuse, cutoff-aware
researcher graphs, candidate eligibility, G-1, G0, empirical G1, and scientific
effects remain pending or not run.
