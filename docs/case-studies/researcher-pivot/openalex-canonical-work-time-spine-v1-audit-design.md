# OpenAlex Canonical Work-Time Spine Audit Design

Status: `IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN`

Production status: `PRODUCTION_NOT_RUN`

This note defines the versioned successor contract for a future standalone
canonical work-time spine over the current OpenAlex work snapshot. It is
paired with
`experiments/configs/openalex_g1_canonical_work_time_spine_v1.yaml`. Its
repository production contract accepts only the exact full temporal-v3 PASS
parent. The closed synthetic executable keeps the v3 schema, status, and tool
ABI but replaces run-specific hashes and counts with its own exact fixture
anchors under the hard caps below. The earlier v0 spine design, runner, tests,
and note remain byte-preserved historical surfaces.

There is now a synthetic-fixture-only execution script and adversarial test
surface. There is still no production report, receipt, or data product. A
separate production implementation review and explicit authorization are
required before any full OpenAlex run.

## Synthetic executable boundary

`scripts/audit_openalex_canonical_work_time_spine_v2.py` exercises the frozen
16-byte staging record, all 4,096 Parquet partitions, scan and partition
checkpoints, exact writer ABI, terminal resume, and receipt-last failure
invalidation against a closed synthetic content/temporal parent chain.

Synthetic reports and receipts use separate synthetic schema names and the
literal status
`PASS_SYNTHETIC_CANONICAL_WORK_TIME_SPINE_TEMPORAL_V3_PARENT_CONTRACT_ONLY`.
This distinct v2 synthetic schema/status surface prevents an old v1 synthetic
receipt from being relabeled as successor evidence. Reports also bind
`SYNTHETIC_FIXTURE_ONLY`, `PRODUCTION_NOT_RUN`, and
`production_component_pass=false`. The runner refuses the repository config,
the production root ID, any partition-count override, and execution without
an explicit synthetic-fixture flag. Its only source CLI is
`--synthetic-source-root`; `--source-root` is not accepted. Resolved source
path components containing the production root ID are rejected. Independently,
hard synthetic caps of 8 tasks, 10,000 footer rows, 64 MiB of source bytes, and
10,000 retained rows are checked first against the temporal-parent control
metadata and then against each live source identity and Parquet footer before
any column payload scan. A highly compressed file cannot hide an over-cap row
count behind parent metadata. Relabeling a full source as synthetic therefore
cannot bypass the authorization boundary.

The synthetic receipt surface contains exactly
`6 + 2*N + 2*4096` artifacts for `N` synthetic input tasks. The production
invariant remains separately frozen at `N=493`, or 9,184 artifacts. Creating
all 4,096 synthetic partitions does not satisfy the 492,293,206-row production
invariant and cannot receive the production component status.

## Decision and claim boundary

The spine remains a standalone component immediately after the full main
temporal audit, not an added output of temporal v3. Fusing the two scans in a
future version could save one main-table scan, but changing the completed v3
audit would reopen its audited tool/config hashes, expand its claim and receipt
surface, and couple a narrow validation gate to a large reusable local data
product.

A future spine pass may establish exact current-byte work-ID accounting,
deterministic publication-time dispositions, numeric partitions, and a
checksummed row-level data product. It cannot establish historical content
identity, validate `W0` as an entity, or pass citation, author, candidate,
G-1, G0, empirical G1, or scientific-effect gates.

The future production status is deliberately bounded:

```text
PASS_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_COMPONENT_WITH_TEMPORAL_V3_PARENT
```

This is a future contract name, not a result. The current state remains
`IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN`; production remains
`PRODUCTION_NOT_RUN`.

## Required parent order and completion ABI

The production implementation must require this order:

1. current-content baseline;
2. full main temporal validity audit; and
3. standalone canonical work-time spine.

The content parent must have a literal `COMPLETE` state and a valid
`PASS_CURRENT_CONTENT_BASELINE_COMPONENT` receipt. Its fixed scope is 21,019
files and 988,404,537,807 bytes.

The temporal parent must have status
`PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_FIELD_VALIDITY_V3_WITH_EXACT_UTC_MILLISECONDS_AND_NONBLOCKING_ORDER_PROFILE`
under config SHA-256
`672940908090fedd0b562dd938ad4583f5e8238830fc1ff5b613be9618f033f7`.
The exact frozen v3 evidence anchors are:

| Commitment | SHA-256 |
| --- | --- |
| temporal tool | `76e25ffb2bb45977f70ac089db1011b4219d6215e546f16cbe11ce1efdb2cc1d` |
| outputs receipt | `2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe` |
| JSON report | `de98be73fb0365672cec7ee770c40de090ae012d45a0422faebd34c0b9d19200` |
| task fingerprint | `5d86a797481a14e95d08ecbe333bb207ff476bf7dc49886aec1db7b5b366001d` |
| result fingerprint | `a844c624e81567bf89536218b0c488144fe2730f93c53e72193e6cfe4931bfb5` |
| checkpoint tree | `725cf22ce96eca153fa1c9542bd09bd628a4ea1687a2fee3e5610a8cded95392` |
| source tree | `da3fa570a5fb49a36668dffdee008c2fea4d4b6fd5a3f7f17873df78b603b2a0` |

Temporal v3 uses a receipt-last completion ABI. Its final state remains
`COMPLETE_PENDING_RECEIPT`, and completion is valid only when the outputs
receipt exists and satisfies
`RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED`. A spine validator must not
incorrectly demand a literal temporal `COMPLETE` state.

The temporal receipt's plan, report, Markdown, state, all 493 checkpoints,
result fingerprint, task fingerprint, checkpoint tree, and source tree must be
validated. Its embedded content commitment must equal the independently
validated content receipt.

The parent ABI is exact rather than schema-name-only. For the repository
production contract, the implementation binds the frozen plan, checkpoint,
report, outputs, and state v3 schema names and all eight tool, config, receipt,
report, task, result, checkpoint-tree, and source-tree digests above. The
closed synthetic path substitutes exact fixture-specific run anchors only
under its explicit cap policy; it cannot emit a production status. Temporal v1
and v2 schemas, statuses, config/tool paths and hashes, and receipts are
rejected without fallback.
The temporal receipt has exactly seven keys and exactly 497 artifacts: the
plan, state, report, Markdown, and the 493 plan-derived checkpoint paths. Every
artifact key must equal its relative path and its byte count and SHA-256 must
match disk. The plan task fingerprint, report result fingerprint, each
checkpoint self/metrics hash, checkpoint tree, and source tree are recomputed;
merely copying their published digest strings is insufficient.
Checkpoint basenames are exactly the SHA-256 of the UTF-8 relative source path.
The report fingerprint payload is the exact temporal-v3 field set from
`status` through `claim_boundary`: it includes `pilot_local_contract_clean`,
accounting, the nonblocking `created_updated_relation_diagnostic`, checkpoint,
source, and content-parent evidence. It does not accept the removed v1
`exception_accounting` surface.

## Six-field and nine-field source cross-binding

Every selected main task is bound to both parent identity surfaces.

The content plan exposes this common six-field identity:

```text
device, inode, mode, bytes, mtime_ns, ctime_ns
```

The temporal plan records the fuller nine-field identity:

```text
device, inode, mode, uid, gid, links, bytes, mtime_ns, ctime_ns
```

Content `mode` is permission bits while temporal `mode` is the full stat mode;
the exact normalization is:

```text
content.mode == stat.S_IMODE(temporal.mode) == (temporal.mode & 0o7777)
```

Both parent task-path sets must be unique and exactly equal. The other five
content fields must equal the corresponding temporal fields directly, while
the current nine-field stat must equal the temporal identity before and after
scan. All 493 tasks must match both parents. Parent receipts and source
identities are revalidated at finalization.

This cross-binding establishes a first current-byte baseline only. It does not
establish historical byte identity.

## Exact source scope

The component scans only the main table:

| Files | Raw rows | Declared source bytes | Projected columns |
| ---: | ---: | ---: | --- |
| 493 | 492,293,207 | 150,062,449,219 | `id`, `publication_date`, `publication_year` |

`created_date` and `updated_date` are not repeated in the spine. Their
lexical/null/snapshot-bound validity remains inherited from the temporal
parent. Their relative ordering is a nonblocking descriptive profile and is not
used by the spine.

The full v3 parent parsed all 492,293,206 retained metadata pairs at exact UTC
millisecond precision. Created/updated nulls, parse failures, and future field
values are all zero. The receipt-bound order profile is 492,293,203
`created <= updated` rows and three `created > updated` rows. Those three rows
are descriptive and nonblocking; they are not exceptions to a validity rule.

## Minimal four-column schema

The spine uses exactly four fields:

| Field | Physical type | Nullability |
| --- | --- | --- |
| `work_id_numeric` | `uint64` | non-null |
| `publication_day` | `int32`, days since 1970-01-01 | null only for both-null |
| `publication_year` | `uint16` | null only for both-null |
| `temporal_disposition` | `uint8` | non-null |

The raw OpenAlex URL is not stored because the exact grammar-valid URL can be
reconstructed from its unsigned numeric suffix. The six frozen disposition
codes are:

```text
0 BOTH_NULL_EXCLUDED
1 VALID_PRE_1900_EXCLUDED
2 USABLE_1900_BEFORE_SNAPSHOT_YEAR
3 USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT
4 VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED
5 VALID_POST_SNAPSHOT_YEAR_EXCLUDED
```

Consumers must use `temporal_disposition` for eligibility. A non-null date is
not sufficient because pre-1900 and future rows remain materialized for exact
accounting.

Classification is an exact ordered ABI. The frozen `W-1` row is removed first,
matching temporal v3's retained-row masks. Null or unexpected malformed IDs
among the retained rows, and invalid publication rows, are then rejected
before any staging shard is published; they receive no fallback disposition.
A retained row is classified as both-null only when both publication fields
are null. Every other row must have a strictly parsed date, an integral year
equal to the year derived from that date, and exactly one of these predicates:

```text
day < 1900-01-01                                      -> code 1
day >= 1900-01-01 and derived_year < 2026             -> code 2
derived_year == 2026 and day <= 2026-03-30            -> code 3
derived_year == 2026 and day > 2026-03-30             -> code 4
derived_year > 2026                                   -> code 5
```

Exactly-one-null, parse-error, invalid/non-integral-year, and year-mismatch
counts must all be zero before staging or partition publication. Every temporal
checkpoint must expose the exact 35-key v3 metric set. The spine recomputes and
compares the frozen 24-key identifier/publication subset plus publication-year
extrema. The 11 created/updated keys remain v3 parent validity and descriptive
order evidence; the three-column spine scan does not claim to recompute them.

## Exact ID accounting

The full component must prove:

```text
492,293,207 raw rows
= 1 frozen W-1 quarantine row
+ 492,293,206 retained spine rows
```

It must also prove:

```text
spine rows = distinct numeric IDs = 492,293,206
duplicate ID groups = 0
duplicate row excess = 0
minimum numeric ID = 0
maximum numeric ID = 7,143,276,261
W0 rows = 1
W-1 output rows = 0
```

`W0` remains a retained grammar-valid numeric ID with
`w_zero_entity_validity=UNRESOLVED`. Its exact temporal-disposition cross-tab
is reported, but its category is not pre-assumed and cannot change entity
validity.

## Publication disposition accounting

The following counts are already frozen by the temporal policy and aggregate
regressions:

| Disposition | Rows |
| --- | ---: |
| `BOTH_NULL_EXCLUDED` | 27,565,532 |
| `VALID_PRE_1900_EXCLUDED` | 6,692,869 |
| `USABLE_1900_BEFORE_SNAPSHOT_YEAR` | 446,514,060 |
| Snapshot-year dispositions combined | 11,516,735 |
| `VALID_POST_SNAPSHOT_YEAR_EXCLUDED` | 4,010 |

The exact full temporal-v3 pass freezes the split at 11,509,999
`USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT` rows and 6,736
`VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED` rows. The two counts sum to
11,516,735. These anchors come only from the receipt-bound full v3 report, not
from a pilot or aggregate-only inference.

All six disposition counts must sum to 492,293,206. Non-null publication years
retain the exact minimum 1000 and maximum 2050. For every non-null row, the
stored year must equal the year derived from the stored day.

Exactly-one-null rows, date parse errors, invalid/non-integral years, and
date/year mismatches must all remain zero.

## Row-level temporal-parent limitation

Temporal v3 publishes per-file metrics and projected-payload commitments, but
it does not publish a row-level disposition artifact or a row-level
disposition digest. Therefore a standalone spine cannot directly prove
byte-for-byte row-disposition equality to a temporal-parent artifact.

The spine instead requires:

- the same content tree and current source identities;
- the frozen temporal config, policy, plan, result, and checkpoint tree;
- exact per-file and global publication-metric equality;
- its own three-column projected-payload hash; and
- its own row-level disposition-payload hash.

The spine becomes the first row-level disposition commitment. Its allowed
claim is deterministic recomputation bound to the temporal-parent policy and
current bytes. It may not claim
`row_level_artifact_identity_to_temporal_v3_output`.

The independent row commitment uses a fixed 16-byte little-endian record:

| Offset | Bytes | Field | Encoding |
| ---: | ---: | --- | --- |
| 0 | 8 | `work_id_numeric` | unsigned 64-bit little-endian |
| 8 | 4 | `publication_day` | signed 32-bit little-endian |
| 12 | 2 | `publication_year` | unsigned 16-bit little-endian |
| 14 | 1 | `temporal_disposition` | unsigned 8-bit |
| 15 | 1 | reserved padding | always zero |

For code 0, day is the exact sentinel `INT32_MIN` and year is zero; those
sentinels are forbidden for non-null rows. One staging shard exists for each
input file at `staging/main/<sha256(relative-path)>.bin`. Canonical row hashes
use `LENGTH_FRAMED_SHA256_V1`: a length-framed domain, the little-endian row
count, then every 16-byte record preceded by its little-endian length. This
makes the digest independent of I/O chunking. The three-column source
projection uses temporal-style length-framed serialized Arrow record batches
with batch size and runtime fingerprint frozen in the plan.

## Deterministic 4,096-partition product

Partitioning is exact numeric modulo, not hashing:

```text
partition_id = work_id_numeric mod 4096
```

The output is exactly `spine/part-0000.parquet` through
`spine/part-4095.parquet`. Every file exists; an empty partition is represented
by a valid empty Parquet file. IDs are strictly increasing within each
partition and appear globally exactly once.

Modulo is a total single-valued assignment, but different IDs normally share
a partition; hash-collision language does not apply. Empty-partition minimum
and maximum IDs are null and strict sortedness is vacuously true.

The Arrow schema is exactly non-null `uint64`, nullable `int32`, nullable
`uint16`, and non-null `uint8` in column order. The corresponding Parquet
physical/logical pairs are `INT64/UINT_64`, `INT32/NONE`, `INT32/UINT_16`, and
`INT32/UINT_8`, with required/optional repetition matching Arrow nullability.
The writer ABI is Parquet 2.6, no compression, no dictionary, no statistics,
262,144-row groups and write batches, data-page v1, no page index or checksum,
no byte-stream split, no stored Arrow schema, no sorting-column metadata, and
combined input chunks. Writes are atomic. PyArrow, Parquet C++, Python, and
machine runtime fields are bound to the plan, task fingerprint, and receipt;
resume requires exact runtime equality because byte identity can remain
runtime-dependent.

## Two-stage checkpoints and resume

The scan stage processes inputs in lexical path order. Every input file emits
one staging shard and one checkpoint binding:

- spine and temporal-parent task fingerprints;
- temporal task and checkpoint hashes;
- source identity, footer rows, schema, and projected-payload hash;
- per-file disposition metrics and row-level disposition hash; and
- staging path, rows, bytes, and SHA-256.

The partition stage binds the scan-checkpoint tree and staging tree. Every
partition checkpoint records partition ID and formula, output path, rows,
minimum and maximum ID, schema, bytes, SHA-256, strict sortedness, disposition
and null counts, and two canonical row digests. The first digest is computed
from verified staging records selected by exact modulo and sorted by numeric
ID before Parquet writing. The second is computed after decoding the published
Parquet and re-encoding every row to the 16-byte ABI. Counts and digests must
match.

The following conservation chain is exact:

```text
492,293,207 raw
= 1 W-1 + 492,293,206 scan-retained
= 1 W-1 + sum(staging shard rows)
= 1 W-1 + sum(partition-input rows)
= 1 W-1 + sum(4,096 Parquet rows)
= 1 W-1 + 492,293,206 decoded final rows
```

Every verified staging record is assigned once, its ID satisfies the partition
formula, all per-partition pre/post digests match, and decoded disposition,
nullability, year/day, W0, extrema, and uniqueness totals equal scan and
temporal-parent accounting. This cross-stage bridge prevents a same-row-count
but value-different Parquet product from passing.

Resume requires the exact same plan. It revalidates config, tool, all parents,
current source identities, current three-column payloads, reused staging
shards, and existing partition outputs. Partition count or execution-mode
changes cannot resume.

A valid completed receipt is terminal. `--resume` must validate the exact
published artifact set and every receipt-bound byte/hash, then stop before any
source payload is opened. It may never rewrite an already valid completion.

Source and output symlinks are rejected. The output is outside the source and
repository, uses an exclusive lock, has `0700` directories and `0600` files,
and rejects undeclared entries.

## Receipt-last completion

The future control surface contains:

- `audit_plan.json` and `run_state.json`;
- scan and partition checkpoints;
- `staging_manifest.json`;
- `work_time_spine_manifest.json`;
- `work_time_spine_audit.json` and Markdown; and
- `work_time_spine_audit.outputs.json`.

The exact full tree additionally contains 493 files under
`staging/main`, 493 scan checkpoints under `checkpoints/scan/main`, 4,096
partition checkpoints under `checkpoints/partition`, and 4,096 Parquet files
under `spine`. Excluding `audit.lock` and the receipt itself, the receipt binds
exactly 9,184 regular artifacts. Staging is deliberately retained for verified
resume; it is private operator-local data, not a release surface.
The release inspector rejects the `staging` and `spine` directory components,
both scan and partition checkpoint subtrees, and both spine manifest basenames
wherever they appear in an sdist.

Before the final receipt, state is `COMPLETE_PENDING_RECEIPT`. The outputs
receipt is the last write. Final validation rehashes all control and data
artifacts. A failure removes the receipt and final report surfaces while
preserving valid resumable checkpoints.

Failure invalidation is ordered: remove the authoritative receipt first,
write `FINALIZATION_FAILED_NO_VALID_RECEIPT` without a result fingerprint,
then remove report and Markdown. Cleanup failures are aggregated and raised,
not swallowed. Receipt deletion, failure-state publication, and descriptive
surface removal are independent invalidation attempts so a failed finalization
cannot remain promotable.

The result fingerprint binds parents, task and source trees, scan/staging/
partition trees, the spine data tree, exact accounting, gate states, and claim
boundary.

## Resource boundary

At 16 fixed staging bytes per retained row, the staging payload anchor is:

```text
492,293,206 * 16 = 7,876,691,296 bytes = 7.336 GiB
```

The planning envelope is:

- RAM: 16 GiB minimum, 32 GiB preferred;
- durable output: 12-20 GiB;
- temporary SSD: 64 GiB minimum, 96 GiB preferred;
- live disk and RAM preflight with at least a 1.5 safety multiplier; and
- no source-disk competition with content freeze or another heavy scan.

The full disk preflight is not a fixed free-space guess. It computes remaining
staging bytes, a bounded partition spool, Parquet upper bytes (32 bytes per row
plus 128 KiB per partition), remaining control artifacts, and the largest next
atomic-publish artifact. It applies the 1.5 multiplier to that sum and then
adds an 8 GiB reserve. Resume may subtract an existing artifact only after its
path, size, SHA-256, and internal checkpoint contract are revalidated. RAM is
the maximum of the planned projected batch and partition-sort chunk bounds,
plus reserve; both bounds are frozen into the plan.

This is local CPU/I/O and external-sort work. It uses no GPU.

## Pilot and full promotion

- `--max-files` is a parser and staging smoke test only.
- A deterministic ID-bucket diagnostic must scan all 493 files to complete
  selected buckets, but cannot establish global coverage or promote.
- Full scope requires explicit authorization, all 493 files, every raw and
  retained row, all 4,096 partitions, all full parent receipts, exact
  accounting closure, and a valid completion receipt.
- A partial or pilot spine cannot be consumed by a full citation run.

## Consumer scan savings

A full citation chronology run in `canonical_work_time_spine_reuse` mode must
open zero main source files and scan zero main source bytes. Its 493-task main
lineage is bound transitively through the complete spine receipt. A synthetic
consumer test must fail immediately if that mode attempts any main-source
open.

```text
main_source_files_opened_by_citation_component=0
main_source_bytes_scanned_by_citation_component=0
```

This removes the citation component's 150,062,449,219-byte main scope and also
removes repeated work-time scans for later citation maturation.

It does not eliminate:

- content freeze or the temporal parent scan;
- work-topic relation;
- the author-history main scan; or
- taxonomy and other relation-specific scans.

Author history can avoid publication-date parsing, but it still needs
`authors_count` and primary field. Expanding this spine to include those columns would
create a broader work-metadata product and is explicitly out of scope.

## Privacy and residual gates

Individual work IDs, staging shards, and spine partitions remain
operator-local `0600` data under `0700` directories. Repository and release
surfaces allow only aggregates, schemas, and checksums. The release inspector
must reject detailed spine data.

Even a future full spine pass retains:

```text
historical_content_identity=UNVERIFIED
raw_source_id_validity=BLOCKED
w_zero_entity_validity=UNRESOLVED
row_level_temporal_parent_artifact_identity=NOT_AVAILABLE_TEMPORAL_V3
citation_edge_chronology=PENDING_NOT_IN_THIS_COMPONENT
openalex_edge_observation_time=PENDING_UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT
fixed_horizon_outcome_maturation=PENDING_SEPARATE_G0_POLICY
author_entity_validity=PENDING
candidate_eligibility=PENDING
cutoff_aware_t_minus_1_researcher_graph=PENDING
g_minus_1_data_readiness=PENDING
g0_measurement_freeze=PENDING
empirical_g1_signature=NOT_RUN
pass_g1=false
scientific_effect=NOT_ESTABLISHED
```

Residual production-implementation risks are the second
150,062,449,219-byte main-source scope scan, absence of a temporal-v3 row-level
digest for direct comparison, writer-runtime dependence of Parquet bytes,
current-snapshot-only retrospection, historical byte-identity limits, and all
downstream citation, author, maturation, and candidate gates. Synthetic
contract success does not reduce those evidence gaps.
