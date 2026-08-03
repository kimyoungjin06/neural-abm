# OpenAlex Citation `available_at` Audit Design

Status: `IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN`

Production status: `PRODUCTION_NOT_RUN`

This note defines a bounded future audit of directed OpenAlex
`referenced_works` pairs. It is paired with
`experiments/configs/openalex_g1_citation_available_at_v0.yaml`.

The repository now contains a synthetic-fixture-only execution script and
adversarial tests. It cannot open production source roots or promote the full
component status. There is still no production report, receipt, or component
result. A separate production implementation review is required before any
full run can be authorized.

## Synthetic implementation boundary

`scripts/audit_openalex_citation_available_at.py` implements only the closed
`SYNTHETIC_FIXTURE_ONLY` ABI. Its CLI has no source-root or production-execute
argument and requires `--execute-synthetic`. A hard 1 MiB fixture cap and hard
10,000-row caps for both the selected work-time table and reference-pair table
make the full production scope unreachable through this executable.

Both `CITATION_DIRECT_RECONSTRUCTION_V1` and
`CITATION_CANONICAL_SPINE_REUSE_V1` mechanics are exercised, but a fixture
contains exactly one mode's temporal payload. Direct fixtures cannot contain a
spine payload, and reuse fixtures cannot contain direct main-work times. The
runner validates mode-conditioned parent order, the big-endian uint64 pair
SHA-256 bucket ABI, raw multiplicity and exact-pair closures, all seven primary
buckets, the six required cross-tab margins, sensitivity counts, and a
mode-invariant result digest.

Every synthetic work-time row is also checked as a canonical inherited spine
row: `publication_day` is decoded as days since 1970-01-01, its calendar year
must exactly equal `publication_year`, and disposition 0 through 5 must equal
the frozen classifier at the 2026-03-30 snapshot boundary. A fixture cannot
exercise chronology with a self-declared but internally impossible time row.

The synthetic terminal surface has its own schemas and status:

```text
report status  = SYNTHETIC_CITATION_EDGE_CHRONOLOGY_ONLY
receipt schema = openalex_g1_citation_edge_chronology_synthetic_outputs.v1
production     = PRODUCTION_NOT_RUN
```

It is deliberately not
`openalex_g1_citation_edge_chronology_outputs.v1`. Its exact receipt set is the
plan, pre-receipt state, one aggregate checkpoint, JSON report, and Markdown
report. No production pair partitions, aggregate Parquet products, or 4,096
canonical-spine partitions are created. On terminal reentry the runner checks
that exact file set and rehashes every receipt-bound artifact before opening
the fixture payload. Invalid terminal or final-disk state removes the receipt
first and leaves no report/Markdown publication surface.

## Claim boundary first

The current referenced-work table contains a citing work ID and a cited work
ID. It contains no independent timestamp for when OpenAlex first observed,
indexed, or changed that edge. This design therefore defines

```text
available_at(edge) := publication_date(citing_work)
```

only under the named resolution
`BIBLIOGRAPHIC_EVENT_TIME_CONVENTION`.

This convention retrospectively publication-dates an edge that exists in the
current snapshot. It does **not** establish any of the following:

- that OpenAlex observed the edge on the citing work's publication date;
- that the edge was visible to a researcher before a historical cutoff;
- that the current edge list was unchanged at that cutoff; or
- that a historical OpenAlex graph has been reconstructed.

`created_date` begins too late for older publication cohorts, and neither
`created_date` nor `updated_date` is an edge-level first-observation timestamp.
Neither field may be substituted for one.

The proposed future full status retains this limitation in its name:

```text
PASS_CURRENT_SNAPSHOT_CITATION_EDGE_CHRONOLOGY_ACCOUNTING_WITH_BIBLIOGRAPHIC_EVENT_TIME_CONVENTION_LIMITATION
```

That string is a future contract name, not a current result. The only current
status is `IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN`.

## Required parent order and cross-binding

A production implementation must refuse full execution unless these parents
have completed and validate in this order:

1. a full current-content baseline with `COMPLETE` state and
   `PASS_CURRENT_CONTENT_BASELINE_COMPONENT` receipt;
2. a full main temporal audit with
   `PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_VALIDITY_COMPONENT_WITH_FROZEN_EXCLUSIONS`;
   and
3. a precommitted temporal-input mode; and
4. the exact mode-conditioned temporal input: either the recommended canonical
   work-time spine or an exact reconstruction of the same temporal
   dispositions from the current source.

The direct mode's exact parent set is content baseline, main temporal
validity, and the work-source regression artifact. The spine-reuse mode adds
the canonical-spine parent. `all parents` always means this mode-conditioned
set; a direct run must neither require nor silently consume a spine receipt.

The earlier work-source membership result is a required regression parent. It
passed its own bounded accounting component, but its historical byte identity
to a later current-content baseline was not established. A future citation
run must reproduce its anchors on the content-bound current bytes instead of
assuming that earlier and current bytes are historically identical.

The direct-reconstruction mode binds all 3,537 source tasks to the content
plan's same six-field rich identity:

```text
device, inode, mode, bytes, mtime_ns, ctime_ns
```

The spine-reuse mode instead scans all 3,044 referenced-work tasks and binds
the 493 main tasks transitively through a valid spine receipt. The execution
mode and exact parent set are frozen in the task fingerprint and cannot change
on resume. In both modes, the plan, checkpoints, report, receipt, and
`COMPLETE_PENDING_RECEIPT` state bind the same content-result fingerprint and
exact mode-specific input tree. Completion exists only after the receipt is
written last and every receipt-bound artifact is rehashed from disk. Source
identity is verified before and after each scanned file. This remains a first
current-byte baseline, not historical identity evidence.

The content parent uses literal `COMPLETE` plus
`REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST`. The temporal parent uses
`COMPLETE_PENDING_RECEIPT` plus
`RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED`; its exact plan, report,
Markdown, state, all checkpoints, artifact set, task/result fingerprints, and
embedded content commitment must validate. A spine parent must satisfy that
same receipt-last completion ABI, expose all 4,096 partition files and exactly
492,293,206 rows, and bind its content and temporal parents exactly.

The main temporal tool is implemented and synthetic-tested but has no promoted
production result. Consequently this design cannot currently run or promote.
The standalone spine is now specified by the repository-only design at
`openalex-canonical-work-time-spine-audit-design.md` and
`experiments/configs/openalex_g1_canonical_work_time_spine_v0.yaml`. It is
`IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN`: its closed synthetic executable
does not create a production spine receipt or data product and therefore cannot
serve as the production reuse parent required here.

## Exact physical scope by input mode

Without a reusable spine, the direct-reconstruction mode projects two tables.

| Table | Files | Rows | Source bytes | Projection |
| --- | ---: | ---: | ---: | --- |
| `openalex_works_20260330` | 493 | 492,293,207 | 150,062,449,219 | `id`, `publication_date`, `publication_year`, `created_date`, `updated_date` |
| `openalex_works_20260330__referenced_works` | 3,044 | 3,042,536,160 | 37,119,669,321 | `id`, `referenced_works` |

The combined scope is 3,537 files, 3,534,829,367 projected rows, and
187,182,118,540 source bytes (`174.327 GiB`). It is a local CPU/I/O and
external-sort workload. It uses no GPU.

With a validated canonical spine, the spine-reuse mode scans only all 3,044
referenced-work files: 3,042,536,160 rows and 37,119,669,321 source bytes. The
spine's 492,293,206-row artifact tree and its 493-main-task content commitment
replace, rather than silently omit, the direct main scan. Both modes may
eventually establish the same bounded component status, but they are distinct
execution ABIs and cannot share checkpoints.

The 3,044 figure is a **source-file** count, not the complete physical input
count. Spine reuse also consumes all 4,096 spine partitions, the exact spine
manifest/receipt surface, and the receipt-declared spine bytes. Its minimum
physical data-file count is therefore 7,140 before control and checkpoint
artifacts. The plan, preflight, report, and receipt must record source scope,
logical 493+3,044 task lineage, spine data scope, and complete physical input
scope separately. Direct mode is `CITATION_DIRECT_RECONSTRUCTION_V1`; reuse is
`CITATION_CANONICAL_SPINE_REUSE_V1`.

## Direction and exact pair deduplication

The direction is frozen as:

```text
id               = citing work
referenced_works = cited work
edge             = citing -> cited
```

The audit never symmetrizes or reverses this edge.

Raw row accounting and exact-pair accounting are separate layers. Let `R` be
the number of retained reference rows, `U` the number of exact distinct
`(citing work, cited work)` pairs, and `D` duplicate row excess. A full run
must prove:

```text
R = sum of exact-pair multiplicities
R = U + D
duplicate groups = count of pairs with multiplicity greater than one
```

One exact directed pair counts once in every graph and outcome. Raw
multiplicity, duplicate groups, duplicate excess, maximum multiplicity, and
duplicates crossing file boundaries remain visible audit metrics.

Raw-to-unique closure is also mandatory. Raw self-loop rows equal both the sum
of multiplicities over unique self-loop pairs and unique self-loop pairs plus
self-loop duplicate excess. Raw in-snapshot and external-target rows each equal
the sum of pair multiplicities in their membership class; the two unique-pair
membership margins sum to `U`. Thus raw anchors cannot pass independently of
the deduplicated pair surface.

## Current exact anchors, not a new result

The earlier current-snapshot work-source component recorded these anchors:

| Measure | Count |
| --- | ---: |
| Raw referenced rows | 3,042,536,160 |
| Target rows in the main snapshot | 2,904,520,313 |
| Snapshot-external target rows | 138,015,847 |
| All distinct valid target IDs | 135,996,024 |
| Distinct in-snapshot targets | 112,406,279 |
| Distinct external targets | 23,589,745 |
| Raw self-loop rows | 1,072,993 |
| Null, malformed, `W-1`, or `W0` target rows | 0 |

Duplicate source-target edges were `NOT_MEASURED`. These values are exact
regressions that a future content-bound run must reproduce. They are not a
production citation-chronology result.

## W-1, W0, and open-world targets

The exact work-ID grammar remains:

```text
https://openalex.org/W(0|[1-9][0-9]*)
```

The one main `W-1` observation remains under its frozen source-row quarantine.
The retained main `W0` remains `UNRESOLVED` as an entity. The earlier
referenced-work scan found zero `W-1` or `W0` source and target rows. A future
run must independently reproduce exact zero incidence; any drift blocks the
component.

A valid target absent from the main snapshot is an open-world coverage
observation, not a source orphan. Its publication date and chronology are not
available from this root. Such an edge may retain its citing work's
bibliographic event date for accounting, but it cannot enter the primary
chronology-safe graph or a fixed-horizon outcome.

The same snapshot cannot measure missing citing works outside its own source
universe. Reconstructed inbound counts are therefore not total citations.

## Seven exact exclusive pair buckets

After exact directed-pair deduplication, every pair receives exactly one
primary bucket under this frozen first-true precedence:

1. `self_loop`
2. `source_temporal_unusable`
3. `target_external_open_world`
4. `target_temporal_unusable`
5. `negative_lag_target_after_source`
6. `same_day`
7. `positive_lag`

The exact conservation identity is:

```text
unique pairs = sum of all seven exclusive pair buckets
```

Full attribute cross-tabs are also required so that the precedence bucket does
not hide another property. For example, a self-loop remains visible in the
source temporal-disposition cross-tab.

The total classifier uses these exact predicates in the order above:

```text
self_loop:                         citing_id = cited_id
source_temporal_unusable:          source disposition not in {2, 3}
target_external_open_world:        target membership = external
target_temporal_unusable:          target disposition not in {2, 3}
negative_lag_target_after_source:  lag_days < 0
same_day:                          lag_days = 0
positive_lag:                      lag_days > 0

lag_days := citing_publication_day - cited_publication_day
```

Disposition codes 2 and 3 are the only usable spine dispositions. An unusable
source has no `available_at` value. Required cross-tabs are primary bucket by
source disposition, target membership, target disposition or external `NA`,
self-loop flag, signed-lag class or `NA`, and pair-multiplicity class. Every
cross-tab margin must reproduce its corresponding primary or attribute
margin. In particular, chronology-safe pairs equal `same_day + positive_lag`.

The primary chronology-safe set is `same_day + positive_lag`; self-loops are
excluded. The primary relation is:

```text
cited publication date <= citing publication date
```

A negative lag is excluded and measured, not automatically declared an
impossible real citation. Online-first dates, issue dates, versions, and
metadata errors can disagree. The primary tolerance is zero days; separately
frozen 30-day, 365-day, and annual-only sensitivities may be reported, but no
sensitivity may replace the primary result.

After the common prerequisites of exact deduplication, non-self-loop status,
usable source/target dispositions, and in-snapshot target membership, the
primary predicate is `lag_days >= 0`. A `k`-day sensitivity is
`lag_days >= -k`; annual-only sensitivity is
`cited_publication_year <= citing_publication_year`. These supplemental sets
remain separate outputs and never rewrite the seven primary buckets.

## Cutoff-aware graph rule

For cutoff date `C`, a primary edge is bibliographically time-eligible only if
it is:

```text
an exact distinct directed pair
AND not a self-loop
AND source and target have usable inherited temporal dispositions
AND cited publication date <= citing publication date
AND citing publication date <= C
```

For annual decision year `t`, the cutoff is December 31 of `t-1`.

The only permitted name for this surface is a
`retrospectively publication-dated chronology-filtered citation proxy graph`.
It is not a historical OpenAlex snapshot graph. It is also not a collaboration
graph. Researcher-level peer exposure requires the separate author-history and
candidate-eligibility parents.

## Fixed-horizon maturation contract

This design deliberately does not choose an outcome horizon. The primary
horizon, work types, same-day rule, leap-day rule, lag sensitivities, field
normalization, and missing-target rule must be fixed in a separately hashed G0
policy before fixed-horizon labels are produced or inspected.

The proposed generic calendar contract is:

```text
maturity_date(d, H) = the same month and day H years after d
Feb 29 maps to Feb 28 in a non-leap target year
window = [target publication date, maturity_date]
```

For target work `w`, the future outcome counts distinct chronology-safe,
in-snapshot citing works whose publication date falls in that closed interval.
A target is snapshot-mature only when its maturity date is on or before
2026-03-30. It is **temporally mature** for decision year `t` only when its
maturity date is on or before December 31 of `t-1`.

Temporal maturity is not historical data availability. Because edge
first-observation time is unavailable, this rule does not establish that the
edge-derived value was observable by `t-1`. Any later-authorized feature must
be named a `retrospectively backfilled temporally mature citation proxy`; its
historical availability remains `UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT`. A G0
same-day choice may affect a future label policy but may not rewrite this
audit's primary bucket definition.

The bounded label name is
`current_snapshot_internal_citing_work_fixed_horizon_count`. It is not a total
citation count and not a historically observed OpenAlex count.

## Leakage boundary

The core projection and all model inputs prohibit:

- current `cited_by_count`;
- `fwci` and current citation-normalized percentiles;
- current author summary statistics or topics; and
- any outcome value used to select fields or candidates.

`counts_by_year` covers only 2012 through 2026. It cannot be a core parent,
fill missing edges, select the horizon, change primary inclusion, or enter a
model feature. A future use requires a separate contract and remains
`DIAGNOSTIC_LABEL_ONLY`.

## Relation and author dependencies

Generic work-level edge chronology does not logically require work-topic
relation or author history. After the required content and temporal parents,
the following heavy jobs can be logically independent:

- citation-edge chronology;
- work-topic relation; and
- work-side author history after its own relation parent is available.

Topic- or field-stratified citation outcomes require the production
work-topic relation. Researcher-level peer exposure requires both work-topic
relation and author history. Candidate or researcher `t-1` panels may be
materialized only after those parents and the hashed G0 policy join.

Logical independence is not an operational recommendation for concurrent
scans. These are heavy source-disk jobs and should run sequentially to avoid
I/O contention.

## Canonical work-time spine recommendation

The current temporal implementation retains per-file aggregate checkpoints,
not a reusable per-work disposition product. A citation implementation would
otherwise scan and parse the 150,062,449,219-byte main table again.

The recommended optimization is a separate canonical work-time spine after a
full temporal pass, rather than expanding the already bounded temporal v0
surface. Its minimum columns are:

```text
work_id_numeric      uint64, non-null
publication_day      int32, nullable only for BOTH_NULL_EXCLUDED
publication_year     uint16, nullable only for BOTH_NULL_EXCLUDED
temporal_disposition uint8, non-null
```

It must contain all 492,293,206 retained main IDs exactly once, reproduce the
temporal parent's disposition margins, bind the same content tree and temporal
result, and use deterministic numeric partitions with a checksummed artifact
tree.

This spine can eliminate the citation job's repeated main-date scan. It does
not by itself eliminate the author-history main scan because that component
also needs `authors_count` and primary field. A universal work spine is not
recommended now because it would broaden policy and provenance coupling.

## Partition and resource plan

The future implementation uses collision-free numeric partitions and external
sort. Reuse mode may only validate an already completed spine; it may never
build, repair, or publish one. Direct mode may build an
`ephemeral_noncanonical_direct_mode_time_index`, but that index is not a
completion output, cannot be consumed as a canonical spine, and is removed
after successful finalization. The common stages are:

1. validate the pre-existing full spine receipt and tree, or build the direct
   mode's ephemeral noncanonical time index;
2. scan every reference row and validate IDs;
3. partition directed pairs by source numeric ID;
4. join source time, then repartition by target numeric ID;
5. join target membership and time; and
6. exact-sort, deduplicate, classify, and emit chronology-safe events.

The reusable spine ABI is the exact four-field schema above, all 4,096
`work_id_numeric mod 4096` partitions, strictly increasing IDs within each
partition, global exactly-once membership, and receipt-bound schema, manifest,
and data-tree hashes.

Checkpoints bind the task fingerprint, path, six-field source identity,
projected schema and payload hashes, and spool segment path, offset, rows, and
SHA-256. Resume revalidates parents, current sources, and completed spool
segments.

The conservative direct-reconstruction planning envelope is:

- RAM: 64 GiB minimum, 128 GiB preferred;
- durable output: 80-160 GiB;
- temporary SSD: 300 GiB minimum, 500 GiB preferred;
- live free-space preflight before a full run; and
- no concurrent source-disk competition with content freeze.

Spine reuse removes 493 source files and 150,062,449,219 source bytes from this
job, but it does not remove the 3.043-billion-pair external-sort requirement;
RAM and temporary-space preflight therefore remain mandatory.

It also does not make spine I/O free: the reuse preflight must add all 4,096
partition files, 492,293,206 spine rows, receipt-declared partition bytes, and
the exact control/checkpoint/data tree. `3,044` is reported only as the scanned
current-source file count; it must never be reported as the total physical
input count.

Two uint64 IDs for all raw pairs alone require 48,680,578,560 uncompressed
bytes. Dates, classes, multiplicities, and intermediate sorting require the
larger envelope above. This remains CPU/I/O work; GPU availability does not
change the gate order.

## Pilot and full promotion

- `--max-files` is a nonrepresentative parser smoke test.
- An isolated `part-*` file is only a schema or pipeline smoke test.
- A deterministic hash-bucket diagnostic must still consume every input in
  its precommitted mode: all main and referenced-work files for direct
  reconstruction, or all referenced-work files plus the complete spine for
  spine reuse.
- Its selection unit is the exact directed pair key encoded as big-endian
  uint64 citing ID followed by big-endian uint64 cited ID. The bucket is the
  unsigned big-endian value of the first eight SHA-256 digest bytes modulo the
  precommitted bucket count. Count and zero-based index are task-fingerprint
  fields; duplicate rows of one pair always select together.
- No pilot may claim global coverage or receive the full component status.
- Full scope requires either all 3,537 direct-mode files or all 3,044
  reference files plus a complete spine receipt, explicit authorization, the
  exact mode-conditioned parent set, private `0700` output, `0600` files,
  exact accounting closure, a `COMPLETE_PENDING_RECEIPT` state, and a receipt
  written last under `RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED`.

## Planned outputs and privacy

The future control surface is `audit_plan.json`, `run_state.json`,
`citation_edge_chronology_audit.json`,
`citation_edge_chronology_audit.md`, and
`citation_edge_chronology_audit.outputs.json`.

Detailed work IDs, pair-level edges, and chronology-safe partitions remain
operator-local. The citation component never publishes a canonical work-time
spine. Its direct-mode ephemeral time index is temporary, noncanonical, and
absent from the completion receipt. Repository and release surfaces allow only
aggregates and checksums. Every local tree requires a schema hash, row count,
byte count, exact file set, and tree SHA-256. Detailed outputs use `0600` files
inside `0700` directories, and the release inspector must reject their exact
basenames and directory names.

## Residual gates after a future component pass

Even a correct future component pass must retain:

```text
raw_source_id_validity=BLOCKED
w_zero_entity_validity=UNRESOLVED
historical_content_identity=UNVERIFIED
openalex_first_observed_time=UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT
historical_graph_reconstruction=RETROSPECTIVE_PROXY_ONLY
external_target_chronology=UNRESOLVED_OPEN_WORLD
fixed_horizon_outcome_maturation=PENDING_SEPARATE_G0_POLICY
cutoff_aware_t_minus_1_researcher_graph=PENDING
author_entity_validity=PENDING
candidate_eligibility=PENDING
g_minus_1_data_readiness=PENDING
g0_measurement_freeze=PENDING
empirical_g1_signature=NOT_RUN
pass_g1=false
scientific_effect=NOT_ESTABLISHED
```

The component would be an accounting and retrospective chronology-proxy
parent. It would not establish observation time, an eligible researcher
panel, G-1, G0, empirical G1, or a scientific effect.
