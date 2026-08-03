# OpenAlex Main-ID Quarantine Amendment

Amendment version: 1.1

Date: 2026-07-17

Status: work-source quarantine accounting and membership passed across all 23
tables; no filtered analysis view was materialized, raw source validity remains
`BLOCKED`, `W0` entity validity is `UNRESOLVED`, parent content identity is
`UNVERIFIED`, G-1 data readiness is `PENDING`, and empirical G1 is `NOT_RUN`.

## 1. Why this is an amendment

The original frozen recount contract required 492,293,207 body rows, zero null
IDs, zero invalid IDs, 492,293,207 exact distinct IDs, and zero duplicate
excess. The first full data-page scan did not satisfy that contract. It found
exactly one malformed value, `https://openalex.org/W-1`, and correctly returned
`BLOCKED` without sorting or promoting the main-ID gate.

That failure is preserved in the
[blocked evidence](../../../experiments/evidence/results/openalex_g1_main_id_recount_20260716.blocked.summary.md).
It is not rewritten as a pass. A source-side key-health report generated on
2026-04-27 had already recorded the same value and count, before the present
empirical protocol. The quarantine rule is therefore a versioned data-quality
amendment made before outcome construction or effect estimation, not an
outcome-dependent outlier deletion.

## 2. Immutable source and derived view

The source root is not modified. Its main table has 492,293,207 body rows and
remains `BLOCKED_RAW_ID_VALIDITY` because it contains `W-1`.

The derived view uses this published grammar:

```text
https://openalex.org/W(0|[1-9][0-9]*)
```

It accepts the exact prefix followed by a decimal-normal-form unsigned integer,
including `W0`, and rejects signs, leading zeros, empty suffixes, overflow, and
all other forms. The [OpenAlex ID documentation](https://developers.openalex.org/guides/key-concepts)
describes IDs as a letter prefix plus a numeric key, but does not state in that
description that the numeric part must be positive. For that reason this
repository calls the retained set `grammar-valid`, not canonical-valid or
entity-valid.

Only this exact malformed observation is quarantined:

| Field | Frozen value |
| --- | --- |
| Raw ID | `https://openalex.org/W-1` |
| Count | 1 |
| Relative file | `openalex_works_20260330/part-384.parquet` |
| Row group | 13 |
| File row | 428,949 |

Any different value, count, file, row group, or file row blocks the amended
recount. Malformed rows are removed before the equality spool is written; no
reserved numeric sentinel is used.

The amended full recount reproduced the conservation identity

```text
492,293,207 body rows
= 492,293,206 grammar-valid retained rows
+ 1 frozen malformed row
+ 0 null rows
+ 0 unexpected malformed rows
```

and found 492,293,206 exact distinct retained IDs, zero duplicate excess, zero
duplicate groups, and maximum multiplicity one. The compact evidence is
[here](../../../experiments/evidence/results/openalex_g1_grammar_valid_main_id_recount_20260716.summary.md).

## 3. W0 is retained but unresolved

The root contains one `https://openalex.org/W0` main row. It is at
`part-322.parquet`, row group 10, file row 340,597, and is retained as the
actual numeric key 0. Its entity validity is not established: its MAG value is
`0`, its sole authorship row has a null author ID, and a dated API probe
returned 404. The API result does not by itself authorize exclusion because
OpenAlex documents that entity IDs can be
[merged and redirected](https://developers.openalex.org/guides/get).

The active decision is therefore:

- `WARN_ZERO_ID_UNRESOLVED`;
- retain `W0` in the grammar-valid view;
- do not describe it as entity-valid;
- use a positive-ID exclusion only as a separately reported sensitivity; and
- require source-owner or lineage evidence before changing this rule.

The registered sensitivity was subsequently run on 2026-07-18. Excluding W0
reduces the total and null-primary-field margins by one while leaving all 26
non-null field counts unchanged. Its status is
`PASS_W0_POSITIVE_ID_EXCLUSION_SENSITIVITY_COMPONENT`; this is not an entity
validity decision, and `W0` remains retained and `UNRESOLVED` in the primary
grammar-valid view.

## 4. Relation incidence and propagation rule

The targeted full-root incidence audit evaluated source `id` in all 23 tables
and the work-target fields in the referenced-works and related-works tables.
It found:

| Sentinel | Source rows | Source files | Declared work-target rows |
| --- | ---: | ---: | ---: |
| `W-1` | 21 | 6 | 0 |
| `W0` | 6 | 6 | 0 |

The 21 `W-1` source rows are the main row, one authorship, one best-OA host
lineage row, one abstract-exception row, 16 location rows, and one
primary-location host-lineage row. The exact incidence evidence is
[here](../../../experiments/evidence/results/openalex_g1_sentinel_incidence_20260716.summary.md).

The targeted audit fixed the exclusion incidence but did not itself create a
filtered relation view. The frozen rule therefore required every derived child
table to prove

```text
raw rows = retained rows + frozen-quarantine rows
```

and contain zero `W-1` source or target values. `W0` remains retained and must
be counted separately. Normal referenced targets absent from the main snapshot
are coverage observations, not `W-1` quarantine rows and not automatically
orphan citations.

The 2026-07-17 full work-source audit subsequently proved the required per-file
and global accounting over all 21,019 files:

```text
20,996,232,817 raw source rows
= 20,996,232,796 retained source rows
+ 21 frozen W-1 quarantine rows
+ 0 null, unexpected-malformed, or grammar-valid orphan source rows
```

Every retained child source work ID was an exact member of the 492,293,206-ID
grammar-valid main-work set. The two declared work-target columns contained
zero null, malformed, `W-1`, or `W0` targets; snapshot-external valid targets
were reported as open-world coverage observations. The audit also produced a
descriptive author-ID/work profile. It did not materialize a filtered child
dataset, validate author entities, select candidate researchers, or establish
temporal or leakage safety. The compact result is
[here](../../../experiments/evidence/results/openalex_g1_work_source_membership_20260717.summary.md).

## 5. Gate consequences

The following statuses coexist and must not be collapsed:

| Layer | Status |
| --- | --- |
| Raw main-ID validity | `BLOCKED` |
| Frozen quarantine specification | satisfied |
| Grammar-valid main-ID recount | passed |
| Work-source membership and row conservation | passed; audit-only |
| Declared work-target coverage | measured; open-world |
| Authorship author-ID/work profile | measured; descriptive only |
| Materialized filtered analysis view | not created |
| W0 entity validity | `UNRESOLVED` |
| W0 positive-ID exclusion sensitivity | passed; component-only, 26 non-null field margins unchanged |
| Parent content identity | `UNVERIFIED` |
| Full-root content SHA-256 | pending |
| G-1 data readiness | `PENDING` |
| Empirical G1 signature | `NOT_RUN`; `pass_g1=false` |

The inventory parent bound paths, sizes, and mtimes and remained stable across
the full audit. It does not prove inventory-era byte identity against same-size,
mtime-restored mutation. A separate 920.523 GiB content-hash job can freeze the
current bytes for future comparisons, but cannot retroactively prove equality
to the earlier footer-only inventory because that inventory has no content-hash
baseline. The audit is an exact statement about the data pages scanned in this
run, not a claim that the source is byte-identical to an earlier inventory date.

No manuscript may summarize this amendment as “492,293,207 valid works,” “the
source passed validation,” “the corpus had no invalid IDs,” or “G-1 passed.”
The allowed denominator is the declared 492,293,206-row grammar-valid view,
with the raw failure, W0 warning, and remaining gates reported alongside it.
