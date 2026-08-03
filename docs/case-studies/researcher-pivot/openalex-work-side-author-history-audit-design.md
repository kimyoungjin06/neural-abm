# OpenAlex Work-Side Author-History Audit Design

Status: `DESIGN_ONLY`

Production status: `PRODUCTION_NOT_RUN`

This note defines a bounded future audit of publication histories attached to
author identifiers observed on the current OpenAlex work-side authorship
table. It is paired with
`experiments/configs/openalex_g1_work_side_author_history_v0.yaml`.

There is no execution script, fixture, report, receipt, local data product, or
component result for this design. This revision makes a later synthetic-only
implementation contract-ready; it does not implement that executable. A
production executable and production output ABI remain a separate review and
are not authorized here.

No executable or result exists in this tranche.

## Claim boundary first

The unit is a **current-snapshot OpenAlex author-ID token**, not a verified person.
A future component pass may establish exact work-side accounting,
deterministic author-work deduplication, observed publication-span summaries,
yearly unique-work productivity, and frozen consortium and hyper-prolific
sensitivity grids for those tokens.

It cannot establish any of the following:

- that one token is one natural person;
- full author-master referential integrity;
- correct author-ID splits or merges;
- ORCID ground truth;
- historically available author identity;
- candidate-researcher eligibility;
- citation `available_at`, outcome maturation, or right censoring;
- G-1, G0, empirical G1, field selection, an NABM effect, or a scientific
  effect.

The future full status therefore has to retain the limitation in its name:

```text
PASS_CURRENT_SNAPSHOT_WORK_SIDE_AUTHOR_TOKEN_LINKED_RETROSPECTIVE_HISTORY_ACCOUNTING_AND_SENSITIVITY_COMPONENT_AUTHOR_ENTITY_AND_PAST_OBSERVABILITY_UNVERIFIED
```

This string is a future contract name, not a current result. Current status is
only `DESIGN_ONLY / PRODUCTION_NOT_RUN`.

## Required parent order

A future production implementation must refuse full execution unless these
full components have completed in this core order and bind the same
current-content and temporal commitments:

1. `current_content_baseline` with a `COMPLETE` state and
   `PASS_CURRENT_CONTENT_BASELINE_COMPONENT` receipt;
2. `main_temporal_validity` with
   `PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_VALIDITY_COMPONENT_WITH_FROZEN_EXCLUSIONS`;
3. `canonical_work_time_spine` with
   `PASS_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_COMPONENT_WITH_INHERITED_TEMPORAL_EXCLUSIONS`;
4. `work_side_author_history` under this bounded contract.

The temporal and spine components have synthetic implementations but no
promoted production result. Consequently this design cannot be run or
promoted now. The work-topic relation is a parallel sibling, not a core parent:
it joins author history only at downstream current-snapshot field-coverage,
candidate, or `t-1` gates. Citation-edge chronology, fixed-horizon outcomes,
and candidate rules remain downstream blockers even after this component.

Every production parent receipt must be full-scope, internally complete,
current at final verification, and cross-bound to the same content tree. The
spine must transitively bind the exact temporal result. The future author plan
must also bind the exact current rich-stat identity and content-plan task
subsets for all 493 main and 1,363 authorship files. This design verifies no
production parent receipt; their production states remain unestablished.

The frozen taxonomy reference is only a bounded current-snapshot regression
and coverage prerequisite. It does not establish that a primary-field
annotation existed or was observable at a historical cutoff.

The config freezes the current parent design/tool identities and future
receipt ABIs rather than accepting status strings alone:

| Component | Current config SHA-256 | Current tool SHA-256 | Required receipt ABI |
| --- | --- | --- | --- |
| content baseline | `0340e9fd01e62d6d29b66245795a19100682847f94c5cb8afecc43e3d1a1c6e4` | `4a084f9f488af17fa199c9367d1948d1c129e6f1ad7af2dee529e2eed4da915f` | `openalex_g1_content_freeze_outputs.v1`, `COMPLETE`, exact three-artifact manifest |
| main temporal | `d208b45cc87de577726a8310a899eb66882a519bae29df5f7dbbf7833562b172` | `b4b39c016ba4c6867a3e08366f70f6125230c2a0a18c1e8c45d40826f373f563` | `openalex_g1_main_temporal_validity_outputs.v1`, 497 exact artifacts, receipt-last |
| canonical spine | `192d78e3df094121e63922a6eaf20f57a57af41bc174c0fdbbcc347e3eaed385` | `827bab5a6edcd9a574149203c68c77ae5178a8da210499b81f05a388ace5c0dc` | future `openalex_g1_canonical_work_time_spine_outputs.v1`, 9,184 exact artifacts, receipt-last |

The author implementation must rehash these repository files and validate all
receipt keysets, artifact bytes/hashes, result/task fingerprints, content tree,
spine data tree, and temporal transitive commitment. Any later parent edit
requires an explicit author-contract hash refresh; silent compatibility is not
allowed.

## Why this component remains useful without a full author master

The work-side profile contains 113,748,578 distinct grammar-valid non-null
author-ID tokens. The only identified separate 2026-03-30 author source is a
delta lineage with 44,099,961 rows. Even under the strongest possible premise
that every delta row is unique and matches a different work-side token, its
maximum theoretical coverage is:

```text
44,099,961 / 113,748,578 = 38.76968115%
```

Actual unique overlap can only be smaller or equal. That delta may later
support `LINKED_SUBSET_DIAGNOSTIC_ONLY` under a separate lineage contract. It
cannot be a full parent, validate every token, or promote author-entity
validity.

The bounded alternative is to preserve the exact OpenAlex `A` token as the
analysis key, never reassign tokens from names or ORCID, and quantify how the
observed work history changes under precommitted anomaly sensitivities. This
closes an accounting gap without disguising the entity-validity gap.

## Exact input projections

The future audit scans only two tables.

| Table | Files | Raw rows | Source bytes | Exact projected columns |
| --- | ---: | ---: | ---: | --- |
| `openalex_works_20260330` | 493 | 492,293,207 | 150,062,449,219 | `id`, `authors_count`, `primary_topic__field__id` |
| `openalex_works_20260330__authorships` | 1,363 | 1,361,551,038 | 103,412,497,044 | `id`, `authorships__author__id`, `authorships__author__orcid`, `authorships__raw_author_name`, `authorships__author__display_name`, `authorships__author_position` |

The declared physical scope is 1,856 files, 1,853,844,245 rows, and
253,474,946,263 bytes (`236.067 GiB`). Publication date and year are no longer
reparsed from main: they come only from the receipt-bound canonical work-time
spine. Work type, affiliations, countries, citations, and current author
summary/topic fields are deliberately outside this component.

`primary_topic__field__id` is used only for outcome-blind missingness and
coverage margins. It cannot select a field or define eligible researchers.
Null fields are explicit; non-null values must be decimal-normal-form field IDs
1 through 26 under `https://openalex.org/fields/`. That grammar is a bounded
current-snapshot regression, not historical field truth.

Work and author numeric suffixes must fit unsigned 64-bit integers; JSON booleans
are never integers. Exact `W-1` is quarantined before numeric parsing, `W0` is
retained with unresolved entity validity, null author IDs count as raw slots but
not histories, and malformed non-null author IDs block the component.
`authors_count` must be a non-boolean, non-null integer from 0 through
4,294,967,295 and must equal the observed raw authorship-slot count.

## Frozen accounting anchors

The design inherits these exact current-work-profile anchors:

- main: 492,293,207 raw rows, one frozen `W-1` quarantine row, and
  492,293,206 retained rows;
- authorships: 1,361,551,038 raw rows, one frozen `W-1` source row, and
  1,361,551,037 retained rows;
- retained author-ID rows: 580,116,419 null, zero malformed non-null, and
  781,434,618 grammar-valid non-null;
- distinct grammar-valid non-null author tokens: 113,748,578;
- ORCID rows: 458,308,219 non-null and 903,242,818 null;
- work classes: 263,750,998 complete, 2,109,042 partial, 177,217,069 none
  usable, and 49,216,097 with zero authorship rows; and
- all 492,293,206 retained main `authors_count` values exactly match observed
  raw authorship slots.

A future full audit must prove, rather than assume, these identities:

```text
retained authorship rows
  = null author-ID rows
  + malformed non-null author-ID rows
  + grammar-valid non-null author-ID rows

grammar-valid author slot rows
  = unique (author token, work) pairs
  + duplicate-pair row excess

history-usable author-work pairs
  = sum of author-year unique-work counts
  = sum of author-token total unique-work counts
  = sum of team-band unique author-work pairs
```

Every retained authorship row must join exactly one retained main work. An
unmatched row or a mismatch between `authors_count` and raw authorship slots
blocks the component. Duplicate author-work rows do not inflate publication
counts. The exact key is `(author_numeric_id, work_numeric_id)`. One pair
counts once while the accumulator preserves raw multiplicity, exact raw and
canonical-valid ORCID sets, exact raw/display and normalized-name sets,
position-category sets, and conflict flags. It never chooses a representative,
first, last, maximum, or otherwise preferred source row.

Every raw ORCID, name, display-name, and position set uses tagged values:
`{"kind":"null"}` for JSON null and `{"kind":"string","value":...}` for an
exact string. Null and the empty string are therefore distinct. Canonical-valid
ORCID sets contain only valid non-null canonical strings. Position diagnostics
separately report category conflict and raw-lexeme conflict; the combined flag
is true when either is true, so two different unexpected lexemes cannot be
silently collapsed into one category.

Physical authorship row order is nonsemantic and cannot break ties. The
projected position categories are `first`, `middle`, `last`, null, and
unexpected non-null. They are descriptive only and cannot change identity,
publication count, team size, or eligibility. The flattened table does not
recover the total ordering among middle authors. The exact raw position set is
also retained, so unexpected lexemes are not collapsed away by the category.
Category sets use the frozen declared order `first`, `middle`, `last`, null,
then unexpected non-null; they do not use lexical or UTF-8 order. Duplicate
category entries are removed before hashing.
Only the exact lowercase UTF-8 strings `first`, `middle`, and `last` enter those
three categories. There is no trimming, casefolding, or Unicode normalization;
every other non-null string, including empty, case variants, and whitespace
variants, is `unexpected_non_null`.

ORCID, name, and position diagnostics are computed and digest-committed for two
explicit populations: all retained grammar-valid distinct author-work pairs,
and the history-usable subset whose spine disposition is 2 or 3. The synthetic
public JSON exposes only each population's distinct-pair count and opaque
canonical aggregate digest; it does not expose readable category or
cross-cardinality tables. Any future detailed tables remain operator-local.
Duplicate source rows contribute once per pair after their tagged sets are
unioned. Neither population establishes a person entity or historical
observability.

## Canonical temporal reuse

Temporal v0 does not publish row-level dispositions, so this component cannot
claim to consume rows that do not exist. It instead requires the canonical
work-time spine, which is the first row-level commitment under
`FROZEN_MAIN_TEMPORAL_POLICY_V1`. Every retained main work must join exactly
one spine row by numeric work ID.

The exact spine disposition contract is:

```text
history usable = {2, 3}
excluded       = {0, 1, 4, 5}
```

These sets are unique, disjoint, and exhaustive over the six-value spine enum.
They retain works from `1900-01-01` through `2026-03-30`, inclusive, and
exclude both-null, valid pre-1900, snapshot-year-after-snapshot, and
post-snapshot rows. The synthetic fixture accepts only codes 0 through 5. Code
0 requires both day and year to be null; codes 1 through 5 require both to be
non-null, require the year derived from the epoch day to equal the stored year,
and must satisfy the exact ordered spine predicate for that code. Codes 6
through 255, a day/year/disposition mismatch, and any missing, duplicate, or
extra spine work ID are rejected before accounting. The fixture spine ID set
must equal the retained main-work numeric-ID set exactly. Any divergence from
the spine receipt blocks the future component. Author history must not reopen
main publication dates to create a competing policy.

The result remains a current-snapshot retrospective reconstruction. It does
not prove when a token assignment, authorship, ORCID, name, position, or
primary-field annotation first became observable.

## Observed token-history metrics

The implementation should use `observed_author_token_publication_span`, not an
unqualified career label. Per author-year it records unique works, team-band
counts, and descriptive primary-field counts. Per token it records first and
last usable publication year, active years, inclusive span, inactive gaps,
total unique works, and maximum annual unique works.

Required consistency includes:

```text
first usable year <= last usable year
1 <= active years <= inclusive span <= 127
maximum annual unique works <= total unique works
```

These full-snapshot summaries are audit-only. For a future decision year `t`,
every candidate, feature, exposure, or `t-1` graph consumer must recompute from
unique events satisfying:

```text
publication_date <= December 31 of (t - 1)
```

Equivalently, it uses the spine `publication_day` on or before the epoch-day
encoding of that cutoff. It may not reuse a full-history last year, span,
total, or maximum that includes post-cutoff works. Even after this restriction,
the token linkage and primary-field annotation remain retrospective
current-snapshot proxies, not evidence that those identifiers or fields were
observable at `t-1`.

`A0` remains grammar-valid in the inherited base syntax but has unresolved
entity validity. The audit reports both the grammar-valid base view and an
exact positive-author-ID sensitivity. Neither view decides whether `A0` is an
entity.

## Consortium and hyper-prolific sensitivity

Team size comes from validated raw authorship-slot count, not the number of
usable author IDs. This prevents the 42.607% null author-ID rate from changing
the large-team classification.

Frozen team bands are `1`, `2`, `3-5`, `6-10`, `11-50`, `51-100`, and `101+`.
Histories are recomputed for team caps `10`, `50`, `100`, and `ALL`.

Annual unique-work bands are `1`, `2-5`, `6-10`, `11-25`, `26-50`, `51-100`,
`101-250`, `251-500`, `501-1000`, and `1001+`. The exact hyper-prolific
sensitivity thresholds are strictly greater than `50`, `100`, `250`, `500`,
and `1000` unique works per year. Each threshold is recomputed within every
team-cap view.

These thresholds are diagnostic. Crossing one does not label an author token
invalid and cannot automatically exclude a candidate researcher.

## ORCID and name diagnostics

ORCID values are partitioned into null, strict canonical URL with valid ISO
7064 MOD 11-2 checksum, recoverable alternate lexical form with valid
checksum, and malformed/checksum-invalid. Null means a true JSON/Arrow null;
empty strings are malformed. v1 performs no trimming, percent decoding, or
Unicode-digit recovery. The strict form is
`https://orcid.org/0000-0000-0000-000X`. Recoverable forms are exactly the
declared `http`, `www`, bare-hyphenated, and bare-16-character ASCII variants.
They are normalized to HTTPS, hyphenated groups, and uppercase terminal `X`
before checksum validation.

For the first 15 ASCII digits, ISO 7064 validation iterates
`total = (total + digit) * 2`; the check value is
`(12 - (total mod 11)) mod 11`, with value 10 represented by uppercase `X`.
The audit measures token-to-ORCID, ORCID-to-token, and
author-work-pair-to-ORCID cardinality.

A token with multiple ORCIDs is only a merge-like anomaly candidate. An ORCID
attached to multiple tokens is only a split-like anomaly candidate. Neither is
ground truth.

Names retain exact raw-string accounting, with null and empty distinct. The
diagnostic normalized view applies Unicode NFKC, Unicode casefold, maps each
character for which Python `str.isspace()` is true to ASCII space, collapses
ASCII-space runs, and strips outer ASCII space. It does not apply a second NFKC
after casefold. The plan, checkpoint, and resume contract bind Python
implementation/version plus `unicodedata.unidata_version`. Exact and normalized
sets sort by unsigned UTF-8 bytes. Normalized names never change token identity.

## Deterministic partition plan

The proposed production implementation is a non-sampling, collision-free,
two-stage exact numeric-modulo partition:

1. validate the full spine receipt, all 4,096 partitions, and its transitive
   content/temporal commitments;
2. scan main works into `work_id_numeric % 4096` partitions with team and
   current-snapshot primary-field metadata, then exact-join the spine;
3. scan authorships into the same work-ID partitions;
4. deduplicate exact author-work keys without representative-row selection;
5. repartition joined events by `author_id_numeric % 4096`;
6. aggregate author-year and author-token histories; and
7. externally sort exact ORCID, name, and position diagnostic sets.

Input files sort by lexical relative path. Work-stage records sort by numeric
`(work_id, author_id)`, author-stage records by `(author_id, work_id)`, and
author-year aggregates by `(author_id, publication_year)`. Raw string sets sort
by unsigned UTF-8 bytes. Hashes may bind records but may not replace the exact
numeric identity or deduplication key.

Checkpoints bind the task fingerprint, path, rich source identity, projected
schema and payload hashes, and spool segment path, offset, row count, and hash.
Resume must verify both current sources and completed spool segments.

## Resource envelope

This remains a local CPU/I/O and external-sort task. It uses no GPU.

- planning RAM: 64 GiB minimum, 128 GiB preferred;
- durable output: 30-80 GiB planning range;
- temporary SSD: 250 GiB minimum, 400 GiB preferred; and
- planning runtime: 4-12 hours, to be replaced by a live preflight estimate.

The production run must not compete with the full content-freeze scan for HDD
bandwidth.

## Pilot and full promotion

- `--max-files` is parser smoke only and necessarily truncates histories.
- An author hash-bucket pilot can test selected-token histories only if it
  scans the full authorship root; it cannot establish global coverage.
- The incomplete author-delta join is linked-subset diagnostic only.
- None of those modes may promote the component.
- Full scope requires all 1,856 files, explicit authorization, all full parent
  receipts, private `0700` output, `0600` files, exact accounting closure, and a
  separately reviewed production completion state/receipt ABI. That production
  ABI is not implemented by this design.

## Frozen synthetic-only implementation ABI

This revision authorizes a later implementation tranche to build only
`scripts/audit_openalex_work_side_author_history.py`. It does not create that
file now. The future CLI is limited to `--config`, `--synthetic-fixture`,
`--out-dir`, explicit `--execute-synthetic`, and `--resume`; there is no
production or generic source-root option. The production root ID in the design
config is nonexecuted provenance metadata. A fixture root ID equal to or
containing that production ID is necessarily checked only after strict JSON
parsing, because the value is inside the payload, but before row validation or
core processing. Before opening the file, the runner can and must reject a
production-root path component and enforce `lstat` regular-file, single-link,
non-symlink, disjoint output-path, and 1 MiB size checks.

The strict duplicate-key JSON fixture rejects NaN, infinities, and lone Unicode
surrogates. The YAML config is also duplicate-key strict: a future tool accepts
only the exact audited repository config whose SHA-256 is embedded in that
tool, rejects aliases, merge keys, and custom tags, and compiles the production
prohibitions, caps, and `0700`/`0600` modes as constants that must equal the
config. It is at most 1 MiB and has at most 10,000 main rows, 10,000 spine rows,
and 20,000 authorship rows. Its top-level and every row object use exact
keysets and types; `snapshot_date` is exactly `2026-03-30`, integers never
accept booleans, and parent digests are lowercase 64-hex strings. Its top level
explicitly says
`surface_kind=SYNTHETIC_FIXTURE_ONLY` and
`production_input_access_allowed=false`, and its root ID starts with
`synthetic-`. Embedded content, temporal, and spine parents have separate exact
keysets and synthetic-only status literals; all three share content
commitments, while temporal and spine also share the exact temporal result
commitment. The runner checks the file-size cap before payload open and all row
caps before core processing.

Synthetic plan, checkpoint, state, report, and receipt schemas are separate
from every future production schema. Their outer and nested keysets and value
types are frozen in the YAML contract, including the two diagnostic
populations, all six temporal dispositions, A0/base and positive-ID views,
position category and raw-lexeme conflicts, team-cap views, and hyper-prolific
thresholds. The only allowed success status is
`PASS_SYNTHETIC_WORK_SIDE_AUTHOR_TOKEN_HISTORY_CONTRACT_ONLY`; it may never
emit the future full component status. The completion receipt binds exactly:

```text
audit_plan.json
run_state.json
checkpoints/synthetic_accounting.json
author_history_audit.json
author_history_audit.md
```

The sensitivity object is not a five-number summary. It contains separate
`base` and `positive_author_id` views, exact pair counts for caps 10, 50, 100,
and ALL, and all 20 cap-conditioned threshold cells formed by four caps times
`gt_50`, `gt_100`, `gt_250`, `gt_500`, and `gt_1000`. A cell counts distinct
tokens whose maximum annual unique-work count, recomputed within that cap view,
is strictly greater than its threshold. Positive-ID cells cannot exceed the
corresponding base cells, and positive-ID cap-pair counts cannot exceed the
corresponding base counts. Because each numeric cap is cumulative, every fixed
threshold obeys cap 10 <= cap 50 <= cap 100 <= ALL. Within each cap, counts are
nonincreasing as the threshold rises. The ALL pair counts equal the respective
base and positive history unique-pair counts, and each visible grid object is
rehashed under its declared base/positive domain to obtain the published grid
digest.

The receipt `author_history_audit.outputs.json` is written last under
`SYNTHETIC_RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED`. A valid-success
terminal re-entry must validate the exact published file set and rehash all
five artifacts before opening the fixture. A failed-state resume instead uses
the separate exact failed-entry and checkpoint revalidation contract.
Finalization failure after a valid checkpoint removes the authoritative receipt
first, writes `FINALIZATION_FAILED_NO_VALID_RECEIPT`, removes final report
surfaces, omits the result fingerprint, and preserves only a validated
resumable checkpoint. A failed-state resume may transition back to
`RUNNING_SYNTHETIC_AUTHOR_HISTORY` only after full plan, config, tool, runtime,
fixture, parent, checkpoint, state, exact-entry-set, type, mode, and link-count
revalidation. Failed revalidation leaves the state failed and the receipt
absent; a direct failed-to-complete transition is always forbidden. For the
same task, a successful failed-state resume must reproduce the clean run's
result fingerprint, although timestamps, receipt bytes, and artifact hashes
need not be identical. Output directories are `0700` and files `0600`.
An error before a valid checkpoint cannot publish the resumable failure status;
it removes all output entries, fsyncs the empty root, and returns nonzero.

All synthetic hashes use UTF-8 canonical JSON with `ensure_ascii=false`,
Unicode-code-point key order, compact comma/colon separators, no NaN or floats,
and no trailing newline. Domain bytes, one NUL byte, and the canonical payload
are SHA-256 hashed under distinct task, metrics, checkpoint, and result v1
domains. The contract fixes each exact payload keyset; raw config, tool,
fixture, and artifact SHA-256 values bind exact file bytes.

The checkpoint's nine metric objects are the semantic source for the nine
same-named JSON report sections and must be exactly equal as parsed canonical
objects. The plan is authoritative for task fingerprint, parent commitments,
and the three row counts; checkpoint, terminal state, report, and receipt bind
the same task, while checkpoint source accounting, terminal completed counts,
and report scope project the same plan row counts. Metrics and checkpoint
self-hashes are recomputed. The result fingerprint is recomputed from the plan
task, checkpoint metrics hash, plan parents and row counts, checkpoint gate
status, and the exact top-level claim boundary, then must equal the terminal
state, report, and receipt values. Status and production-status literals are
cross-bound as well. Before writing the success state, validation covers the
plan, checkpoint, in-memory report, and expected result but explicitly excludes
the not-yet-published terminal-state and receipt targets. After writing
`COMPLETE_PENDING_RECEIPT`, validation reopens all five pre-receipt artifacts
and includes the state targets. A prospective receipt object and its five
artifact records are then validated before the receipt is atomically written;
the post-write disk validation and valid-success terminal re-entry include the
receipt targets. Validation first establishes the repository config, plan,
task, parents, and checkpoint self-hashes. If any authoritative source is
invalid, the nonresumable pre-checkpoint cleanup route is used. Receipt-first
invalidation and the resumable failed state are reserved for a dependent
report, state, receipt, or artifact-binding failure while the authoritative
plan and checkpoint remain valid. The future executable test suite must mutate
both endpoints of every declared semantic edge, recompute local self-hashes,
and assert the corresponding failure route. It must also recompute receipt
artifact hashes, so rejection exercises the semantic link rather than stopping
at a simpler byte-hash mismatch. Those runtime mutation tests are not
implemented in this design-only tranche.

The four history digests, two diagnostic-population digests, and two team-grid
digests also have separate domains. Their exact canonical record ABIs freeze
the deduplicated author-work diagnostic pair (multiplicity, every tagged set,
and all category/raw conflict flags), per-author-year aggregate,
per-author-token aggregate, and base/positive team-grid payload. Records are
framed as one canonical JSON array and sorted by the declared numeric key;
grids are one canonical JSON object. Empty populations hash the canonical empty
array. Base and positive token counts are checked against the corresponding
per-token record-array cardinalities during construction, before aggregate and
metrics hashing. In each view, `(author_numeric_id, publication_year)` is unique
in the per-year array, `author_numeric_id` is unique in the per-token array, and
the two arrays must have the same author-key set. Every token record must
exactly reduce only that author's year records: minimum and maximum year,
distinct active-year count, inclusive span and
gap, sum of annual unique-work counts, and maximum annual count are all checked
jointly. This rejects a swapped-author summary even when global counts and sums
are unchanged. The positive-ID per-year and per-token arrays, together with
every positive-ID hidden cap-specific pair/year/token array, must be the exact
stable-order filter of the corresponding base array under
`author_numeric_id > 0`.

For each view and each cap, the hidden capped pair records are also the sole
construction-time source for that cap's pair count and threshold cells. Numeric
caps filter on `declared_authors_count <= cap`; ALL retains every history-usable
pair in the same view. Cap-specific author-year records count unique works,
cap-specific token records apply the same same-author reductions, and each grid
cell exactly counts distinct same-cap author IDs whose cap-specific maximum
annual unique-work count is strictly above the declared threshold. A record
array from another view or cap cannot substitute as the reduction source.

These are construction-time hidden cross-record closures only. The hidden
arrays are not published or reopened, so terminal re-entry validates the
visible receipt-bound counts and digests but does not claim an independent
recomputation of hidden cardinality, projection, author-wise reduction, or grid
reduction. Thus changing multiplicity, a tagged ORCID/name/position value, a raw or
category conflict, A0 inclusion, a cap, or a threshold necessarily changes the
relevant aggregate digest and the outer result fingerprint. For identical
fixture bytes, internal traversal order, chunking, and valid resume do not
change the result. Permuting row arrays in the fixture creates different exact
fixture bytes: canonical aggregate objects and the metrics hash remain equal,
but fixture SHA, task fingerprint, and outer result fingerprint must differ.

Fixture and output paths may not be equal or ancestor/descendant overlaps.
Inputs are opened with `O_RDONLY|O_NOFOLLOW` and post-open `fstat` must equal
the pre-open `lstat`. Existing and published output files must be regular,
operator-owned, mode `0600`, and link count one. Publication uses same-directory
exclusive `0600` temporary files, `fsync`, `os.replace`, and parent-directory
`fsync`; final validation rechecks every mode and link count. A valid terminal
`--resume` is a zero-mutation path: bytes, hashes, inode, mode, `mtime_ns`, and
`ctime_ns` for every existing entry remain unchanged before it returns.
Run-state tuples and transitions are exact: planned/running states have null
errors and result fingerprints, successful `COMPLETE_PENDING_RECEIPT` has a
non-null result fingerprint, and finalization failure has a redacted error and
null result fingerprint. `completed_row_counts` reuses the exact fixture row
count schema. Receipt artifact records have exact path/string, non-boolean byte
count, and lowercase digest types; mapping keys equal repeated paths, and byte
length and SHA-256 are recomputed from each published payload. Main and
authorship `W-1` rows are reported separately with an explicit sum closure.
Markdown is a receipt-bound but nonauthoritative v1 presentation; the JSON
report is the semantic authority.

The synthetic core is only a candidate for later code sharing. Because the
production executable and writer ABI do not exist, this design does not claim
synthetic/production implementation parity.

## Production outputs and privacy

The production executable, Parquet schemas, writer ABI, receipt, and promotion
path remain `NOT_IMPLEMENTED_REQUIRES_SEPARATE_REVIEW`. Existing control and
Parquet names are provisional privacy-routing names, not a frozen production
schema and not outputs created by this design.

Detailed author IDs, raw names, display names, ORCIDs, and top-author rankings
must remain operator-local `0600` artifacts. Repository and release surfaces
allow only aggregates and checksums. Filename denylisting alone is insufficient:
compressed or renamed data could bypass it. The release inspector therefore
uses an exact path-and-content-SHA allowlist for the public
`docs/case-studies/researcher-pivot/` subtree and rejects every unmatched,
non-regular, or byte-drifted member. Tests include `.parquet.gz`, `.csv`,
`.tar.zst`, renamed-record, and modified-allowed-path canaries. The older
provisional basenames and `author_history_staging`,
`author_history_checkpoints`, and `author_history_data` directory names remain
defense-in-depth rejections.

The tar boundary is checked before root stripping: exactly one top-level sdist
root is allowed; absolute paths, empty or dot segments, `..`, repeated
separators, backslashes, control characters, duplicate raw or stripped names,
and every non-regular member are rejected. Actual adversarial tar tests cover
dot-dot and backslash aliases, duplicate allowed paths with one forbidden
payload, symlink/hardlink members, unexpected directories, and multiple roots.

## Remaining blockers after a future component pass

Even a correctly promoted bounded component leaves:

- `author_entity_validity=UNRESOLVED_WORK_SIDE_ONLY`;
- `author_master_coverage=BLOCKED_INCOMPLETE_MAX_38_76968115_PERCENT`;
- `split_merge_validity=UNRESOLVED_PROXY_DIAGNOSTICS_ONLY`;
- `orcid_validation=DIAGNOSTIC_NOT_GROUND_TRUTH`;
- `token_and_field_past_observability=UNVERIFIED_CURRENT_SNAPSHOT_RETROSPECTIVE_PROXY`;
- candidate eligibility, citation availability, maturation, and `t-1` graphs
  pending;
- G-1 and G0 pending, empirical G1 not run, and no scientific effect
  established.

The component is therefore an accounting prerequisite for later panel design,
not a substitute for an author-entity source or an eligible researcher panel.
Work-topic relation remains a parallel sibling and joins only in downstream
field-conditioned or candidate gates.
