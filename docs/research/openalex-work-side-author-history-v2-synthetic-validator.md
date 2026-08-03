# OpenAlex work-side author history v2 synthetic validator

Status: `IMPLEMENTED_SYNTHETIC_FIXTURE_VALIDATOR_ONLY`

## What is implemented

The v2 surface adds a closed, stdout-only synthetic validator for the
work-side author-token accounting core. It reads one exact repository config
and one bounded JSON fixture, validates the parent commitments and row
closures, reduces duplicate `(author_numeric_id, work_numeric_id)` pairs, and
reports the base and positive-author-ID history views. It writes no files and
has no production or generic source-root option.

The unit remains a **current-snapshot OpenAlex author-ID token, not a person**.
The result is a retrospective accounting check, not evidence that an author
token, authorship, ORCID, name, position, or field annotation was visible at a
historical decision time.

The contract is
`experiments/configs/openalex_g1_work_side_author_history_v2.yaml`. It overlays
the byte-exact v1 design at SHA-256
`2904f47609fa8229897cbba8a14e00d6e2babd4f1696c1ca436c5dd17d1cdc66`.
The legacy v0 config, design test, and design note remain byte-exact and are
rehash-bound by the v2 config; v2 does not edit or silently reinterpret them.

Before it opens any fixture, the validator reopens and rehashes six exact
repository foundations: the author v1 config, all three legacy v0 surfaces,
the canonical v2 config, and the canonical v3 runner. Every item must be a
bounded regular single-link non-symlink file with the path, byte count, and
SHA-256 frozen in the v2 config. A mismatch fails before fixture `lstat` or
payload access. Successful stdout exposes the six sorted
`foundation_verification` records with
`repository_relative_path`, `bytes`, `sha256`, and
`verified_against_config=true`.

## Deliberately smaller than the frozen v0 execution ABI

This implementation tranche validates the scientific reduction core only. It
does **not** implement or claim the v0 plan/checkpoint/state/report/receipt
lifecycle, terminal re-entry, resumable finalization, private production
artifacts, or a production writer ABI. Its only successful product is one
canonical JSON report on stdout. That smaller surface is intentional and is
named `synthetic fixture validator`, not a completed author-history audit.

The currently bound canonical-spine production surfaces are:

- config `experiments/configs/openalex_g1_canonical_work_time_spine_v2.yaml`,
  SHA-256
  `6e8b74ddd12cbef46c8d15d3560f0d54f470b5017685fc1bce2389250cde0484`;
- runner `scripts/audit_openalex_canonical_work_time_spine_v3.py`, SHA-256
  `bf9ed534b2e9f9c2d6c9851be7fde298eb372d8a3cf5e17363ea6afe51cf1280`.

Those surfaces are production-capable, but no canonical production receipt
exists. Therefore a synthetic author fixture is never a production parent and
author production remains
`BLOCKED_CANONICAL_PRODUCTION_RECEIPT_NOT_AVAILABLE`.

## Closed input contract

The fixture top level has exactly these keys:

```text
schema_version
surface_kind
production_input_access_allowed
root_id
snapshot_date
synthetic_parents
main_rows
spine_rows
authorship_rows
```

The fixed values are
`schema_version=openalex_g1_work_side_author_history_synthetic_fixture.v2`,
`surface_kind=SYNTHETIC_FIXTURE_ONLY`,
`production_input_access_allowed=false`, a `synthetic-` root ID, and snapshot
date `2026-03-30`. The file is limited to 1 MiB, 10,000 main rows, 10,000 spine
rows, and 20,000 authorship rows. Strict JSON rejects duplicate keys, NaN,
infinities, lone Unicode surrogates, missing or extra keys, and booleans where
integers are required. JSON and YAML nesting is capped at 64. JSON integer
lexemes and work/author decimal suffixes are capped at 20 digits before
`int()` conversion, followed by the narrower field-specific range check.
Parser recursion errors, over-depth structures, and huge IDs are converted to
the same controlled exit-2/no-stdout/one-line-stderr failure surface; they do
not escape as Python tracebacks.

The final fixture and repository files are protected by component checks,
`lstat`, `O_NOFOLLOW`, pre/post-open identity comparison, and link-count
checks. Intermediate directory components are checked for symlinks but are not
held with an `openat` descriptor walk. The path race boundary is therefore
best-effort for a trusted, single-user local filesystem, not a claim of safety
against a concurrent hostile directory renamer.

The row keysets are exact:

| Surface | Keys |
| --- | --- |
| main | `id`, `authors_count`, `primary_topic__field__id` |
| spine | `work_id_numeric`, `publication_day`, `publication_year`, `temporal_disposition` |
| authorship | `id`, `authorships__author__id`, `authorships__author__orcid`, `authorships__raw_author_name`, `authorships__author__display_name`, `authorships__author_position` |

Exact `W-1` rows are quarantined before numeric parsing. Every other retained
work and author token uses decimal normal form (`0` or a nonzero leading digit)
and has no leading zero. Every retained work has one and only one spine row.
Disposition 0 requires null day and year;
dispositions 1--5 require a valid epoch day, the matching derived year, and
the frozen ordered predicate. Only dispositions 2 and 3 are history-usable.

Every retained authorship row must match a retained main work, and each main
row's `authors_count` must equal its raw authorship-slot count. A null author ID
is a raw slot but not a token history. A malformed non-null author ID is a hard
failure. Duplicate author-work rows count once while retaining their diagnostic
value sets; no representative row is selected.

The validator hashes those diagnostic unions twice: once for every retained
grammar-valid distinct pair and once for the history-usable subset. The v2
diagnostic record retains multiplicity, tagged raw and canonical-valid ORCID
sets, tagged raw/display and normalized-name sets, position category/raw sets,
and a v0-compatible nested nine-flag conflict layout. The two pair-diagnostic
domains are explicitly v2. Only opaque digests enter stdout; readable
diagnostic values and identifiers do not.

The three embedded parents use separate exact synthetic-only schema and status
literals. Their content commitments must agree, the temporal and spine temporal
commitments must agree, and the spine parent row count must equal the fixture
spine row count. Production parent status literals are rejected.

## Evidence classes

The report never treats every schema-valid fixture as checked evidence.

- `PINNED_CHECKED_FIXTURE` requires both the exact configured repository path
  and SHA-256. It receives
  `PASS_SYNTHETIC_WORK_SIDE_AUTHOR_TOKEN_HISTORY_V2_PINNED_FIXTURE_CONTRACT_ONLY`
  and its embedded parents are classified
  `CONFIG_PINNED_SYNTHETIC_PARENT_STUBS`. A byte mismatch at the pinned path is
  a hard failure, not a downgrade to generic.
- Every other valid fixture is `UNATTESTED_GENERIC_FIXTURE`, including a
  same-byte copy at another path. It receives only
  `VALID_UNATTESTED_GENERIC_WORK_SIDE_AUTHOR_TOKEN_HISTORY_V2_MECHANICS_ONLY`;
  its parents are `SELF_DECLARED_UNVERIFIED_SYNTHETIC_PARENT_STUBS`.

The CLI contains no production source interface, so
`production_source_interface_used=false` in both classes. For the pinned
fixture, the contract additionally supports `production_input_accessed=false`.
For a generic fixture its origin is unattested, so that field is `null`, not a
false claim. `pinned_fixture_sha256_match` reports raw byte equality but never
upgrades a copied fixture's evidence class without the exact path.

## Minimal fixture and expected reductions

The checked fixture is
`tests/fixtures/openalex_work_side_author_history_v2_minimal.json`, SHA-256
`e77556ac457424193dd33cc9760b00b36aaba06fd207bd8cc1b0fb356cf101ff`.
It has 8 main rows, 7 spine rows, and 21 authorship rows. It exercises exact
`W-1` quarantine, all six dispositions, a history-usable `A0`, a null author
slot, a duplicate pair with raw and category position conflict, and a team of
11 that separates cap 10 from cap 50.

The visible reduction anchors are:

| Metric | Expected |
| --- | ---: |
| retained works | 7 |
| declared / observed retained slots | 20 / 20 |
| grammar-valid author slots | 19 |
| distinct author-work pairs | 18 |
| duplicate-pair row excess | 1 |
| history-usable pairs | 14 |
| base / positive history tokens | 13 / 12 |
| base / positive history pairs | 14 / 13 |
| base / positive cap-10 pairs | 3 / 2 |
| base / positive cap-ALL pairs | 14 / 13 |

All 20 hyper-prolific cells in both views are zero in this minimal fixture.
They are nevertheless recomputed for caps 10, 50, 100, and `ALL`, using the
strict thresholds greater than 50, 100, 250, 500, and 1,000 annual unique
works. The adversarial test suite separately exercises 50 versus 51 annual
works and cap-10 versus cap-50 recomputation without changing the checked
fixture contract.

The fixture also freezes eight visible aggregate digests under the reported
runtime:

| Digest | SHA-256 |
| --- | --- |
| all-pair diagnostics | `580bfbbe89572338b4971cb89508f5af7c256011a799db99197095cac589ca97` |
| history-usable pair diagnostics | `4f896b6f08f9b04b936430c843b4523b1878a52f43b57414a4855a8cd4c6be36` |
| base per-year history | `7cd8491b7d30a2a7c24cf40d55a16d140c3709e1d3e4532095a8a5fd83549834` |
| positive per-year history | `830c671dd50327d1e163a289fb1549ce3cbcf4258e9ad86d3cf9eb7b4e3c83ab` |
| base per-token history | `14fa7c82315f20ada64e3984f1d685e498b7bd5cc963c5d3bdd5ee628a64ea99` |
| positive per-token history | `3ff5096fa11e161e71a86bf0ea9e2a5ec7ff2f48e7676dff4d2ee9414085073e` |
| base cap/threshold grid | `3cd666d759763357023450014b5fd64b2b8675f321e26e0757cf57892b901c2d` |
| positive cap/threshold grid | `c904a488ddbdbb17295993db125ac0071e4fc05e48d540d7fe1f1dfa556d80ff` |

The report binds `python_implementation`, `python_version`,
`unicodedata_unidata_version`, and canonical-JSON ABI in
`runtime_fingerprint`. These fingerprints and digests make the checked run
auditable; the contract does not assert byte-identical diagnostic hashes
across different Python or Unicode runtimes.

## CLI and stdout ABI

Run the validator with only the two accepted path flags:

```bash
.venv/bin/python scripts/verify_openalex_work_side_author_history_fixture_v2.py \
  --config experiments/configs/openalex_g1_work_side_author_history_v2.yaml \
  --synthetic-fixture tests/fixtures/openalex_work_side_author_history_v2_minimal.json
```

Success emits one canonical JSON object plus a newline to stdout and nothing to
stderr. The exact top-level report keys are frozen in the v2 config. The report
contains only aggregate counts and digests; it may not expose author IDs,
names, ORCIDs, or operator paths. `inputs` binds the config and fixture hashes,
runtime fingerprint, and six foundation-verification records. `scope` reports
the fixture and parent evidence classes, no GPU, no production source
interface, no files written, and no scientific-effect estimation, with
production-input access conditioned on evidence class as above.

For the checked repository run with config SHA-256
`f0359f9b9ab7c2e5a51b1e760f94c70d51e544dc7f594d8e0512a9cc8ff216c2`,
fixture SHA-256
`e77556ac457424193dd33cc9760b00b36aaba06fd207bd8cc1b0fb356cf101ff`,
CPython `3.14.0`, and Unicode database `16.0.0`, canonical stdout is 6,806
bytes with SHA-256
`ee2ca45756f09b130f6ad00fe7b56219eaf9083429d80eb657424212c5381ef7`.
This is runtime- and config-specific execution evidence recorded in this
unbound v2 note; it is intentionally not stored inside the config and creates
no config/report self-reference.

Both statuses retain `PRODUCTION_NOT_RUN`, `PENDING` for G-1 and G0, `NOT_RUN`
for the empirical G1 signature, `pass_g1=false`, and
`scientific_effect=NOT_ESTABLISHED`. Their distinct exact claim-boundary
strings are frozen in the v2 config and include the runtime and provenance
limits rather than relying on the status name alone.

## What remains blocked

This validator does not establish author-entity validity, correct token splits
or merges, complete author-master coverage, past observability, candidate
eligibility, citation chronology, cutoff panels, or causal effects. A separate
version and review are still required for the full production executable,
private-output schema, receipt, and promotion path. Until a real canonical
production receipt and all other G-1 receipts exist, G-1 and G0 remain pending.
