# OpenAlex G-1 Work-Source Membership and Profile Audit

## Status and gate boundary

The 2026-07-17 full audit passed the frozen work-source membership/profile
component across all 23 declared tables:
`PASS_WORK_SOURCE_MEMBERSHIP_AND_PROFILE_AUDIT_WITH_FROZEN_QUARANTINE`.
This is not a G-1 or scientific-result pass. Raw source ID validity remains
`BLOCKED` by the registered `W-1` rows, `W0` entity validity is `UNRESOLVED`,
parent content identity is `UNVERIFIED`, G-1 data readiness is `PENDING`, and
empirical G1 is `NOT_RUN` (`pass_g1=false`). No filtered analysis view was
materialized.

- Component status:
  `PASS_WORK_SOURCE_MEMBERSHIP_AND_PROFILE_AUDIT_WITH_FROZEN_QUARANTINE`

## Exact row accounting

The audit read data pages from 21,019 files and conserved every source row.

| Scope | Raw rows | Retained rows | Frozen W-1 | Retained W0 | Orphan / null / other malformed |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 23 tables | 20,996,232,817 | 20,996,232,796 | 21 | 6 | 0 / 0 / 0 |
| Main works | 492,293,207 | 492,293,206 | 1 | 1 | 0 / 0 / 0 |
| 22 child tables | 20,503,939,610 | 20,503,939,590 | 20 | 5 | 0 / 0 / 0 |

Every retained child source ID belongs to the 492,293,206-ID grammar-valid
main-work set. The 21 `W-1` and six retained `W0` rows exactly match their
frozen locator ledgers. This audit accounts for the quarantine; it does not
rewrite the source or produce a filtered child dataset.

## Open-world work-target coverage

| Target | Rows in snapshot | External rows | Row coverage | Unique in snapshot | Unique external | Unique coverage | Self-loops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Referenced works | 2,904,520,313 | 138,015,847 | 95.463789% | 112,406,279 | 23,589,745 | 82.654092% | 1,072,993 |
| Related works | 2,366,002,635 | 127,158,784 | 94.899697% | 90,310,458 | 9,754,731 | 90.251624% | 4,663,862 |

Both target columns contain zero null, malformed, `W-1`, or `W0` target rows.
Snapshot-external targets are open-world coverage observations, not source
orphans. Duplicate source-target edges were not measured.

## Descriptive author-ID/work profile

The retained authorship view contains 1,361,551,037 rows: 781,434,618
grammar-valid non-null author IDs (57.393%), 580,116,419 null author IDs
(42.607%), and zero malformed non-null author IDs. It contains 113,748,578
distinct grammar-valid author IDs. ORCID is non-null in 458,308,219 rows
(33.661%). Grammar-valid syntax and ORCID presence do not establish author
entity validity.

Across 492,293,206 retained works, the descriptive classes are:

- `complete`: 263,750,998;
- `partial`: 2,109,042;
- `none_usable`: 177,217,069; and
- `zero_authorship_row`: 49,216,097.

All declared author-slot counts match observed retained slots. The report also
closes the margins for 1,050 publication-year buckets, 27 primary-field
buckets, eight declared-author bands, seven observed-team-size bands, five
authorship-lineage masks, and 122,472 sparse joint cells. These are descriptive
profiles, not candidate-author eligibility or entity-validation results.

## Commitments, verification, and remaining work

- Result fingerprint:
  `8b7d0273a04eb7a231a9d8b466151c562ac80eac6557ff7444508cb174862242`
- Output-manifest SHA-256:
  `8fb34f92e3525b58c09a31ae970d913ea6858211a7595b165884fa389863156d`
- Verification-receipt SHA-256:
  `591b8fbb32698bd0d9dd8e167527e2fd75996fe0d9bdcff048ac074ec0db6d66`
  ([path-redacted JSON receipt](openalex_g1_work_source_membership_20260717.verification.json))
- Canonical plan SHA-256:
  `7afef25a788a938b9eeb12a1802d346eb02da1d3b369a116c784851e984891cb`
- Parent manifest SHA-256:
  `6b5929a0155c033e818a816aad57095380a3907ab558c8d2cd58a89055f22b46`
- Tool SHA-256:
  `a5f1a4175b575c16a8c45df68c15fa2dbae8554c2c7ae258cdaa61586915f3f0`
- Audit-time Git provenance: commit
  `10dd3e8fd1c7cc1f30088fcdbef510439e172917`, `dirty=true`. The separately
  frozen tool, config, test, plan, parent, and output hashes are the executable
  provenance boundary for this result.
- Operator-local full audit report JSON / Markdown SHA-256 (not these compact
  summaries):
  `1c385d42595fc34794116189ce940526e1095ccd2a5da9eccd3da902e8f152ce` /
  `34b658b8fcfd601ec133f7af20d262d5c153abb36c5a33336da0b221dfc88275`
- Checkpoint tree: 21,019 files, 51,417,147 bytes,
  `1c311833193a7073a5ca78e6cf86be702f38e69bb5ef2e9388f814eddab9f01c`.
- Data-artifact tree: 8,757 files, 89,957,407,904 bytes,
  `bb0c711f8cd6b678d6ced92d10ff5530cf25d37c3f3126dc25b96ea986a61ba9`.

Two independent read-only audits found no P0 or P1 issue. They recomputed the
path sets and tree hashes, verified no missing, extra, symlinked, non-regular,
or empty-directory evidence, revalidated all 21,019 checkpoint bindings,
physically reread all 27 sentinel locators across 12 source files, and exactly
reconstructed the reported quantitative margins.

The clean semantic-resume replay passed with exit code 0 after semantically
replaying all 21,019 existing checkpoints from the original Parquet inputs. It
returned the same component status, result fingerprint, and output-manifest
SHA-256 as the fresh full run. The path-redacted verification receipt preserves
the captured final stdout payload and records that a byte-for-byte terminal
progress transcript was not retained.

Remaining gates include a current full-root content-hash baseline, the
unresolvable-with-current-evidence question of earlier inventory byte equality,
`W0` exclusion sensitivity, author and field entity validity, duplicate-edge
measurement, publication-date and censoring checks, cutoff-aware `available_at`
and `t-1` graph validation, leakage tests, candidate eligibility, a
deterministic representative pilot, and the measurement/holdout freeze.
Consequently this result does not establish G-1, empirical G1, or any
scientific effect.
