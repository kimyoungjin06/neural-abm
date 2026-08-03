# OpenAlex Temporal and Author Gate Inventory

Status: `TEMPORAL_V3_COMPONENT_PASS_AUTHOR_GATE_PRELIMINARY`

This note records the CPU/I/O audit boundary after the current main-ID,
work-source, taxonomy-reference, bounded W0-sensitivity, content-freeze, and
main temporal components. The linked v3 audit is now a bounded temporal-field
component pass. This inventory is still not an author-entity pass, G-1, field
selection, or empirical evidence.

## Available temporal columns

The current work export provides `id`, `publication_date`,
`publication_year`, `created_date`, and `updated_date` in the main table.
Referenced-work edges provide citing and cited work IDs but no independent
first-observed timestamp. The counts-by-year table provides year-specific
citation counts only over its represented range.

Current frozen profile and footer inspection records these diagnostics:

- 27,565,532 main works have null `publication_date`; the aggregate null count
  for `publication_year` is also 27,565,532;
- non-null publication years range from 1000 through 2050;
- 6,692,869 works are before 1900;
- 4,010 works are after the 2026 snapshot year;
- `created_date` begins in 2016, so it cannot be treated as an observation-time
  clock for older publication cohorts; and
- counts-by-year covers 2012 through 2026, not the complete historical citation
  trajectory of older works.

Equal aggregate null totals do not prove row-level date/year concordance. The
v1 full audit later confirmed exact publication date/year concordance but used
the wrong lexical parser for metadata timestamps. The corrected v2 one-file
pilot parsed all 1,000,000 metadata pairs and found three
`created_date > updated_date` rows. Because OpenAlex documents the two fields'
meanings but not a monotone relationship, v3 treats ordering as a descriptive
profile rather than a validity gate. Production validity must still parse each
field exactly, verify date-derived publication year, separate the frozen
publication buckets, reject metadata null/parse failures, and reject timestamps
after the snapshot boundary.

The full v3 execution subsequently established those bounded checks across all
493 main files: 492,293,206 retained metadata pairs parsed exactly with zero
null, parse, or future-field violations; all 464,727,674 nonnull publication
date/year pairs matched; and all accounting contracts passed. The three
`created_date > updated_date` rows remained descriptive. This pass does not
turn either metadata field into an edge-observation timestamp.

## Availability and maturation boundary

Using the citing work's publication date as a citation edge's `available_at`
is a bibliographic event-time convention. It does not prove when OpenAlex first
observed or indexed that edge. The convention may be used only after it is
precommitted, labeled, and tested for chronology violations and sensitivity to
date uncertainty.

The later repository-only design at
`openalex-citation-available-at-audit-design.md` freezes that convention,
directed-pair deduplication, open-world targets, seven exclusive chronology
buckets, and mode-specific input accounting. It remains `DESIGN_ONLY /
PRODUCTION_NOT_RUN` and does not change this inventory into a gate result.

The fixed outcome horizon, maturation buffer, right-censoring rule, allowed
work types, and treatment of missing or implausible dates remain unchosen.
Consequently no 2022-to-2025 cohort outcome, past-only graph, candidate panel,
or temporal holdout is analysis-ready.

## Author-source blocker

The work-side audit contains 113,748,578 distinct grammar-valid author IDs. The
only preserved separate 2026-03-30 author source currently identified is a
delta lineage with 44,099,961 author rows, not an authoritative current full
author master. Even if every delta row were unique and overlapped a different
work-side author, its theoretical coverage ceiling would be 38.77 percent.
Actual unique overlap can only be lower or equal.

That delta can support a clearly labeled linked-subset diagnostic. It cannot
establish referential integrity for all work-side author IDs, resolve true
split/merge errors, or provide independent ORCID ground truth. Current author
and topic summary fields also cannot be used as past-only model features.

The author gate is therefore assigned the preliminary blocker label
`BLOCKED_AUTHOR_MASTER_INCOMPLETE` until one of the following is precommitted.
This label is an inventory conclusion, not a passed or executed component
status:

1. reconstruct or acquire a same-snapshot full author master and audit exact
   work-side coverage; or
2. use OpenAlex author IDs as the analysis identity while explicitly giving up
   a full entity-validity claim, then freeze anomaly exclusions and an
   ORCID-linked sensitivity as robustness checks rather than ground truth.

Either route still requires author-to-ORCID cardinality, name inconsistency,
career-span, yearly productivity, hyper-prolific-author, and consortium
sensitivity diagnostics before candidate eligibility.

The later repository-only design at
`openalex-work-side-author-history-audit-design.md` chooses the bounded second
route for future implementation while retaining
`AUTHOR_ENTITY_UNVERIFIED`. It is `DESIGN_ONLY / PRODUCTION_NOT_RUN`, not an
author-entity pass.

## Required execution order

1. Completed: before the main-work temporal component, precommit only its
   strict parse,
   date/year concordance, null/bucket, snapshot, and descriptive metadata-order
   policies. Cohort cutoffs, maturation buffers, work-type rules, and author
   eligibility remain separate and must be frozen before extraction or G0.
2. Completed: create the current-content baseline; this is a first
   current-byte baseline only.
3. Completed: run the resumable v3 main-work temporal projection audit.
4. Next: bind and build the separately specified canonical work-time spine
   after the full temporal pass. This is an optimization parent, not another
   temporal pass.
5. Audit reference-edge multiplicity and chronology, then reconstruct fixed
   horizon outcomes with explicit open-world target accounting.
6. Resolve the author-source decision and run work-side author-history
   diagnostics.
7. Only then materialize candidate researchers and `t-1` graphs for the
   outcome-blind field-structure screen.

These are CPU, RAM, disk, and sequential-I/O tasks. GPU work begins later at
NABM fitting, repeated factorial simulation, and sealed-holdout evaluation.

## Claim boundary

The linked v3 result validates the bounded main-work date fields for the
current snapshot, while this inventory identifies the remaining author-source
blocker. It does not validate an author entity, establish edge observation
time, choose a maturation rule, construct an eligible panel, pass G-1 or G0,
run empirical G1, select fields, or establish a scientific effect.
