# Lean H3 OpenAlex footer preflight v1

Date: 2026-08-06

Status:
`PASS_REPRESENTATIVE_REMOTE_FOOTER_SCHEMA_STABLE_WORKS_SUFFICIENT_FOR_LEAN_H3_AUTHORS_DEFERRED_NO_ROW_ACCESS_NO_DOWNLOAD`

## Outcome

The lean H3 source dependency can be narrowed to Works plus the tiny
Topics/Subfields/Fields taxonomy files. The earliest-partition, latest-partition,
and smallest-file samples from both Works and Authors were opened through
anonymous S3 random access. Only Parquet footer metadata and Arrow schemas were
inspected; no row value was decoded and no Parquet file was persisted locally.
The S3 client's internal range/cache transfer bytes were not independently
metered.

All three sampled Works files had the same schema SHA-256,
`2dcf41ac6cecb07d812d0d9348a6344481e5a093186c95857ed46ace11a30781`.
All three sampled Authors files also shared one schema SHA-256,
`f6a4c5ac1f90eaa6684a6b69b569963c36323c73a4718387b4841972ddf7d8fa`.
Every footer row count equaled its captured-manifest record count.

This is a representative three-file check per entity, not proof that all 2,446
Works files or all 1,961 Authors files have one schema.

## Why Authors is deferred

The sampled Works schema contains all lean H3 core primitives:

- `authorships.author.id` and `authorships.author.orcid`;
- `publication_date` and `publication_year`;
- `type`, `is_paratext`, and `is_retracted`; and
- `primary_topic.id` with nested domain, field, and subfield identifiers.

The separate Authors entity can therefore be deferred unless a later,
separately authorized identity-enrichment design requires names, affiliations,
or author-level summaries. This removes 52,769,910,496 bytes from the initial
raw transfer. The standalone taxonomy files remain in the core set because they
are tiny and provide an authoritative hierarchy binding.

## Exact transfer scope

| Item | Value |
| --- | ---: |
| Entities | Works, Topics, Subfields, Fields |
| Captured-manifest URLs | 2,454 |
| Raw content-length cap | 724,973,106,144 bytes |
| Working/transformed/replay cap | 1,087,459,659,216 bytes |
| Total reserved cap | 1,812,432,765,360 bytes |
| Free space observed at intake | 3,023,754,072,064 bytes |
| Margin after reserve | 1,211,321,306,704 bytes |

The 2.5-times total reserve is an operational cap, not a measured
transformation ratio or scientific threshold. Download must follow the exact
URLs in the captured manifests, not a mutable live prefix. The planned method
is a manifest-driven resumable range downloader using partial files and a
checkpoint. Each completed file must match the captured content length, then
receive a local SHA-256 in the immutable download inventory.

## Remaining boundary

The preflight does not establish full-file schema unanimity, historical or
calendar-year completeness, H3 horizon maturity, disjoint discovery or
confirmatory cohorts, or estimator-specific precision. It does not authorize
the full source download or any source row read. Current G1, model-selection,
G4, and holdout rows remain unauthorized and unread; no seal was opened and no
model or scientific effect was established.

The next bounded action is implementation and dry-run validation of the exact
manifest-driven resumable downloader. Starting the 724.97 GB transfer remains a
separate authorization decision.
