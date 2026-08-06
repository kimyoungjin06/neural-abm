# Lean H3 OpenAlex source intake v1

Date: 2026-08-06

Status:
`PASS_MANIFEST_ONLY_NEWER_PUBLIC_SNAPSHOT_CANDIDATE_BOUND_NO_DATA_DOWNLOAD_NO_ROW_ACCESS`

## Outcome

The free public OpenAlex Parquet manifest dated 2026-06-26 is a real source
candidate newer than the current lineage's 2026-03-30 input. Six manifest
files were captured under
`/ssd/pivot-feedback-lean-h3-openalex-source-intake-v1-20260806` and immediately
refetched. Every first capture and refetch was byte-identical. The directory is
`0700` and all captured manifests are `0600`.

The live URLs are mutable. This candidate is defined by the locally captured
manifest bytes and SHA-256 values in the JSON receipt, not by whatever those
URLs may contain after a future quarterly refresh.

## Manifest inventory

| Entity | Role | Records | Files | Manifest content length |
| --- | --- | ---: | ---: | ---: |
| Works | H3 core | 510,372,821 | 2,446 | 724,970,323,127 bytes |
| Authors | Conditional identity/ORCID sensitivity | 119,129,660 | 1,961 | 52,769,910,496 bytes |
| Topics | Taxonomy core | 4,516 | 6 | 2,589,362 bytes |
| Subfields | Taxonomy core | 252 | 1 | 174,417 bytes |
| Fields | Taxonomy core | 26 | 1 | 19,238 bytes |

The combined manifest contains 21 entities, 649,096,577 aggregate entity
records, and 783,555,949,110 bytes. Works plus taxonomy is approximately
724.97 GB; adding Authors raises the manifest total to approximately 777.74
GB. The observed `/ssd` free space was 3,023,754,072,064 bytes, so a raw copy
fits, but transformation and independent-replay headroom has not been
established.

## Dependency decision

Works and the Topics/Subfields/Fields hierarchy are the minimum declared H3
candidate. Authors remains conditional until a metadata/schema preflight
determines whether Work authorships alone can support the author-token history
and whether ORCID/identity sensitivity requires the separate Authors entity.
All peer, institution, citation-success, funding, and full-text dependencies
remain outside the lean H3 core unless that preflight proves a direct need.

## What is not yet established

No Parquet data file or source record was read. The manifests do not establish:

- the Works Parquet schema needed by the existing pipeline;
- historical or calendar-year completeness;
- complete H3 `k=0,1,2` plus `t+1...t+3` maturation;
- eligible disjoint discovery or confirmatory episodes;
- estimator-specific precision or support; or
- sufficient disk headroom for raw data, transformed products, and replay.

Therefore this is a candidate-intake pass, not source-readiness, preregistration,
or execution passage.

## Next bounded action

The next action is a no-row or footer-only schema/dependency preflight and an
explicit transfer/storage budget. It must decide the exact entity prefixes,
whether Authors is mandatory, the resumable download method, and required
working-space multiplier before any source-data download begins.

Current G1, model-selection, G4, and holdout rows remain unauthorized and
unread. No seal was opened, `g4_open=false`, and no model or scientific effect
was established.
