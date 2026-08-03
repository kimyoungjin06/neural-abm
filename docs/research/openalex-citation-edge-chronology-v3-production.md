# OpenAlex Citation Edge Chronology v3 Production Successor

Status: `FULL_PRODUCTION_COMPONENT_PASS_WITH_BIBLIOGRAPHIC_EVENT_TIME_LIMITATION`

This is the production successor to the two frozen synthetic citation
chronology contracts. It does not edit or promote either predecessor. The new
config is
`experiments/configs/openalex_g1_citation_available_at_v2.yaml`; the producer
and independent verifier are
`scripts/audit_openalex_citation_available_at_v3.py` and
`scripts/verify_openalex_citation_available_at_v3.py`.

## What the component can establish

The input edge is directed from the citing work (`id`) to the cited work
(`referenced_works`). Exact duplicate source-target rows are counted once in
the graph product while their raw multiplicity is preserved. Every unique pair
is assigned to exactly one first-true bucket in this order:

1. self-loop;
2. source temporal unusable;
3. external open-world target;
4. target temporal unusable;
5. cited date after citing date;
6. same-day;
7. positive lag.

Only same-day and positive-lag internal, non-self pairs enter the primary
chronology-safe proxy product. Thirty-day, 365-day, and annual-only rules are
reported as sensitivity counts and cannot replace the primary zero-day rule.

`available_at` is the citing work publication date under a
`BIBLIOGRAPHIC_EVENT_TIME_CONVENTION`. It is not an OpenAlex first-observation
timestamp, evidence that the edge was indexed at publication, or evidence that
a researcher could observe the edge by a historical cutoff. Current-snapshot
external targets remain open-world and receive no fabricated target date.

## Exact parent chain

The production mode is available only because the canonical parent now has a
real receipt. The runner validates and cross-binds all three parents before it
opens a reference payload:

| Parent | Receipt SHA-256 |
| --- | --- |
| current content baseline | `082161a29bcd8fd79fd85e0bf6814a5c7950b97b847ea08056e17eebcd369a12` |
| temporal field-validity v3 | `2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe` |
| canonical work-time spine | `2760b3aa5c5960d9356cc30339338418ff9000ab22e9c4b94caf82f6216f9aa1` |

The canonical result fingerprint is
`58492b9182d9b213c0616ce5d29d69bd82714167f57619885dc013dd0219144d`,
and its spine data-tree commitment is
`a5e14799ed8da2366c126219ba79e6b83b07ed605cb1270e17494f5cfb70975d`.
The citation component may consume that spine but may not create, repair, or
silently substitute it.

## CPU/I/O execution model

This is not a GPU workload. The runner builds a temporary 28.6 GB direct
lookup from the 492,293,206-row canonical spine, then projects all 3,044
reference Parquet files. The lookup encodes canonical date32 plus temporal
disposition in one uint32 value. It removes the earlier direct-mode requirement
to rescan and reparse the 150,062,449,219-byte main table.

Raw pairs are partitioned by citing-work numeric ID modulo 4,096. Each shard is
sorted and deduplicated exactly, classified, and written to a receipt-bound
deduplicated-edge Parquet. Chronology-safe pairs are repartitioned by cited-work
numeric ID modulo 4,096 and written in cited/source order for the later cutoff
panel. Temporary lookup and binary spool files are deleted before publication.

The final private output contains:

- 4,096 exact deduplicated directed-edge partitions;
- 4,096 chronology-safe partitions by cited work;
- 3,044 source-scan checkpoints;
- 4,096 pair checkpoints and 4,096 safe-product checkpoints;
- multiplicity, lag-bin, open-world coverage, and cutoff-sensitivity tables;
- two partition manifests plus plan, state, report, Markdown, and a receipt
  written last.

Detailed work IDs are operator-local with directory mode `0700` and file mode
`0600`. Repository-facing evidence may contain only aggregates and hashes.

## Execution

A prefix run is permanently nonpromotable. Full execution requires all three
authorization flags and distinct source/output devices:

```bash
uv run python scripts/audit_openalex_citation_available_at_v3.py \
  --source-root /operator/local/openalex-derived \
  --content-receipt /operator/local/content/content_freeze_audit.outputs.json \
  --temporal-receipt /operator/local/temporal/main_temporal_validity_audit.outputs.json \
  --canonical-root /operator/local/canonical-work-time-spine \
  --output-dir /local/ssd/openalex-g1-citation-chronology-v3-full \
  --execute \
  --authorize-full \
  --acknowledge-exclusive-source-scan
```

Independent verification rehashes every receipt artifact, decodes every
deduplicated and chronology-safe row, independently recomputes the classifier,
lag arithmetic, multiplicity bridges, cross-tabs, and sensitivity rules, and
checks both sorted partition products:

```bash
uv run python scripts/verify_openalex_citation_available_at_v3.py \
  --output-dir /local/ssd/openalex-g1-citation-chronology-v3-full
```

## Full production result: 2026-07-22

The full run passed after scanning all 3,044 source files, 3,042,536,160
reference rows, and 37,119,669,321 source bytes. Exact reduction found
3,042,536,160 unique directed pairs and no duplicate groups. The primary
zero-day chronology product contains 2,858,679,472 safe pairs: 5,820,320
same-day and 2,852,859,152 positive-lag pairs. The remaining exclusive buckets
are 1,072,993 self-loops, 903,389 source-temporal-unusable pairs, 137,937,261
external open-world pairs, 4,472,749 target-temporal-unusable pairs, and
39,470,296 negative-lag pairs.

The final receipt binds 19,438 artifacts and 49,118,338,001 bytes. Its SHA-256
is `fbe753c84ceb3858a06c627f318eb756e997f73577498b06a5e88f72cacabe76`.
An independent full-artifact and full-row verifier passed after scanning both
4,096-partition products. The compact repository evidence is
`experiments/evidence/results/openalex_g1_citation_chronology_production_full_20260722.summary.json`.

## Remaining boundary

Even a full component PASS cannot select a fixed-horizon outcome, resolve
external-target chronology, reconstruct historical indexing, prove past
researcher visibility, or materialize the researcher-level `t-1` panel.
Maturation and allowed work types still require a separately hashed G0 policy.
Author and candidate validity, outcome-blind field structure, representative
discovery, G-1, G0, empirical G1, NABM fitting, and scientific effects remain
separate gates.
