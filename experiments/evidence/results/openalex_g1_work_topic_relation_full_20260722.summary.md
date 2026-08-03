# OpenAlex Work-Topic Relation Full Audit — 2026-07-22

- Status: `BLOCKED_WORK_TOPIC_RELATION_COMPONENT`
- Scope: 1,423 files, 1,420,855,753 footer rows
- Selected events: 1,420,855,752 after the frozen one-row `W-1` quarantine
- GPU used: `false`
- Component pass: `false`

## What passed

The full CPU/I/O run bound all 493 main and 930 topics files to the exact
current-content receipt and revalidated every projected payload and checkpoint.
It found 928,562,546 raw topic rows and 928,561,783 unique work-topic pairs.
All 354,145,248 retained non-null primary-topic rows had a corresponding child
topic edge. There were no orphan topic-work groups, duplicate main-work groups,
unexpected malformed work IDs, or topic-code-zero rows.

The bounded W0 structural sensitivity also passed: excluding the retained W0
row changed only the main-work and null-primary counts by one. W0 entity
validity remains `UNRESOLVED`.

## Why the component is blocked

The raw relation contains 714 duplicate work-topic groups and 763 excess rows,
with maximum multiplicity three. Most duplicates have the same score, but two
groups contain different source scores:

| Work | Topic | Scores | Source rows |
| --- | --- | --- | --- |
| `W5401631` | `T13112` | `0.0805`, `0.0806` | `delta-12.parquet`, rows 527787–527788 |
| `W26888826` | `T10455` | `0.7346`, `0.7347` | `delta-41.parquet`, rows 256211–256212 |

A targeted readback confirmed both pairs directly in the source Parquet files.
The frozen policy forbids silently choosing the first, maximum, mean, or another
score. The exact-score deduplicated relation was therefore not materialized and
the component correctly failed closed as `BLOCKED_SCORE_CONFLICT`.

## Evidence integrity

- Report SHA-256: `88aa8fa4d854e508c7e513b63f3240002a1ae19e25a1248ebf5e4fb2a08798f5`
- Receipt SHA-256: `7cf91e50a3934f783d278ae47b3764ec650b7d36c3625bfc4564da405a551507`
- Result fingerprint: `ce13bc8aa2e07578c84c1f540d2f21f7e46b18189ddda12c963a2ccc154db6d5`
- Task fingerprint: `edf3261950461461fc6f2a339c8d63899443b620b71aeb1aa254ed85dd940e9d`
- Checkpoint tree: `8cac1795cf72a3d659d2526cfb3178e79328cbb26bf359e8e69e8463f88a645f`
- Receipt artifacts independently rehashed: `1,430 / 1,430`
- Receipt-bound bytes independently rehashed: `68,205,416,394`

The separate read-only rehash imported none of the producer code and verified
the receipt artifact set, path/size/SHA-256 contracts, report/state bindings,
directory mode `0700`, file mode `0600`, and absence of symlinks.

## Claim boundary and next decision

This result establishes current-snapshot multiplicity, source/work
conservation, primary-child inclusion, and the bounded W0 structural delta. It
does not establish a promoted work-topic set because the two score conflicts
remain unresolved. A successor policy must be frozen before any new
materialization: either scores are scientifically required and require explicit
adjudication, or pair membership is the estimand and scores move to a
nonblocking conflict sidecar with sensitivity checks. Neither choice may be
made implicitly from this blocked run.

G-1 and G0 remain `PENDING`; empirical G1 is `NOT_RUN`; no field selection,
NABM estimate, or scientific effect is established.
