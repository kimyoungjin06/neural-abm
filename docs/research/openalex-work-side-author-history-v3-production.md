# OpenAlex work-side author history v3 production component

Status: `PASS_BOUNDED_AUTHOR_TOKEN_ACCOUNTING_ONLY`

Production execution: `COMPLETED_2026-07-22`

The machine-readable contract is
`experiments/configs/openalex_g1_work_side_author_history_v3.yaml`. The
producer is `scripts/audit_openalex_work_side_author_history_v3.py`, the
read-only verifier is
`scripts/verify_openalex_work_side_author_history_v3.py`, and the compact
repository result is
`experiments/evidence/results/openalex_g1_author_history_production_full_20260722.summary.json`.

## What this component establishes

This component answers a narrow data-engineering question: can every
grammar-valid OpenAlex author-ID token in the frozen work-side authorship
surface be linked to the exact canonical work-time spine, deduplicated as an
author-work event, and reduced into reproducible annual and token-level
publication histories?

The answer is yes for the current snapshot. The unit is an OpenAlex author-ID
token, not a natural person. A passing result establishes current-snapshot
token accounting and deterministic reductions. It does not establish correct
person identity, split/merge truth, author-master completeness, or what an
observer knew at a historical cutoff.

## Bound data flow

```text
content receipt + temporal-v3 receipt + canonical-spine receipt
                              |
          +-------------------+-------------------+
          |                                       |
   main projection                         authorship projection
   work, team, field                       work, author/null
          |                                       |
          +------------ 4,096 work shards --------+
                              |
          exact main/canonical and authors_count closure
                              |
                 distinct author-work events
                              |
                    4,096 author shards
                              |
        +---------------------+----------------------+
        |                     |                      |
  event Parquet       author-year history    author-token history
        +---------------------+----------------------+
                              |
              receipt-last terminal validation
                              |
                 independent read-only verifier
```

The main projection retains work ID, declared team size, and the current
primary-field ID. The production source uses the actual 26-field OpenAlex
grammar, field IDs 11 through 36. The authorship projection retains work ID and
author ID, with null author slots represented only for accounting. Exact
`W-1` rows are quarantined before numeric parsing.

Within each work shard, the runner requires full main-to-canonical closure,
zero unmatched authorship slots, and exact equality between declared
`authors_count` and raw authorship-slot counts. Valid non-null author-work rows
are globally deduplicated. Canonical temporal dispositions 2 and 3 are the
only history-usable events. The author shards are then sorted by author, year,
and work before annual and token reductions.

## Full result

| Quantity | Value |
| --- | ---: |
| Main raw / retained rows | 492,293,207 / 492,293,206 |
| Authorship raw / retained rows | 1,361,551,038 / 1,361,551,037 |
| Null author-ID slots | 580,116,419 |
| Valid non-null author-ID slots | 781,434,618 |
| Unmatched slots / team-count mismatch works | 0 / 0 |
| Distinct author-work pairs | 779,630,258 |
| Duplicate-pair excess rows | 1,804,360 |
| History-usable pairs | 776,803,175 |
| Distinct author tokens across all events | 113,748,578 |
| Tokens with usable history | 112,897,930 |
| Author-year rows | 310,943,172 |
| Receipt artifacts | 24,196 |

The usable-pair team-cap counts are 660,330,519 at cap 10, 750,120,927 at cap
50, 755,307,596 at cap 100, and 776,803,175 without a cap. These are
sensitivity diagnostics, not exclusion decisions. The snapshot contains no
`A0` author event, so the base and positive-ID grids are identical; this does
not validate `A0` as an entity in general.

The production receipt SHA-256 is
`7b3613eaac6900f376a999a551f15e20dd184de2b4dc6758a60d93f3eedb51a8`,
and the result fingerprint is
`73a3da022a1dc9373e8941f794cafe7b0a4c5239846783e693a661c02e053b7e`.
Large event and history Parquet files remain operator-local; only aggregate
counts, commitments, and claim boundaries enter the repository.

## Independent verification boundary

The verifier does not import the producer. It checks the exact output file
set, rehashes all 24,196 receipt artifacts, validates staging record sizes,
reopens all 4,096 event/year/token triples, independently recomputes the annual
and token reductions and every sensitivity cell, and checks the source-to-work,
deduplication, temporal-disposition, event, `A0`, and cap-ALL aggregate
closures. It observed an unchanged output tree and emitted
`PASS_READ_ONLY_INDEPENDENT_AUTHOR_HISTORY_OUTPUT_REDUCTION_VERIFICATION`.

The first full-output verifier attempt exposed a verifier-only defect: Arrow
reader chunk boundaries changed the semantic digest even though Parquet bytes,
schema, and values were unchanged. The verifier now removes reader-specific
chunk boundaries before fixed-size digest batching, and a regression test
covers that invariance. The production artifacts were not rewritten.

This verifier intentionally opens no production source payload and does not
independently replay the source projection or work-partition join. Those steps
are supported by receipt-bound staging/checkpoints and cross-surface closures,
not a second full scan. A future stronger audit could add a separately pinned
source replay if the cost is justified.

## Operational limits

The runner is CPU/I/O work and uses no GPU. It requires distinct source and
output devices, at least 64 GiB available memory, 256 GiB free output space,
and an open-file soft limit of at least 4,224. Full execution additionally
requires both explicit authorization flags.

A completed receipt supports terminal read-only re-entry. A failed partial run
does not resume in place: the current v3 implementation requires a fresh
output directory. This is a reproducibility and operational-cost limitation,
not a result limitation, and should be addressed before routine repeated runs.

## Remaining scientific boundary

Author IDs remain current-snapshot tokens. Current primary-field annotations
and reconstructed publication histories are retrospective proxies, not
past-observable features. Candidate eligibility must be constructed inside a
separate cutoff-aware panel without using future history. Citation chronology
is outside this component; the subsequent production successor passed only a
current-snapshot bibliographic-time proxy. Outcome maturation, outcome-blind
field structure, and a representative discovery pilot also remain outside this
component.

Accordingly, the G-1 registry now records this component as passed but remains
`G_MINUS_1_PENDING` with three missing components: cutoff panel, outcome-blind
field structure, and representative discovery pilot. G0, empirical G1, an NABM
effect, and any scientific effect remain unestablished.
