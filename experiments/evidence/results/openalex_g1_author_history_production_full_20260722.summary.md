# OpenAlex G-1 work-side author history production result

- Status: `PASS_CURRENT_SNAPSHOT_WORK_SIDE_AUTHOR_TOKEN_LINKED_RETROSPECTIVE_HISTORY_ACCOUNTING_AND_SENSITIVITY_COMPONENT_AUTHOR_ENTITY_AND_PAST_OBSERVABILITY_UNVERIFIED`
- Scope: `FULL_CURRENT_SNAPSHOT_PRODUCTION`
- GPU used: `false`
- Main source: 493 files, 492,293,207 raw rows, 492,293,206 retained rows
- Authorship source: 1,363 files, 1,361,551,038 raw rows, 1,361,551,037 retained rows
- Null author-ID slots: 580,116,419
- Grammar-valid non-null author-ID slots: 781,434,618
- Distinct author-work pairs: 779,630,258
- Duplicate-pair row excess: 1,804,360
- History-usable author-work pairs: 776,803,175
- Distinct author-ID tokens across all events: 113,748,578
- Tokens with at least one history-usable event: 112,897,930
- Author-year rows: 310,943,172
- Receipt-bound artifacts: 24,196
- Output receipt SHA-256: `7b3613eaac6900f376a999a551f15e20dd184de2b4dc6758a60d93f3eedb51a8`
- Result fingerprint: `73a3da022a1dc9373e8941f794cafe7b0a4c5239846783e693a661c02e053b7e`

## Closure result

The full run joined all 492,293,206 retained main works to the exact canonical
work-time spine. Authorship-to-main unmatched slots and `authors_count`
mismatches were both zero. One exact `W-1` row was quarantined in each source
surface, malformed non-null author IDs were zero, and the 781,434,618 valid
non-null slots closed as 779,630,258 distinct pairs plus 1,804,360 duplicate
rows.

The canonical temporal dispositions yielded 776,803,175 history-usable pairs.
Team-size sensitivity retained 660,330,519 pairs at cap 10, 750,120,927 at cap
50, and 755,307,596 at cap 100. The source contained no `A0` event, so the
base and positive-author-ID sensitivity grids are identical in this snapshot.

## Independent verification

The read-only verifier did not import the producer and opened no OpenAlex
source payload. It rehashed all 24,196 receipt artifacts, checked the exact
output file set and staging record sizes, reopened all 4,096 event/year/token
partition triples, recomputed every published reduction and sensitivity cell,
and checked source-accounting, work-event, deduplication, temporal-disposition,
and event-history cross-surface closures. The output tree remained unchanged.

The verifier does not independently replay the source projection or the
work-partition join from OpenAlex payloads. Those operations are established by
the receipt-bound producer checkpoints and cross-checked against the retained
staging and published event surfaces, not by a second production scan.

## Claim boundary

This passes the bounded work-side author-token accounting component only.
OpenAlex author IDs are tokens, not verified people; split/merge truth and
author-master completeness remain unresolved. The histories and current field
annotations are retrospective current-snapshot proxies, not evidence that the
same information was observable at a historical decision cutoff. Candidate
eligibility, citation chronology, cutoff panels, outcome-blind field structure,
the representative discovery pilot, G-1, G0, empirical G1, an NABM effect, and
any scientific effect remain pending or not established.
