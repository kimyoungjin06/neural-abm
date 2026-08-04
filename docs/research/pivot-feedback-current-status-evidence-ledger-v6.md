# Pivot-feedback current-status and evidence ledger v6

Date: 2026-08-04

Status: `CURRENT_HUMAN_FACING_INDEX_ALIGNED_RESCREEN_COMPLETE_FINAL_LEAVEOUT_REPLAY_NEXT_NO_F_PASS`

## Immutable predecessors

The machine-readable authority remains
[`pivot_feedback_current_state_v1.yaml`](../../experiments/configs/pivot_feedback_current_state_v1.yaml),
SHA-256
`52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd`.
Ledgers v1-v5, the original aggregate assignment, the blocked final-panel
replay, and the definition-and-theory audit remain immutable. This v6 ledger
records their narrowly authorized aligned-rescreen successor; it does not
relabel an earlier result, pass F, pass G0, or open a holdout.

## The normalization repair was executed

The
[`aligned strict-incumbent rescreen`](../../experiments/evidence/results/pivot_feedback_f_aligned_strict_incumbent_rescreen_v1_20260804.summary.md)
recomputed the conservative closed-team measure over all eligible topics in
each frozen field-by-decision-year attempt. It then computed the median and MAD
inside that same strict universe. Aggregate counts and aggregate normalization
were not reused.

The production surface contains 7,099 positive-reference topic-year rows over
all 45 registered field-year groups. All groups have finite positive MAD. The
unchanged recent-greater-than-reference, positive acceleration, and inclusive
`z >= 3` rules produce 99 candidates over 70 topics. The raw replay also
matches all 194 predecessor strict reference and recent counts exactly.

## Taxonomy recovery is bounded and closed

The original 4,516-row projection file was no longer present at its frozen
path. A local Science Atlas table explicitly sourced from the 2026-03-30
OpenAlex primary-topic projection recovers exactly 4,516 topics, 26 fields,
and 252 subfields. It agrees on field, topic name, subfield, and subfield name
for every overlapping frozen mapping: 476 distinct aggregate mappings and 994
distinct D0 mappings. It maps all 70 new candidates. No current API taxonomy
was silently substituted.

## Frozen-order result

Every nominal and reserve field was evaluated, but selection still used the
first conforming field in each frozen order.

| Slot | Frozen order | First field with at least three families | Episodes | Families |
| --- | --- | ---: | ---: | ---: |
| Physical G1 | 25, 17, 16 | 16 | 8 | 3 |
| Social G1 | 32, 18, 20 | 32 | 3 | 3 |
| Life G1 | 28, 11, 24 | 11 | 17 | 7 |
| Health G4 operator-only | 36, 29, 27 | 27 | 37 | 16 |

Thus fields 16, 32, 11, and 27 supply 65 selected topic-year episodes over 45
field-topic pairs. Fields 25 and 17 do not reach three families before field
16; field 32 passes immediately; field 28 fails before field 11; and fields 36
and 29 fail before field 27. This is not threshold relaxation or a search
outside the frozen order.

## Independent verification and next bounded action

The independent verifier recomputed all 7,099 acceleration values, 45
field-year medians and MADs, 7,099 modified z-scores and candidate flags,
1,989 distinct surface topic mappings, all 12 attempt dispositions, all four
first-conforming selections, and all 194 predecessor count closures. It
returned
`PASS_INDEPENDENT_STRICT_INCUMBENT_ALIGNED_ALL_TOPIC_RESCREEN_NO_F_PASS`.

Exactly one next artifact is authorized: replay the outcome-blind focal and
preexisting-peer leaveout rules for the 65 selected episodes and reassess the
three-family rule within each selected field. Atomic partition assignment and
holdout materialization remain unauthorized until that final-panel
conformance succeeds.

`pass_f=false` and `pass_g0=false`. No confirmatory entry or persistence label,
realized citation effect, model score, private holdout byte, or G4 analytic
outcome was read. This artifact establishes an internally aligned structural
candidate panel; it establishes neither a scientific association nor a null,
prediction, transport claim, causality, or effect.
