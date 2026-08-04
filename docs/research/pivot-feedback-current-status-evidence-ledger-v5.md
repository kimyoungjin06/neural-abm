# Pivot-feedback current-status and evidence ledger v5

Date: 2026-08-04

Status: `CURRENT_HUMAN_FACING_INDEX_DEFINITION_AUDIT_COMPLETE_ALIGNED_RESCREEN_NEXT_NO_F_PASS`

## Immutable predecessors

The machine-readable authority remains
[`pivot_feedback_current_state_v1.yaml`](../../experiments/configs/pivot_feedback_current_state_v1.yaml),
SHA-256
`52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd`.
Ledgers v1-v4 and the blocked final-panel production tree remain immutable.
This v5 ledger records the authorized post-block definition-and-theory audit;
it does not relabel the v1 replay, pass F, pass G0, or open a holdout.

## Reproduced composition diagnosis

The
[`definition-and-theory audit`](../../experiments/evidence/results/pivot_feedback_f_definition_theory_audit_v1_20260804.summary.md)
replayed the raw cutoff work and author-event sources for all 194 blocked
topic-year attempts. It classified each episode-window work into four mutually
exclusive observation states. The author-linked states close exactly to the
aggregate parent, and the all-incumbent state closes exactly to the strict
final-panel parent.

| Work state | Reference | Recent | Net change |
| --- | ---: | ---: | ---: |
| Every linked author is a pre-signal incumbent | 39,507 | 31,221 | -8,286 |
| Mixed incumbent/new team | 139,434 | 243,767 | 104,333 |
| No pre-signal incumbent | 315,277 | 866,278 | 551,001 |
| Aggregate author-linked total | 494,218 | 1,141,266 | 647,048 |

The aggregate screen therefore selects a work-growth construct whose net
increase is 85.16% attributable to works with no pre-signal incumbent. The
registered closed-team primary construct declines in total and its aggregate
share falls from 7.99% to 2.74%. This is a construct/universe diagnosis, not an
entry outcome or scientific null.

The stage replay also locates the remaining loss. All 194 attempts pass the
aggregate screen; 143 have strict recent counts no greater than their strict
reference counts; 51 retain strict positive growth; and only 15 pass when the
strict acceleration is compared with the old aggregate field-year median/MAD.
Network leaveout kills none of those 15 and creates four marginal passes.
Focal leaveout is zero by the risk-set anti-target definition, and the median
network-adjusted counts are unchanged in all 194 attempts.

## Theory decision

The primary estimand remains
`CLOSED_PRE_SIGNAL_INCUMBENT_TEAM_ATTENTION_V1`: a work counts only when every
linked author entered the exact current-snapshot target topic before the
reference window. This conservative choice is retained after the block because
switching the primary to any-incumbent mixed-team inclusion now would be
result-contingent relaxation. It measures legacy-only team production, not all
field attention or all attention visible to a focal researcher.

The audit separately freezes two report-only definition sensitivities before
the next structural result: any-pre-signal-incumbent work and fractional
pre-signal-incumbent author share. Neither may replace the primary estimand,
select the panel, or rescue F.

## Normalization repair and next bounded action

The upstream scientific contract defines incumbent attention normalized across
eligible topics within primary-field by decision-year groups. The blocked
implementation instead inserted strict incumbent counts into median/MAD values
computed from aggregate counts. The audit now forbids that cross-universe
normalization.

Exactly one next artifact is authorized: compute the strict incumbent primary
measure for all eligible topics, calculate each field-year median/MAD in that
same strict universe, and rerun topic/field eligibility in the already frozen
nominal-primary and reserve order. The inclusive `z >= 3` threshold, positive
raw acceleration, recent-greater rule, three current-snapshot subfield-family
minimum, and field order remain unchanged. Aggregate-screen reuse as primary,
threshold relaxation, and fields outside the frozen order remain unauthorized.

## Verification and hard stops

The independent verifier rescanned the raw source partitions using a separate
SQL formulation, matched all 388 composition rows, verified the exact parent
closures, confirmed raw author/work IDs are absent from the output, and
returned
`PASS_INDEPENDENT_DEFINITION_THEORY_AUDIT_RAW_COMPOSITION_REPLAY_NO_F_PASS`.

`pass_f=false` and `pass_g0=false`. No confirmatory entry or persistence label,
realized citation effect, model score, private holdout byte, or G4 analytic
outcome was read. The audit establishes why the frozen structural panel failed
and licenses a construct-aligned structural rescreen; it establishes neither a
null nor a positive association, prediction, transport, causality, or
scientific effect.
