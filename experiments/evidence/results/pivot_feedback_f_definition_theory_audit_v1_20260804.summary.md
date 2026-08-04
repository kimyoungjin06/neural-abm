# Pivot-feedback F definition and theory audit v1 summary

Status: `PASS_OUTCOME_BLIND_DEFINITION_THEORY_AUDIT_ALIGNED_PRIMARY_RESCREEN_AUTHORIZED_NO_F_PASS`

The production audit replayed 1,887,623 episode-window target works and
7,235,450 linked author events for the 194 previously blocked structural
episodes. An independent raw-source SQL path matched all 388 aggregate output
rows and closed both the aggregate author-linked counts and the strict
all-incumbent counts against the frozen final-panel parent.

| Work category | Reference | Recent | Recent minus reference |
| --- | ---: | ---: | ---: |
| All linked authors are pre-signal incumbents | 39,507 | 31,221 | -8,286 |
| Mixed incumbent/new teams | 139,434 | 243,767 | 104,333 |
| No pre-signal incumbent | 315,277 | 866,278 | 551,001 |
| Aggregate author-linked total | 494,218 | 1,141,266 | 647,048 |

Works with no pre-signal incumbent account for 85.16% of net aggregate growth.
The strict closed-team share falls from 7.99% to 2.74%. Of 194 aggregate-screen
episodes, 143 reverse before normalization, 51 retain positive strict growth,
15 pass only when those strict counts are compared with the old aggregate
median/MAD, and 19 have at least one final peer-leaveout surge opportunity.
Peer leaveout creates four marginal episode passes and kills none; focal
leaveout remains exactly zero.

The audit retains `CLOSED_PRE_SIGNAL_INCUMBENT_TEAM_ATTENTION_V1` as the
primary estimand so that the observed block does not trigger a result-driven
switch to mixed-team inclusion. It forbids aggregate median/MAD reuse and
authorizes one next artifact: an all-topic strict-incumbent replay that
normalizes the primary construct within its own field-by-year universe while
preserving `z >= 3`, the three-family requirement, and the frozen field order.
Any-incumbent and fractional-incumbent variants are frozen as report-only
sensitivities and cannot rescue the primary panel.

Independent verification status:
`PASS_INDEPENDENT_DEFINITION_THEORY_AUDIT_RAW_COMPOSITION_REPLAY_NO_F_PASS`.

`pass_f=false`, `pass_g0=false`; no confirmatory label, realized effect,
holdout byte seal, model fit, or G4 analytic outcome was opened.
