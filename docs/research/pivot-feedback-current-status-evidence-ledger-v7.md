# Pivot-feedback current-status and evidence ledger v7

Date: 2026-08-04

Status: `CURRENT_HUMAN_FACING_INDEX_ALIGNED_FINAL_PANEL_CONFORMED_ATOMIC_PARTITION_NEXT_NO_F_PASS`

## Immutable predecessors

The machine-readable authority remains
[`pivot_feedback_current_state_v1.yaml`](../../experiments/configs/pivot_feedback_current_state_v1.yaml),
SHA-256
`52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd`.
Ledgers v1-v6 and every production predecessor remain immutable. This v7
ledger records the separately authorized aligned final-panel replay; it does
not relabel the blocked aggregate-normalized replay, pass F, pass G0, or open a
holdout.

## Final-panel leaveout now conforms

The
[`aligned final-panel conformance replay`](../../experiments/evidence/results/pivot_feedback_f_aligned_final_panel_conformance_v1_20260804.summary.md)
uses only the 65 retained candidates and fields 16, 32, 11, and 27 from the
verified all-topic strict-incumbent screen. Each replayed base count closes to
the aligned parent. The field-year median and MAD come from the same strict
all-topic universe; no aggregate normalization is reused.

| Slot | Field | Episodes | Risk rows | Leaveout episodes | Leaveout families |
| --- | ---: | ---: | ---: | ---: | ---: |
| Physical G1 | 16 | 8 | 4,118,455 | 8 | 3 |
| Social G1 | 32 | 3 | 2,064,365 | 3 | 3 |
| Life G1 | 11 | 17 | 23,948,527 | 17 | 7 |
| Health G4 operator-only | 27 | 37 | 217,839,267 | 37 | 16 |

Every episode exceeds the frozen 500-row outcome-blind support floor and every
selected field retains the required three leaveout-surge subfield families.
The aligned final panel therefore conforms without field replacement,
post-result reserve fallback, threshold relaxation, aggregate normalization,
or a team cap.

## Why the leaveout gate did not bind

The result is not that leaveout performed no work. Of 247,970,614 focal risk
rows, 40,753,340 have at least one qualifying target-active prior peer. The
network rule excludes 2,229,480 attention-work incidences, and all 65 episodes
show some minimum-count change.

The effect on the screening decision is nevertheless small. Only 7,182 risk
rows, or 0.00290%, cease to satisfy the aligned surge rule; 247,963,432 rows,
or 99.9971%, retain it. Nineteen episodes contain any non-surge risk row, but
the median adjusted reference/recent counts change in zero episodes. Each of
the 65 candidates retains at least one surge opportunity, so its family
remains represented.

Focal leaveout incidence is exactly zero for a definitional reason: risk-set
eligibility requires no pre-entry exact-target-topic work, whereas every
strict attention work is authored only by pre-signal exact-topic incumbents.
The focal check is therefore a consistency invariant. The empirical part of
this replay is the prior-peer exclusion, which changes counts but does not
eliminate a selected family.

## Independent verification and next bounded action

The independent verifier matches all 65 aligned parent rows, recomputes their
acceleration and modified z algebra, verifies strict-count closure, checks
support and leaveout monotonicity, reconstructs the four family counts as
3, 3, 7, and 16, and confirms that raw author/work identifiers are absent. It
returns
`PASS_INDEPENDENT_ALIGNED_FINAL_PANEL_LEAVEOUT_AND_SUPPORT_CONFORMANCE_NO_F_PASS`.

Exactly one next artifact is authorized: atomic assignment of the conformed
panel to its already registered G1, model-selection, and operator-only G4
partitions. Private holdout materialization, preflight, and byte seal follow
only after that successor; they are not performed here.

`pass_f=false` and `pass_g0=false`. No entry or persistence label, realized
citation effect, model score, private holdout byte, or G4 analytic outcome was
read. This component establishes structural final-panel conformance only, not
an association, null, prediction, transport claim, causality, or effect.
