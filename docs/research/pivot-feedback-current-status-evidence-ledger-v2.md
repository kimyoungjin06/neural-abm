# Pivot-feedback current-status and evidence ledger v2

Date: 2026-08-03

Status: `CURRENT_HUMAN_FACING_INDEX_CURRENT_STATE_V1_F_IN_PROGRESS`

## Current authority

The machine-readable authority is now
[`pivot_feedback_current_state_v1.yaml`](../../experiments/configs/pivot_feedback_current_state_v1.yaml),
SHA-256
`52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd`.
Its independent replay is
[`pivot_feedback_current_state_v1_20260803.summary.json`](../../experiments/evidence/results/pivot_feedback_current_state_v1_20260803.summary.json),
SHA-256
`e58cd56086bfc919d76241c194d1b9d96051b969c5c00ced0e73f925757d32ce`.
If this human-facing index conflicts with that authority, the authority controls.

The former machine authority, gate v61, and the
[`v1 evidence ledger`](pivot-feedback-current-status-evidence-ledger-v1.md)
remain immutable provenance. They describe the state immediately before the
validated six-phase routing was activated; they are no longer the active
routing source.

## Active route

| Phase | Legacy gates | Current state | Claim ceiling |
| --- | --- | --- | --- |
| D | G-1A, S0 | Complete in bounded source and structure roles | Source and structure readiness only |
| C | D0 | Complete for discovery measurement and description | No association, causality, or transport |
| F | G0, G0.5 | **In progress** | Freeze and seal only |
| E | G1 | `NOT_RUN` | Bounded observational signature only if passed |
| M | G2, G3 | `NOT_RUN` | Predictive or generative adequacy only if passed |
| H | G4 | Logically reserved, not byte-sealed, unopened | Bounded noncausal transport only if passed |
| Optional causal lane | G5 | `NOT_RUN` | Design-specific causality only if separately passed |

## Decision and section staging

The pre-G1 decision registry is complete: the twelve outcome-blind choices in
G0 v4 and the independently validated one-cohort-year G1 replication tolerance
are hash-bound as one thirteen-choice registry. This is a policy freeze, not an
empirical result, power guarantee, or claim that one year is scientifically
true.

Six choices are intentionally deferred and do not block F:

- three G2 predictive thresholds, which must be frozen before the first G2
  model-score read and may not depend on the observed G1 effect magnitude; and
- three G3 simulation-budget choices, which must be frozen before G3 is run.

All thirteen pre-G1 scientific sections are now outcome-blind frozen and
independently replayed in the
[`F pre-G1 scientific-sections v1 receipt`](../../experiments/evidence/results/pivot_feedback_f_pre_g1_scientific_sections_v1_20260803.summary.md).
This is a provenance component, not a new micro-gate and not an F pass. Because
F is promoted atomically, the active authority continues to report all thirteen
sections as pending until final-panel conformance and the atomic partition plus
holdout byte seal are jointly verified. The `g2_predictive_gate` and
`g3_simulation_gate` sections remain deferred to their registered M-stage
boundaries and do not block F.

## One remaining F milestone

F is one atomic completion milestone, not a chain of new micro-gates. The
active authority records one of its four required receipts as complete:
`PRE_G1_DECISION_REGISTRY_COMPLETE`. The thirteen-section component described
above is also validated, but it is not promoted as a standalone transition.

The remaining execution work is:

1. build and verify final-panel conformance; and
2. assign atomic partitions, materialize the private holdout operator-only,
   run preflight, and write the byte seal last.

Intermediate receipts preserve lineage but do not create new decision gates.
F passes only when the frozen sections, panel conformance, and seal are jointly
verified and an additive authority successor records the atomic transition.

## Hard stops and claim boundary

`pass_f=false` and `pass_g0=false`. No confirmatory or holdout analytic row has
been read, empirical G1 is `NOT_RUN`, no model has been fitted, and G4 remains
unopened. F-phase work may construct and seal private partitions without
analytically opening them; E/G1 access remains prohibited until F passes.

Nothing in this transition establishes an attention-entry association, an H3
timing gap, peer learning, predictive added value, NABM necessity, holdout
transport, causality, or a scientific effect.
