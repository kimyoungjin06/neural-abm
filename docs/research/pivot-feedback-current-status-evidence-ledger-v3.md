# Pivot-feedback current-status and evidence ledger v3

Date: 2026-08-03

Status: `CURRENT_HUMAN_FACING_INDEX_F_FIELD_EPISODE_COMPONENT_COMPLETE_PANEL_AND_SEAL_PENDING`

## Current authority

The machine-readable authority remains
[`pivot_feedback_current_state_v1.yaml`](../../experiments/configs/pivot_feedback_current_state_v1.yaml),
SHA-256
`52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd`.
The v1 and v2 ledgers remain immutable provenance. This v3 ledger records an
F-phase component result but does not promote F or replace the active
authority.

## Completed outcome-blind assignment component

The previously missing exact topic taxonomy was recovered as the registered
4,516-row `b000000.parquet`, SHA-256
`566facdbc02bf4a37783fe64db4c1d104d00b73e250999bd824d560083827ba5`.
The exact field/episode run then scanned the full registered cutoff-work and
author-event inputs and was independently verified in the
[`field/episode assignment receipt`](../../experiments/evidence/results/pivot_feedback_f_field_episode_assignment_v1_20260803.summary.md).

The frozen domain-disjoint slots are:

| Role | Domain | Nominal field | Selected field | Structural result |
| --- | --- | ---: | ---: | --- |
| G1 | Physical sciences | 25 | 17 | rank-2 reserve activated; 5 families |
| G1 | Social sciences | 32 | 32 | nominal retained; 3 families |
| G1 | Life sciences | 28 | 11 | rank-2 reserve activated; 7 families |
| G4 operator-only | Health sciences | 36 | 36 | nominal retained; 3 families; sealed/author-cold assignment only |

The first attempt stopped before writing output when nominal field 25 failed
the registered three-family structural requirement. The implementation was
then reconciled to the pre-existing two-reserve rule without changing the
threshold, field ranking, temporal partitions, or any scientific outcome.
This was contract execution, not result-based substitution.

## What remains in F

This component is not a new gate. F remains one atomic completion milestone.
The next authorized component is final-panel conformance: replay the aggregate
shock candidates with the registered focal-author and preexisting-peer
leaveouts and verify all support requirements. Only after that passes may the
operator-only private partitions be materialized, preflighted, and byte-sealed
with the receipt written last.

## Hard stops and claim boundary

`pass_f=false` and `pass_g0=false`. Final-panel conformance is incomplete, the
private holdout byte seal does not exist, confirmatory entry or persistence
labels have not been computed, realized citation effects have not been read,
no model has been fitted, and G4 remains analytically unopened.

The assignment establishes neither an attention-entry association, an H3
timing gap, peer learning, predictive added value, NABM necessity, holdout
transport, causality, nor a scientific effect.
