# Pivot-feedback current-status and evidence ledger v8

Date: 2026-08-04

Status:
`CURRENT_HUMAN_FACING_INDEX_ATOMIC_EPISODE_FAMILY_PARTITIONS_ASSIGNED_PRIVATE_MATERIALIZATION_NEXT_NO_F_PASS`

## Immutable predecessors

The machine-readable authority remains
[`pivot_feedback_current_state_v1.yaml`](../../experiments/configs/pivot_feedback_current_state_v1.yaml),
SHA-256
`52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd`.
Ledgers v1-v7 and all production predecessors remain immutable. This v8 ledger
indexes the separately authorized atomic metadata assignment; it does not
materialize or open private rows, pass F, pass G0, or open G4.

## Atomic assignment now closes the final shock panel

The
[`atomic partition assignment`](../../experiments/evidence/results/pivot_feedback_f_atomic_partition_assignment_v1_20260804.summary.md)
binds all 65 verified final-panel episodes and 29 field-subfield families to
the registered event-year partitions without splitting or outcome-based
reselection.

| Partition | Entry years | Episodes | Families | Boundary |
| --- | --- | ---: | ---: | --- |
| Discovery/calibration | through 2012 | 0 | 0 | not a final shock-panel cell |
| G1 empirical replication | 2013--2016 | 28 | 13 | open only after F |
| Model selection | 2017--2019 | 0 | 0 | row-level folds deferred; no score read |
| G4 temporal holdout | 2020--2022 | 37 | 16 | sealed assignment, operator-only, author-cold |

All 13 G1 families and 16 G4 families are disjoint. Thirty-nine episodes have
one or more `k=0,1,2` response years outside their event partition's entry-year
interval; the frozen whole-episode rule retains every such cohort with its
event instead of moving or splitting it.

## Why model selection has zero selected shock episodes

This is a distinction between episode assignment and later analytic-row
materialization, not missing evidence. The conformed shock panel contains the
three empirical fields in 2013--2016 and the held-out field in 2020--2022. It
does not contain an event registered in 2017--2019, and this successor does not
fabricate one to make the table look balanced.

The frozen five-fold rule remains unchanged: connected components of author
token, topic, subfield, and shock episode are assigned without splitting using
the registered SHA-256 seed and balancing rule. Actual components and folds
cannot be truthfully emitted until authorized row-level materialization. No
model-selection score is authorized before E passes and the pre-G2 registry is
locked.

## G4 boundary and next bounded action

The G4 assignment contains field 27, 37 episodes, and 16 families and is marked
sealed, operator-only, and author-cold. Those are metadata bindings. This run
read zero author-token membership rows, enumerated zero holdout paths, and
hashed zero holdout bytes, so row-level author-cold enforcement and the private
artifact set do not yet exist.

Exactly one next artifact is authorized: operator-only materialization of the
private G4 artifact set with row-level author-cold enforcement. Explicit-list
preflight follows a successful materialization successor; byte sealing follows
a successful preflight successor. None is collapsed into this receipt.

`pass_f=false` and `pass_g0=false`. No entry or persistence label, realized
citation effect, model score, private holdout byte, or G4 outcome was read. The
component establishes partition provenance and atomicity only, not an
association, null, prediction, transport result, causality, or effect.
