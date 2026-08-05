# Pivot-feedback current-status and evidence ledger v14

Date: 2026-08-05

Status:
`CURRENT_HUMAN_FACING_INDEX_F_ATOMICALLY_CLOSED_E_READY_G4_SEALED_UNOPENED`

## Active machine authority

The active machine-readable authority is now
[`pivot_feedback_current_state_v2.yaml`](../../experiments/configs/pivot_feedback_current_state_v2.yaml),
SHA-256
`ca431afb63da4cf2e09d0563c0509e3685efac29dc8679df0f9cc7f1ae7851ab`.
The v1 authority, ledgers v1-v13, legacy gate files, and all component receipts
remain immutable provenance.

## One aggregate F closure

The independent
[`current-state v2 replay`](../../experiments/evidence/results/pivot_feedback_current_state_v2_20260805.summary.md)
bound and verified the exact conjunction required by F:

| Transition receipt | Verified state |
| --- | --- |
| `PRE_G1_DECISION_REGISTRY_COMPLETE` | 13 outcome-blind choices hash-bound |
| `THIRTEEN_PRE_G1_REQUIRED_SECTIONS_COMPLETE` | 13 of 13 sections frozen |
| `OUTCOME_BLIND_FINAL_PANEL_CONFORMANCE` | 65 aligned episodes independently conformed |
| `ATOMIC_PARTITION_AND_HOLDOUT_BYTE_SEAL` | Four partitions assigned; 153-artifact private seal independently verified and unopened |

All receipt hashes and statuses matched. No receipt remains pending, no
predecessor was rewritten, and the closure read zero operator, row-level,
confirmatory, or holdout files. It added no micro-gate: the four intermediate
receipts are the inputs to the single atomic F milestone already declared by
the active authority.

F is therefore complete with `pass_f=true`. Since active F is the compacted
replacement for legacy G0 and G0.5, its compatibility flag is
`pass_g0=true`. This compatibility passage is a freeze-and-seal readiness
result, not an empirical signature, predictive result, causal result, or
scientific effect. Immutable G0 v4 remains unchanged as lineage.

## Current routing and hard boundaries

The active route is now E with state `READY_NOT_RUN`. D, C, and F are complete;
E, M, and H are incomplete. H has advanced only to `BYTE_SEALED_UNOPENED`.

The closure itself did not run G1, derive or read a confirmatory entry or
persistence label, inspect a realized effect, read a model-selection score,
open the private seal, fit a model, or open G4. Confirmatory access is allowed
only through a separate exact E execution successor. Holdout access remains
blocked until M is resolved and the model and scoring contract is byte-locked.

## Next bounded action

The next artifact is an E/G1 execution input preflight and contract for the
frozen 2013-2016 replication partition. It must bind current-state v2, the
frozen estimands and decision rules, exact non-holdout input artifacts, label
materialization rules, negative controls, and output schemas before the first
confirmatory read. It may not include G4 artifacts or open the private seal.

No empirical G1 result exists yet. `empirical_g1_status=NOT_RUN`,
`model_fitted=false`, `g4_open=false`, and
`scientific_effect=NOT_ESTABLISHED`.
