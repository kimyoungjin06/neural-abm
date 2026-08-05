# Pivot-feedback current-status and evidence ledger v17

Date: 2026-08-05

Status:
`CURRENT_HUMAN_FACING_INDEX_E_G1_TERMINAL_BLOCKED_NO_CONFIRMATORY_READ`

## We moved from audit to execution

The E/G1 execution was started from the exact conjunction of preflight v1 and
the independently verified v2 completeness correction. It did not continue as
another audit cycle. The execution runner replayed the frozen rules and active
permissions before opening the first confirmatory row, as required by the
fail-closed contract.

## The execution found a hard authorization contradiction

The frozen atomicity rule requires the exact G4 author-cold guard to be
anti-joined from all earlier fit or selection rows, including G1. The verified
guard contains 8,174,698 current-snapshot author tokens and its application to
earlier rows was explicitly pending until those rows existed.

At the same time, the E input authority says that the private G4 root or
manifest is not an execution input, G4 field/year data are excluded, private G4
artifact access is not authorized, and the seal may not be opened. Thus the
runner could neither read the exact guard, reconstruct it, omit it, nor change
the author-cold rule while remaining compliant.

This is not weak evidence, estimator failure, or an adverse scientific result.
It is a logical conflict between two frozen requirements.

## Registered terminal outcome

The production execution package was written to
`/ssd/pivot-feedback-e-g1-execution-v1-20260805` with terminal status
`BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION`. Its receipt
was written last and independently verified. Six registered Parquet outputs are
schema-correct and empty; the report and receipt make clear that no estimate or
decision was produced.

| Boundary | Verified state |
| --- | --- |
| Confirmatory rows read | 0 |
| Holdout rows read | 0 |
| Private G4 artifacts read | 0 |
| Labels or controls materialized | 0 |
| Estimators or bootstrap refits | 0 |
| Model-selection scores | 0 |
| Seal opens | 0 |
| `g4_open` | `false` |

## Active state

The independently verified
[`current state v3`](../../experiments/evidence/results/pivot_feedback_current_state_v3_20260805.summary.md)
now records E as `BLOCKED_INPUT_AUTHORIZATION_CONFLICT_NO_CONFIRMATORY_READ` and
`empirical_g1_status=BLOCKED`. F remains complete, G0 compatibility remains
passed, H remains `BYTE_SEALED_UNOPENED`, and `pass_e=false`, `pass_g1=false`,
`model_fitted=false`, and `scientific_effect=NOT_ESTABLISHED`.

## What is required next

This cannot be repaired by another metadata check or by relaxing a threshold.
Progress requires one new explicit authority that resolves exactly one choice:
permit a narrowly scoped application of the already sealed author-cold guard to
G1 row materialization, or revoke/change the frozen author-cold exclusion rule.
The former preserves the scientific design; the latter changes it. Until that
choice is authorized, confirmatory access and route promotion remain blocked.
