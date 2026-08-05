# Pivot-feedback current-status and evidence ledger v15

Date: 2026-08-05

Status:
`CURRENT_HUMAN_FACING_INDEX_E_G1_INPUT_PREFLIGHT_VERIFIED_EXECUTION_NOT_RUN`

## Active authority remains E-ready

The active machine authority remains
[`pivot_feedback_current_state_v2.yaml`](../../experiments/configs/pivot_feedback_current_state_v2.yaml),
SHA-256
`ca431afb63da4cf2e09d0563c0509e3685efac29dc8679df0f9cc7f1ae7851ab`.
It records F complete, E `READY_NOT_RUN`, H `BYTE_SEALED_UNOPENED`, and
`empirical_g1_status=NOT_RUN`. The E/G1 preflight does not rewrite that state.

## The single E entry boundary

The independently verified
[`E/G1 execution-input preflight`](../../experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v1_20260805.summary.md)
implements the single E entry boundary already required before the first
confirmatory read. It is not another scientific gate and does not fragment E
into a chain of passable micro-gates. Its only role is to make the irreversible
execution boundary exact and auditable.

The registered execution scope is now fixed as follows:

| Item | Exact bound value |
| --- | --- |
| Partition | `g1_empirical_replication` |
| Entry years | 2013-2016 |
| Fields in registered slot order | 16, 32, 11 |
| Shock episodes | 28 |
| Field-subfield families | 13 |
| G4 field 27 | Excluded |
| Private G4 artifacts or manifest | Not an input |
| G4 seal | Unopened and not required for E execution |

## Exact input identity without analytic access

The preflight binds four operator metadata receipts. The two large normalized
input groups are committed transitively by the exact receipt SHA, the receipt's
artifact array, and an independently replayed canonical inventory fingerprint:

| Input group | Bound artifacts | Bound bytes |
| --- | ---: | ---: |
| Normalized work metadata | 4,096 | 4,120,921,809 |
| Author-history events | 4,096 | 5,006,392,019 |
| Final-panel metadata | 2 | 23,905 |
| Atomic-partition metadata | 3 | 15,943 |

No analytic Parquet file was opened to create or verify the preflight. The
partition predicate and field allowlist must be applied before label or
estimator logic in the execution successor, so scope cannot be widened after
outcomes are visible.

## Frozen scientific and output contract

The preflight binds the exact hashes of the thirteen pre-G1 scientific sections
and the G1 replication-tolerance contract. It fixes the entry and three-year
persistence labels, alternative pivot, ten-controls-per-case sampling and
weights, all seven negative controls, H3 and field-replication rules, the
registered bootstrap and multiplicity procedure, fail-closed support behavior,
and seven execution output schemas.

The three allowed execution terminal states are PASS, registered-decision FAIL,
or input/support/estimator BLOCKED. A failed or blocked result cannot be rescued
by changing a field, episode, threshold, control, estimand, or output after the
confirmatory read.

## Current boundary and next action

No G1 estimate or decision exists yet. `pass_e=false`, `pass_g1=false`,
confirmatory and holdout rows read remain zero, no label or negative control was
materialized, no estimator ran, no model-selection score was read, no predictive
or NABM model was fitted, `g4_open=false`, and
`scientific_effect=NOT_ESTABLISHED`.

The next bounded artifact is the separate exact E/G1 execution successor. It
may read only the bound G1 scope and inputs, must write its receipt last, and
must accept PASS, FAIL, or BLOCKED without post-access redesign. G4 artifacts,
model-selection scores, predictive or NABM fitting, causal promotion, and seal
opening remain outside that successor.
