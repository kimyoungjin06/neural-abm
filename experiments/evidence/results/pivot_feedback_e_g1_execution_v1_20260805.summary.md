# E/G1 execution v1 terminal result

Date: 2026-08-05

Execution status:
`BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION`

Independent verification:
`PASS_INDEPENDENT_VERIFICATION_OF_TERMINAL_BLOCKED_E_G1_NO_CONFIRMATORY_READ`

## Result

The actual E/G1 execution attempt stopped before the first confirmatory read.
The frozen scientific contract requires every G1 fit row to be anti-joined
against the exact G4 author-cold guard. The active E execution contract
simultaneously excludes every private G4 artifact, forbids G4 field/year input,
and forbids opening the seal. The exact guard has not been applied to an earlier
analytic row set because that row set did not yet exist.

These requirements cannot be satisfied together. Reconstructing the guard,
reading it from the private package, omitting it, or silently changing the
author-cold rule would each violate a frozen boundary. The registered terminal
BLOCKED outcome was therefore emitted without redesign.

## Verified access accounting

| Action | Count or state |
| --- | ---: |
| Confirmatory rows read | 0 |
| Holdout rows read | 0 |
| Private G4 artifacts read | 0 |
| Labels materialized | 0 |
| Estimators run | 0 |
| Bootstrap refits run | 0 |
| Model-selection scores read | 0 |
| Seal opens | 0 |
| `g4_open` | `false` |

The production package at
`/ssd/pivot-feedback-e-g1-execution-v1-20260805` contains seven receipt-bound
artifacts plus the receipt written last. All six Parquet result tables contain
zero rows with the registered schemas; no empty table is represented as a null
or failed scientific estimate.

## Consequence

`empirical_g1_status=BLOCKED`, `pass_e=false`, and `pass_g1=false`. No E result,
association, null, timing signature, model result, transport result, causal
claim, or scientific effect exists. Resolving the conflict requires new
explicit authority that says how the exact author-cold exclusion may be applied
without silently opening or reconstructing the G4 holdout.
