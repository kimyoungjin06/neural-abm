# Pivot-feedback current-status and evidence ledger v9

Date: 2026-08-04

Status:
`CURRENT_HUMAN_FACING_INDEX_G4_PRIVATE_PREDECISION_MEMBERSHIP_AND_AUTHOR_COLD_GUARD_COMPLETE_REMAINING_PACKAGE_NEXT_NO_F_PASS`

## Immutable predecessors

The machine-readable authority remains
[`pivot_feedback_current_state_v1.yaml`](../../experiments/configs/pivot_feedback_current_state_v1.yaml),
SHA-256
`52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd`.
Ledgers v1-v8 and all production predecessors remain immutable. This v9 ledger
indexes the separately authorized private G4 membership foundation; it does
not relabel that foundation as a complete scoring package, run a preflight,
issue a byte seal, pass F or G0, or open G4.

## Actual private rows now exist

The
[`G4 private membership materialization`](../../experiments/evidence/results/pivot_feedback_f_g4_private_membership_materialization_v1_20260804.summary.md)
replays the registered risk and preexisting-peer leaveout construction for all
37 final field-27 G4 episodes. It materializes 217,839,267 author-token by
episode rows in 37 private Parquet files.

All 37 episode risk counts close to the verified final panel. Of those rows,
217,832,376 retain the registered network-leaveout surge and 6,891 do not. The
zero focal exclusion incidence remains the risk-definition invariant already
identified in final-panel conformance; this successor adds row-level private
membership, not a new robustness claim.

## Author-cold guard

The distinct G4 risk-row projection contains 8,174,698 current-snapshot author
tokens. The private guard contains exactly that set: no duplicate, missing, or
extra token was found by independent bidirectional verification.

Every guard token is marked for exclusion from future discovery, G1-fit, and
model-selection rows. The guard itself is complete, but application to earlier
rows is pending because those analytic row sets do not yet exist. Calling that
future filtering complete now would overstate the bytes. The guard is an
OpenAlex author-token rule, not a verified natural-person identity boundary.

## Privacy and outcome boundary

The receipt binds 41 artifacts totaling 573,119,853 bytes under a mode-`0700`
private root with mode-`0600` files. Raw author tokens are repository-forbidden
and appear only inside the operator-local tree. No raw work ID, entry or
persistence label, postdecision target event, realized citation effect, or
model score was emitted. Confirmatory or holdout outcome rows read remains
exactly zero.

The independent verifier checks every private row's parent episode keys, base
counts, leaveout subtraction, surge Boolean, and `(episode, author)` uniqueness,
then verifies the author-cold projection in both directions. It returns
`PASS_INDEPENDENT_PRIVATE_G4_PREDECISION_MEMBERSHIP_AUTHOR_COLD_GUARD_NO_OUTCOME_READ_NO_F_PASS`.

## Next bounded action

Exactly one next artifact is authorized: remaining operator-only private
predecision feature and postdecision source materialization needed for the
complete G4 scoring package. Future earlier-row builders must anti-join the
frozen guard.

Explicit-list preflight is not authorized while the private package is
incomplete. Byte sealing follows a successful preflight successor; analytic
opening remains a separate later gate. `pass_f=false` and `pass_g0=false`. No
association, null, prediction, transport result, causality, or scientific
effect is established.
