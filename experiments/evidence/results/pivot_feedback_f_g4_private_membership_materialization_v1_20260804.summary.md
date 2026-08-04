# Pivot-feedback G4 private membership materialization v1 summary

Status:
`PASS_OPERATOR_ONLY_G4_PREDECISION_MEMBERSHIP_AND_AUTHOR_COLD_GUARD_REMAINING_PRIVATE_PACKAGE_PENDING_NO_F_PASS`

This successor performs actual operator-only row materialization rather than
another metadata audit. It writes 217,839,267 predecision risk-measurement rows
for the 37 field-27 G4 episodes into 37 private Parquet files. Every episode's
risk and network-leaveout surge counts close exactly to the independently
verified final panel; 217,832,376 rows retain the registered surge and 6,891 do
not.

The exact distinct projection of those rows contains 8,174,698 current-snapshot
author tokens. It is materialized as an author-cold guard that must be
anti-joined from future discovery, G1-fit, and model-selection row sets. The
guard has no duplicate, missing, or extra token relative to the private risk
rows. It is not a natural-person identity claim.

The private tree contains 41 receipt-bound artifacts totaling 573,119,853
bytes. Its root and nested directories are mode `0700`; every file is mode
`0600`. Raw author tokens occur only inside that private tree. Raw work IDs,
entry or persistence labels, postdecision target events, realized citation
effects, and model scores are absent.

Independent verification returned
`PASS_INDEPENDENT_PRIVATE_G4_PREDECISION_MEMBERSHIP_AUTHOR_COLD_GUARD_NO_OUTCOME_READ_NO_F_PASS`.
It checked all 217,839,267 rows for parent closure, leaveout algebra, surge
recomputation, and key uniqueness, and verified the guard projection in both
directions.

This is the private membership foundation, not the complete G4 scoring
package. The remaining private predecision feature and postdecision source
materialization is the next authorized artifact. Earlier-row guard application
is pending because those analytic rows do not yet exist. Explicit-list
preflight, byte sealing, analytic reads, model fitting, and G4 opening remain
unauthorized. `pass_f=false`, `pass_g0=false`.
