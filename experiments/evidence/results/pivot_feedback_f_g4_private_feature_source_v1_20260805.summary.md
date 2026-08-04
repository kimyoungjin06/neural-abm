# Pivot-feedback G4 private feature and source v1 summary

Status:
`PASS_OPERATOR_ONLY_COMPLETE_G4_FEATURE_AND_POSTDECISION_SOURCE_NO_LABEL_NO_ANALYTIC_OPEN_NO_F_PASS`

The operator-only materialization now supplies the complete private G4 source
package for the 37 field-27 episodes. It contains 217,839,267 predecision
feature rows, 267,491,648 contributing-peer rows, 30,198,415 field-year public
success reference-work rows, 1,607,759 target-topic author-work events at
offsets t through t+3, and 3,392,785 complete work-byline rows.

Independent verification replayed the materialized columns from the frozen
membership and three normalized source families rather than importing the
producer. It reproduced all feature and relation rows, the inclusive-tie
public-success proxy, postdecision events, and bylines. All 14 output-row
algebra discrepancies and all five full-source-replay mismatch counts are
zero. The verification status is
`PASS_INDEPENDENT_COMPLETE_PRIVATE_G4_FEATURE_AND_POSTDECISION_SOURCE_NO_LABEL_NO_ANALYTIC_OPEN_NO_F_PASS`.

The receipt binds 153 artifacts totaling 3,067,046,872 bytes under a mode
`0700` private tree with mode `0600` files. Raw author and work tokens stay in
that operator-local tree. No entry or persistence label, descriptive outcome
rate, model score, fitted model, association, causal mechanism, prediction,
transport result, or scientific effect was produced. The public-success field
is a current-snapshot bibliographic event-time proxy; historical platform
observation and researcher visibility are not established.

This completes the private G4 *source set*; it is not a new gate and does not
pass F or G0. The next bounded artifact may bind this exact private root and
its complete explicit artifact list. Explicit-list preflight remains
unauthorized until that separate binding exists; byte sealing and analytic
opening remain later successors. Earlier-partition guard application is also
pending until those rows exist. `pass_f=false`, `pass_g0=false`,
`g4_open=false`.
