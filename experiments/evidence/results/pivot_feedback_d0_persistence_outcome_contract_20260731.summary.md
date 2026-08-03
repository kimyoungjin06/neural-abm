# Pivot-feedback D0 persistence outcome contract v1

Status:
`PASS_INDEPENDENT_D0_PERSISTENCE_OUTCOME_CONTRACT_AND_FIXTURE_REPLAY_NO_PRODUCTION_ROW_READ`

Persistence is defined only for the 4,301 independently verified
`NEW_TOPIC_ENTRY` rows. A matured row persists when at least one distinct
ABI-eligible same-author, same-certified-topic work occurs in `t+1`, `t+2`,
or `t+3`. Nonprimary classifier states are not assigned a persistence zero.

The bound 2026-03-30 snapshot makes 2025 the last complete calendar year. All
four D0 episodes are mature because the latest required outcome year is 2015.
Incomplete windows remain `RIGHT_CENSORED`; even a visible partial positive
cannot turn their nullable count or boolean into a completed outcome.

An independent validator checked 11 repository bindings and three operator
receipts and replayed complete-window and censoring fixtures. The fixtures
covered `MATURED_PERSISTENCE`, `MATURED_NO_PERSISTENCE`, and
`RIGHT_CENSORED` with zero row and accounting mismatch.

No production classification or post-entry row and no Parquet data page was
opened. No API request, outcome interpretation, G0 freeze, model fitting, G4
opening, association, causal mechanism, or scientific effect was authorized or
established.
