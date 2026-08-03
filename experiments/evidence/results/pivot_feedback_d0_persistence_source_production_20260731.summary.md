# Pivot-feedback D0 persistence source production v1

Status:
`PASS_INDEPENDENT_EXACT_4301_THREE_YEAR_PERSISTENCE_SOURCE_ALL_ROW_REPLAY`

The production source retained all 4,301 observable-history
`NEW_TOPIC_ENTRY` rows. Under the frozen `t+1..t+3` current-snapshot work/topic
ABI, 1,487 rows have at least one subsequent target-topic work and 2,814 do
not, for an unadjusted descriptive rate of 34.57%. All rows are mature and
none are right-censored.

The four unadjusted episode rates are 30.42%, 39.69%, 36.69%, and 20.73%.
They must not be treated as comparable field effects because the topics,
calendar periods, and opportunity structures differ.

The DuckDB producer rehashed 4,096 work and 4,096 author-event partitions. A
separate PyArrow/NumPy verifier did the same, reconstructed the exact
denominator and 3,689 qualifying author-work pairs, and replayed every value in
all 4,301 output rows. Every key, count, state, and boolean mismatch count is
zero.

No confirmatory entry-year row, citation or impact outcome, peer exposure, API
result, model, or G4 data was used. This is a deterministic descriptive source,
not an association, causal effect, field ranking, final eligibility decision,
or scientific effect. A successor gate is required before any D0 calibration
readout beyond these source-accounting totals.
