# Pivot-feedback D0 production pivot classification v1

Status:
`PASS_INDEPENDENT_EXACT_FOUR_ALL_ROW_OBSERVABLE_HISTORY_STATE_REPLAY`

The frozen threshold-free classifier retained and classified all 3,256,271
registered author-episode rows: 3,246,984 `NO_FOCAL_ENTRY`, 4,301
`NEW_TOPIC_ENTRY`, 125 `RETURN`, and 4,861 `CONTINUATION`. The states are
exhaustive and mutually exclusive, with no population filtering or source
feature mutation.

The DuckDB producer read only the independently verified eight-column
three-count source. A separate PyArrow/NumPy verifier preserved all eight
source columns and recomputed the state and primary boolean on every row. It
found zero source-column, state, and boolean mismatches and did not import or
reuse the producer engine.

`NEW_TOPIC_ENTRY` is an observable-history state: zero prior observed
target-topic work and positive focal-year work under the frozen
current-snapshot ABI. It is not proof of a researcher's true lifetime-first
topic entry.

No raw work or author-event partition, post-entry persistence or impact row,
API result, model, or G4 result was opened. The classification establishes no
association, causality, or scientific effect. A successor gate is required
before even the design of the D0 persistence outcome source can proceed.
