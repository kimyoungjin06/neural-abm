# Pivot-feedback G0 G-1A binding reconciliation

Status:
`PASS_METADATA_ONLY_G_MINUS_1A_BINDING_RECONCILIATION_FUTURE_G0_V2_ONLY_NO_RELABEL`

The metadata-only contract is independently valid. It defines a distinct
future `g_minus_1a_binding` over the exact existing G-1A receipt.

The contract forbids treating G-1A as an alias of legacy G-1, relabeling its
status, or inferring a legacy G-1 pass. Legacy G-1 remains pending. All 14
unresolved receipt boundaries and the exact receipt claim boundary are
preserved.

A future binding-only G0 v2 must preserve eight G0 v1 subtrees and may make
only two semantic changes: retain the legacy binding as pending immutable
lineage and add the exact G-1A bounded-evidence binding. The holdout binding,
17 unresolved values, 15 not-freeze-ready sections, and `pass_g0 = false`
remain unchanged.

Validation read nine repository documents and no operator JSON, Parquet,
row-level, confirmatory, or holdout data. It made no API request. This contract
does not author G0 v2 or authorize a G0 seal, model fitting, or G4.
