# E/G1 exact guard-only exception v1

Date: 2026-08-05

Independent verification:
`PASS_INDEPENDENT_EXACT_GUARD_ONLY_EXCEPTION_G1_RESUME_AUTHORIZED`

The user-approved exception authorizes exactly one non-sealed private file,
`g4_author_cold_guard.parquet`, solely as a pre-label G1 exclusion set. Its
SHA-256, 8,174,698 rows, 33 row groups, schema, mode, and exact projection were
independently verified. Every projected `exclude_from_g1` value is true.

The exception does not authorize any other private G4 file, sealed-root access,
field/year reconstruction, use of guard membership as a feature or outcome,
scope or threshold changes, a seal open, G4 analytic open, model-selection
scores, model fitting, causality, or a scientific effect.

No confirmatory or holdout outcome row was read while validating the exception;
`g4_open=false` and the byte seal remained unopened.
