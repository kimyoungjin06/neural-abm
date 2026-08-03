# Pivot-feedback holdout-access seal contract v1

Status:
`PASS_METADATA_ONLY_HOLDOUT_SEAL_CONTRACT_PARTITION_FROZEN_CREATION_AND_OPENING_SEPARATE`

The metadata-only contract is independently valid. It binds the exact G0 v2
and prerequisite closure plan and freezes the G4 entry years 2020--2022,
outcome follow-up through 2025, and domain- and author-cold rules.

Contract validation, future seal creation, independent seal verification, and
future seal opening are separate phases. Seal creation requires a successor
gate; opening requires a distinct G4 gate. The future procedure requires an
explicit canonical regular-file manifest, final SHA-256 closure, atomic
directory finalization, and a receipt written last.

Missing, changed, nonregular, symlinked, duplicate, noncanonical, or previously
opened artifacts fail closed without a passing receipt. A failed independent
verification invalidates the seal and cannot authorize opening.

Validation read four repository documents. It enumerated zero holdout paths,
hashed zero holdout bytes, opened no operator JSON or Parquet file, read no
confirmatory or holdout row, and made no API request. No seal or receipt was
created. G0 remains unsealed; model fitting and G4 remain unauthorized.
