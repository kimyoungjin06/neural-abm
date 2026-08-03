# Pivot-feedback holdout-access seal contract v1

Date: 2026-07-31

Status:
`PASS_METADATA_ONLY_HOLDOUT_SEAL_CONTRACT_PARTITION_FROZEN_CREATION_AND_OPENING_SEPARATE`

## Result

The contract fixes the future G4 holdout boundary without touching any
holdout artifact. It binds G0 v2 and the prerequisite closure plan exactly and
freezes entry years 2020--2022, outcome follow-up through 2025, and the
domain- and author-cold rule.

## Separation and atomicity

Contract validation, seal creation, independent verification, and seal
opening are four distinct phases. This pass completes only contract validation.
A later successor gate is required before paths may be enumerated, artifact
bytes hashed, a manifest finalized, or a receipt issued. A separately
authorized G4 opening gate is required after that; neither the creation
receipt nor independent seal verification can open the holdout.

The future creation procedure must build an explicit canonical regular-file
manifest in a private staging area, close every SHA-256 binding, revalidate the
final atomic directory, and write the completion receipt last. The receipt
must use `pivot_feedback_holdout_access_seal_outputs.v1` with status
`PASS_CONFIRMATORY_PARTITIONS_SEALED_AND_UNOPENED`.

## Fail-closed boundary

Missing, changed, nonregular, symlinked, duplicate, noncanonical, or previously
opened artifacts produce no passing receipt. Partial staging is quarantined.
Failed independent verification invalidates the seal and cannot authorize an
opening.

Validation read four repository documents. It enumerated no holdout path,
hashed no holdout byte, opened no operator JSON or Parquet file, read no
confirmatory or holdout row, and made no API request. G0 remains unsealed;
model fitting and G4 remain unauthorized.
