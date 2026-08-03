# Pivot-feedback holdout seal creation tooling v1

Date: 2026-07-31

Status:
`PASS_CLOSED_FIXTURE_HOLDOUT_SEAL_TOOLING_PRODUCTION_NOT_RUN`

## Result

The v54-authorized creator and independent receipt verifier are implemented.
The creator requires a separate operator preflight containing a sorted,
explicit, content-hash-bound artifact allowlist. It refuses implicit
discovery, directories, symlinks, nonregular files, changed bytes, duplicate
or noncanonical paths, existing final output, and existing receipts.

The creator copies only allowlisted bytes with no row parser, rehashes the
private staging tree, writes a canonical manifest, atomically finalizes the
sealed directory, rehashes the final tree, and writes the receipt create-only
and last. The independent verifier reads the receipt and manifest, rehashes
every sealed artifact byte, and rejects any extra, missing, changed,
symlinked, or nonregular member.

## Closed-fixture validation

A two-file, 39-byte closed fixture passed creation and independent replay. The
test suite also rejected hash drift, a symlinked input, unsorted and duplicate
allowlists, a second creation attempt, a receipt timestamp preceding seal
creation, sealed-artifact mutation, an unexpected sealed-tree member, and a
receipt claiming that the seal had been opened.

The fixture is synthetic tooling evidence only. No production preflight was
provided, no production artifact path was enumerated, no production artifact
byte was hashed, and no production seal or receipt was created. G0 remains
unsealed; seal opening, model fitting, and G4 remain unauthorized.
