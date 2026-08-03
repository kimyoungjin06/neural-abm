# Pivot-feedback holdout seal preflight tooling v1

Date: 2026-07-31

Status:
`PASS_CLOSED_FIXTURE_EXPLICIT_PREFLIGHT_TO_SEAL_CHAIN_PRODUCTION_NOT_RUN`

## Result

The operator preflight author is implemented under the v55 scope. It accepts
only explicitly repeated relative file paths. It does not accept globs,
wildcards, directories, recursive discovery, symlinks, duplicate paths,
noncanonical paths, or an existing output file.

For each explicit regular file, the tool opens the byte stream with
no-follow semantics, records the size and SHA-256 digest, verifies that the
file did not change during hashing, sorts the canonical allowlist, and writes
the uniquely named operator-local preflight create-only with mode `0400`.
The operator must explicitly attest that the artifact set is complete, the
source root is private and access controlled, no row parser was used, no prior
open event occurred, and no confirmatory or holdout row was read.

## Closed-fixture chain

A two-file, 27-byte fixture passed the complete preflight -> seal creation ->
independent receipt-verification chain. An additional file in the source root
was not discovered or included. Fourteen tests cover the full chain,
wildcards, directory input, symlinks, duplicate paths, each mandatory
attestation, nonzero access counts, create-only output, source-root symlinks,
and read-only output permissions.

This is synthetic tooling evidence only. No production source root or
artifact list was supplied, no production path or byte was enumerated or
hashed, and no production preflight, seal, receipt, or independent
verification result exists. G0, seal opening, model fitting, and G4 remain
closed.
