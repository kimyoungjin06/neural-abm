# Pivot-feedback holdout seal tooling closed fixture

Status:
`PASS_CLOSED_FIXTURE_HOLDOUT_SEAL_TOOLING_PRODUCTION_NOT_RUN`

The v54-authorized create-only seal tool and independent read-only receipt
verifier pass a two-file, 39-byte closed fixture. Neither tool imports a row
data library. The creator requires a sorted explicit allowlist with expected
sizes and SHA-256 values, atomically finalizes the private sealed tree, and
writes the receipt last. The verifier independently rehashes the manifest and
every sealed byte and rejects unlisted tree members.

Ten tests also cover hash drift, symlinks, allowlist order and uniqueness,
create-only replay, receipt chronology, sealed-byte mutation, extra files, and
false opening claims.

This is synthetic tooling validation only. No production preflight exists, no
production path or byte was enumerated or hashed, and no production seal or
receipt was created. G0 remains unsealed; seal opening, model fitting, and G4
remain unauthorized.
