# Pivot-feedback G4 private explicit-list preflight v1 summary

Status:
`PASS_OPERATOR_ONLY_AND_INDEPENDENT_BOUND_153_ARTIFACT_BYTES_REHASHED_PREFLIGHT_NO_SEAL_NO_F_PASS`

The production preflight consumed only the exact bound private source root and
its sorted 153-entry artifact list. It opened every listed regular file with
no-follow semantics and hashed all 3,067,046,872 bytes. Every current byte size
and SHA-256 matched the binding, and no file changed while it was being read.

An independently implemented verifier then repeated the full 153-file,
3,067,046,872-byte pass. It confirmed the exact list projection, source-root
identity, preflight fingerprint, all repository bindings, and zero size, hash,
or during-read-change mismatches. Neither pass used implicit discovery, parsed
Parquet or rows, or read a confirmatory or holdout analytic outcome.

The create-only preflight and verification controls are mode `0400` under the
mode-`0700` operator-control directory. The existing byte-seal creator accepts
the preflight directly as its expected schema, status, partition, repository
bindings, and 153-entry allowlist.

This is a chain-of-custody control, not a new scientific gate. It establishes
current byte identity immediately before sealing; it does not create or open a
seal, derive labels, score or fit a model, pass F or G0, or open G4. The next
bounded action is byte-seal creation from this exact verified preflight. Seal
opening remains separately unauthorized. `pass_f=false`, `pass_g0=false`, and
`g4_open=false`.
