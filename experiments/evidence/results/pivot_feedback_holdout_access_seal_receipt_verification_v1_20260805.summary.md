# Pivot-feedback production holdout byte-seal receipt verification v1 summary

Status:
`PASS_PRODUCTION_PRIVATE_G4_BYTE_SEAL_RECEIPT_INDEPENDENTLY_VERIFIED_UNOPENED_NO_F_PASS`

The atomic seal creator consumed the exact verified preflight with SHA-256
`ba70eb0b8fb84ffbfec9e8f94b35daa444b3dc579d45fae794a4961d960f8148`.
It copied and hash-checked all 153 allowlisted artifacts totaling
3,067,046,872 bytes into a private final tree, finalized the manifest, checked
the final hashes, and wrote the create-only receipt last.

The final tree contains exactly 153 sealed artifacts plus the manifest, with no
symlink or extra file. All seven directories are mode `0700`; the sealed files
and manifest are mode `0400`. The receipt is mode `0444` and binds manifest
SHA-256
`7f28efe29942e413e9d14664e5167d9520b2ae24e75e7f03a9348af7bafbdfc7`.

The separate read-only verifier then rehashed all 153 sealed artifacts and all
3,067,046,872 bytes. It confirmed the exact tree, valid manifest fingerprint,
receipt bindings, zero prior open events, and zero row reads. The receipt's
internal `independent_verification_complete=false` records its intentional
pre-verification creation state; this summary records the later independent
verification result without mutating that immutable receipt.

The F phase now has evidence for all four transition receipts, but the atomic
F-closure reconciliation has not run. This seal component does not itself pass
F or G0, open the seal, materialize analytic labels, read outcomes, score or fit
a model, or open G4. The next bounded artifact is one aggregate F-closure
successor over all four receipts. `pass_f=false`, `pass_g0=false`, and
`g4_open=false`.
