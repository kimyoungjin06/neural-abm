# Pivot-feedback current-status and evidence ledger v13

Date: 2026-08-05

Status:
`CURRENT_HUMAN_FACING_INDEX_PRIVATE_G4_BYTE_SEAL_INDEPENDENTLY_VERIFIED_F_CLOSURE_NEXT_NO_F_PASS`

## Immutable predecessors

The machine-readable authority remains
[`pivot_feedback_current_state_v1.yaml`](../../experiments/configs/pivot_feedback_current_state_v1.yaml),
SHA-256
`52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd`.
Ledgers v1-v12 and their production predecessors remain immutable. This v13
ledger indexes the separately authorized byte-seal creation and receipt
verification successor. It does not pass F or G0, open the seal, derive an
analytic label, fit a model, or open G4.

## Production byte seal

The production
[`seal receipt`](../../experiments/evidence/results/pivot_feedback_holdout_access_seal_receipt_v1.summary.json)
binds the exact verified preflight SHA-256
`ba70eb0b8fb84ffbfec9e8f94b35daa444b3dc579d45fae794a4961d960f8148`.
The atomic creator copied and hash-checked the 153 allowlisted artifacts,
totaling 3,067,046,872 bytes, into
`/ssd/pivot-feedback-f-g4-private-operator-control-v1/pivot-feedback/holdout-seal-v1`.
It finalized and rechecked the manifest and final artifacts before writing the
receipt last.

The final private tree contains seven directories and 154 regular files: 153
sealed artifacts plus the manifest. It contains no symlink or extra file. All
directories are mode `0700`; every sealed artifact and the manifest are mode
`0400`. The manifest SHA-256 is
`7f28efe29942e413e9d14664e5167d9520b2ae24e75e7f03a9348af7bafbdfc7`,
and its fingerprint is
`1f035cbbbb7476b4d26b9bf3f363997d1b968913dbbcf3df0fa18f864a83196b`.
The create-only mode-`0444` receipt was written last and has SHA-256
`b0668308674c432aef798ba703fdaebdab622fa0d8ac390360ffef04f4db7851`.

## Independent receipt verification

The
[`independent verification summary`](../../experiments/evidence/results/pivot_feedback_holdout_access_seal_receipt_verification_v1_20260805.summary.md)
records a separate read-only replay. The verifier rehashed all 153 final sealed
artifacts and all 3,067,046,872 bytes, closed the exact tree to the manifest,
validated the manifest fingerprint and repository bindings, and confirmed that
the seal remains unopened.

The receipt deliberately records
`independent_verification_complete=false` because it is immutable creation-time
evidence written before the required verifier runs. The later verification
summary records completion without rewriting the receipt. Both creation and
verification performed zero Parquet or row parsing, zero confirmatory or
holdout analytic reads, zero model fitting, and zero seal-opening events.

## Why this is not another micro-gate

Seal creation plus its required independent replay closes the single
`ATOMIC_PARTITION_AND_HOLDOUT_BYTE_SEAL` transition receipt. The intermediate
preflight, manifest, and receipt are chain-of-custody controls, not independent
scientific thresholds. Continuing to add seal audits would be excessive.

The active authority defines F completion as one atomic milestone over four
transition receipts. Evidence now exists for the pre-G1 decision registry, all
thirteen required sections, outcome-blind final-panel conformance, and atomic
partition plus byte sealing. This ledger does not silently convert those
component results into `pass_f=true`.

## Next bounded action

The next artifact is a single aggregate F-closure successor. It must bind and
replay all four transition receipts, update the active milestone atomically,
and fail closed if any predecessor or hash has drifted. It may set `pass_f=true`
only if the complete conjunction passes; it may not open the private seal or
read a confirmatory row as part of closure.

Until that successor passes, E/G1 execution, analytic label derivation,
confirmatory access, model scoring, fitting, and G4 opening remain
unauthorized. `pass_f=false`, `pass_g0=false`, and `g4_open=false`.
