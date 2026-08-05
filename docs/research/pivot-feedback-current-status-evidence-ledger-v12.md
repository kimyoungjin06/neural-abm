# Pivot-feedback current-status and evidence ledger v12

Date: 2026-08-05

Status:
`CURRENT_HUMAN_FACING_INDEX_PRIVATE_G4_EXPLICIT_LIST_PREFLIGHT_INDEPENDENTLY_VERIFIED_SEAL_NEXT_NO_F_PASS`

## Immutable predecessors

The machine-readable authority remains
[`pivot_feedback_current_state_v1.yaml`](../../experiments/configs/pivot_feedback_current_state_v1.yaml),
SHA-256
`52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd`.
Ledgers v1-v11 and their production predecessors remain immutable. This v12
ledger indexes the separately authorized explicit-list preflight successor. It
does not create a new scientific gate, issue or open a byte seal, derive a
label, pass F or G0, or open G4.

## Current-byte preflight

The
[`G4 private explicit-list preflight`](../../experiments/evidence/results/pivot_feedback_f_g4_private_explicit_list_preflight_v1_20260805.summary.md)
consumed only the exact root and 153 entries fixed by the v11 binding. The
production implementation opened every listed regular file with no-follow
semantics, checked stable device, inode, size, and modification time across the
read, and hashed all 3,067,046,872 bytes. All computed sizes and SHA-256 values
matched the bound expectations.

The preflight is create-only, mode `0400`, under the mode-`0700` operator
control directory outside the source root. Its SHA-256 is
`ba70eb0b8fb84ffbfec9e8f94b35daa444b3dc579d45fae794a4961d960f8148`
and its internal fingerprint is
`70ada641f274c03c370aafbfef1e51b2c4bcc68e9209ee834281c275a5cdf681`.

## Independent full-byte verification

A separate verifier independently repeated the full 153-file,
3,067,046,872-byte pass. It found zero size mismatches, zero SHA-256 mismatches,
and zero files changed during hashing. It also confirmed the exact bound-list
projection, root identity, repository bindings, preflight fingerprint, and
direct compatibility with the existing seal creator. The verification control
is create-only, mode `0400`, with SHA-256
`3166b4df6c9f0689971376be5f2ea7be740809cb05c3e17520f0e8c215fc6874`.

Both passes treated the files as bytes only. They performed no implicit
artifact discovery, Parquet or row parsing, label derivation, analytic outcome
read, model scoring, or fitting. The observed state is therefore
`holdout_preflight_complete=true` and
`independent_preflight_verification_complete=true`, while
`holdout_byte_seal_complete=false`.

## Why this control is proportionate

This component does not add another empirical threshold or decision gate. It
closes one narrow chain-of-custody risk left deliberately open by v11: whether
the bytes present immediately before sealing still match the already bound
list. The independent second pass checks the preflight implementation rather
than changing the scientific estimand. Repeating more audits of this same
stage would add little; the workflow should now advance to the seal successor.

## Next bounded action

Exactly one next artifact is authorized: create the byte seal from the exact
verified preflight and independently verify its receipt. That successor may
consume the preflight allowlist but may not open the seal or parse analytic
rows. Seal opening, label derivation, analytic reads, model scoring, and fitting
remain later and separately authorized actions. Earlier-partition author-cold
guard application is still pending until those rows exist. `pass_f=false`,
`pass_g0=false`, and `g4_open=false`.
