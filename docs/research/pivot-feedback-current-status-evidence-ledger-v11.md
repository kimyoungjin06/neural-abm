# Pivot-feedback current-status and evidence ledger v11

Date: 2026-08-05

Status:
`CURRENT_HUMAN_FACING_INDEX_EXACT_PRIVATE_G4_ROOT_AND_COMPLETE_LIST_BOUND_PREFLIGHT_NEXT_NO_F_PASS`

## Immutable predecessors

The machine-readable authority remains
[`pivot_feedback_current_state_v1.yaml`](../../experiments/configs/pivot_feedback_current_state_v1.yaml),
SHA-256
`52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd`.
Ledgers v1-v10 and their production predecessors remain immutable. This v11
ledger indexes the separately authorized root/list binding successor. It does
not create a new scientific gate, run a preflight, issue a byte seal, derive a
label, pass F or G0, or open G4.

## Exact root and complete explicit list

The
[`G4 private root and artifact-list binding`](../../experiments/evidence/results/pivot_feedback_f_g4_private_root_artifact_binding_v1_20260805.summary.md)
binds the resolved operator-local root
`/ssd/pivot-feedback-f-g4-private-feature-source-v1-20260804`, its path digest,
filesystem device and inode, and its mode-`0700` identity.

The explicit list is the exact sorted projection of all 153 source-artifact
entries in the independently verified production receipt. Their expected byte
total is 3,067,046,872 and their expected artifact-tree digest is
`62d6978c776c32464c4c862df12a52a580a6066dda45d4fb0dfe75910a562084`.
The receipt itself is control metadata and is the sole root file outside that
153-entry source list.

## Independent binding verification

Independent metadata verification found exactly six directories including the
root and 154 regular files: 153 bound artifacts plus the production receipt.
Every listed path is canonical, sorted, unique, regular, and size-matched; the
actual tree has no missing, extra, nonregular, or symlinked path. Directory
modes are `0700`, source-file modes are `0600`, and the create-only binding and
verification controls are mode `0400` outside the source root.

Expected per-artifact SHA-256 values were inherited without refresh from the
production receipt whose artifacts already passed independent full-source
replay. The binding and its verifier did not open or hash source-artifact
content and did not parse Parquet or any row. Source-artifact content files
opened, source-artifact bytes hashed, operator Parquet files opened, and
confirmatory or holdout analytic rows read all remain exactly zero.

This distinction is deliberate: root/list binding fixes the chain-of-custody
inputs, while preflight must independently rehash the current bytes. Collapsing
the two would make it unclear which artifact established current byte identity.

## Next bounded action

Exactly one next artifact is authorized: an operator-only explicit-list
preflight driven solely by the bound root and 153 entries. It must reject any
root-identity, path, size, or digest drift and independently hash every listed
artifact byte without parsing rows.

Byte-seal creation remains unauthorized until that preflight succeeds. Seal
opening, label derivation, analytic reads, model scoring, and fitting remain
later and separately authorized actions. Earlier-partition author-cold guard
application is still pending until those rows exist. `pass_f=false`,
`pass_g0=false`, and `g4_open=false`.
