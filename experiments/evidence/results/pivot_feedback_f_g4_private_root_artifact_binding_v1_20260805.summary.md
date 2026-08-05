# Pivot-feedback G4 private root and artifact-list binding v1 summary

Status:
`PASS_OPERATOR_ONLY_EXACT_PRIVATE_G4_ROOT_AND_COMPLETE_153_ARTIFACT_LIST_BOUND_NO_PREFLIGHT_NO_F_PASS`

The exact operator-local source root
`/ssd/pivot-feedback-f-g4-private-feature-source-v1-20260804` is now bound by
resolved path, path digest, filesystem device and inode, and mode. Its complete
source-artifact list is the sorted, unique, exact projection of the 153 entries
in the independently verified production receipt, totaling 3,067,046,872
expected bytes.

The binder and independent verifier enumerated the actual tree only to close
its paths, file kinds, sizes, and permissions. The tree contains six
directories including the root and 154 regular files: the 153 bound source
artifacts plus the production receipt control file. No symlink or extra file
was present. Directories remain mode `0700`, source files mode `0600`, and the
create-only binding and verification controls are mode `0400` outside the
source root.

Expected per-artifact SHA-256 values were copied without refresh from the
production receipt already covered by independent full-source replay. This
binding did not open or hash source-artifact content, parse Parquet or rows,
write a preflight, create a seal, derive a label, or read an analytic outcome.
Both source-artifact content files opened and source-artifact bytes hashed are
exactly zero.

This component closes chain-of-custody input resolution; it is not a new gate.
The next bounded artifact is an explicit-list preflight driven only from this
binding. That preflight must independently rehash all 153 bound artifact bytes.
Byte-seal creation remains unauthorized until the preflight succeeds.
`pass_f=false`, `pass_g0=false`, `g4_open=false`.
