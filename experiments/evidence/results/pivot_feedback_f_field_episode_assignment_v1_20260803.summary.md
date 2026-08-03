# Pivot-feedback F field and episode assignment v1

Date: 2026-08-03

Status: `PASS_OUTCOME_BLIND_FIELD_AND_EPISODE_ASSIGNMENT_COMPONENT_FINAL_PANEL_AND_SEAL_PENDING`

The exact outcome-blind production run scanned 492,293,206 cutoff-work rows
and 779,630,258 author-event rows, retained 176,124,383 linked eligible works,
and produced 1,744 aggregate attention-shock candidates. The independent
verifier rehashed the receipt and all five declared artifacts, recomputed all
26 field metrics and rankings, replayed the four domain slots and reserve
activation order, and verified all 1,744 candidate rows.

| Slot | Domain | Nominal | Selected | Frozen reserves | Activated | Candidates | Families |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| G1-01 | Physical sciences | 25 | 17 | 17, 16 | yes | 37 | 5 |
| G1-02 | Social sciences | 32 | 32 | 18, 20 | no | 18 | 3 |
| G1-03 | Life sciences | 28 | 11 | 11, 24 | yes | 40 | 7 |
| G4 author-cold | Health sciences | 36 | 36 | 29, 27 | no | 12 | 3 |

The first production attempt stopped fail-closed before writing any output
because nominal field 25 had fewer than three independent subfield families
in its assigned period. The implementation had not applied the already-frozen
two-reserve rule. The successor execution kept the threshold and ranking
unchanged, recorded both reserves, and applied the registered order: nominal,
domain rank 2, then domain rank 3. No outcome was read during this
reconciliation.

External receipt SHA-256:
`19550868b92a97c178917f6b0629e73a354cdce7508df880c94eacb782edfdc4`.
Independent verification SHA-256:
`163842e1c090e4331ebd2770d5ff1898269088c92a7f1b044f322839a773a478`.

This is an F-phase component, not a new gate and not an F pass. Final-panel
focal-author and preexisting-peer leaveout/support replay is next. Private
holdout materialization and byte sealing remain unauthorized until that replay
passes. `pass_f=false`, `pass_g0=false`, no confirmatory labels or effects were
read, no model was fitted, and G4 remains unopened.
