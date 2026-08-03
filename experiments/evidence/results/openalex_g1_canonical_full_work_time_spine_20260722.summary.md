# OpenAlex G-1 Canonical Work-Time Spine Full Result

- Status: `PASS_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_COMPONENT_WITH_TEMPORAL_V3_PARENT`
- Execution scope: `FULL_CURRENT_SNAPSHOT_PRODUCTION`
- GPU used: `false`
- Source files: `493`
- Source bytes: `150,062,449,219`
- Raw rows: `492,293,207`
- Frozen `W-1` quarantine: `1`
- Retained unique work IDs: `492,293,206`
- Duplicate numeric-ID groups/excess: `0 / 0`
- Published Parquet partitions: `4,096`
- Receipt-bound artifacts: `9,185`
- Output receipt SHA-256: `2760b3aa5c5960d9356cc30339338418ff9000ab22e9c4b94caf82f6216f9aa1`
- Result fingerprint: `58492b9182d9b213c0616ce5d29d69bd82714167f57619885dc013dd0219144d`

## Result

The fresh full run created a deterministic current-snapshot row-level
work-time spine bound to the exact content-freeze and temporal-v3 receipts.
All retained numeric work IDs were globally unique. Publication date/year
parse errors, mismatches, one-sided nulls, and invalid years were all zero.
The retained `W0` row is explicitly preserved as a sensitivity-policy issue;
its entity validity is not inferred from grammar validity.

## Independent verification

The separate repository-only verifier did not import the producer. In `full`
mode it rehashed all 9,185 receipt artifacts, decoded all 4,096 Parquet
partitions, reconstructed their row-record commitments, and observed an
unchanged output tree. Its terminal status was
`PASS_READ_ONLY_INDEPENDENT_OUTPUT_VERIFICATION`.

## Claim boundary

This passes only the canonical current-snapshot work-time-spine component.
Raw source-ID validity remains `BLOCKED`, `W0` entity validity remains
`UNRESOLVED`, historical content identity remains `UNVERIFIED`, and citation,
author, candidate, cutoff-panel, field-structure, representative-pilot, G-1,
G0, empirical G1, NABM-effect, and scientific-effect claims remain pending or
not established.
