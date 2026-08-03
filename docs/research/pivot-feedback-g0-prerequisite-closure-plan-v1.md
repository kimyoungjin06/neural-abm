# Pivot-feedback G0 prerequisite closure plan v1

Date: 2026-07-31

Status:
`PASS_METADATA_ONLY_G0_PREREQUISITE_CLOSURE_PLAN_FOUR_WORKSTREAMS_NO_EXECUTION`

## Result

The plan decomposes the remaining G0 work into four acyclic workstreams:

1. reconcile the G-1/G-1A binding contract;
2. define and later issue an atomic holdout-access seal;
3. precommit the remaining 17 quantitative choices in five packages; and
4. close 15 required sections only after their receipt and choice dependencies
   are bound.

This pass performs planning and independent validation only.

## G-1 diagnosis

The first blocker is not missing evidence. The existing G-1A receipt passes and
is closed over its exact config and verifier hashes. The incompatibility is:

- immutable G0 v1 expects `pivot_feedback_g_minus_1_data_readiness_outputs.v1`
  with `PASS_G_MINUS_1_DATA_READINESS`;
- the acyclic successor provides `pivot_feedback_g_minus_1a_receipt.v1` with
  its bounded current-snapshot pass status; and
- the legacy G-1 registry remains pending and is explicitly unable to promote
  production or G0.

Therefore the safe route is a versioned metadata-only binding-reconciliation
contract. It must bind the existing G-1A receipt exactly, retain all
current-snapshot and author-token caveats, leave legacy G-1 unpassed, and keep
the final cutoff panel owned by G0.5.

## Remaining workstreams

The holdout workstream first requires a versioned contract that separates seal
creation from seal opening. It must freeze the 2020–2022 entry partition,
follow-up through 2025, and domain/author-cold rule; require an atomic
hash-closed manifest; and fail closed before any confirmatory read.

The 17 quantitative choices are divided into:

- four measurement and proxy rules;
- six risk-set, weighting, support, and stability rules;
- one G1 replication rule;
- three G2 predictive rules; and
- three G3 simulation and complexity rules.

No value was selected in this step. Later contracts may use existing D0
discovery evidence and separately authorized methodological evidence, but not
confirmatory or holdout outcomes.

Each of the 15 required sections now has a future closure rule and explicit
dependencies. None is marked freeze-ready. The section matrix can close only
after the three prerequisite workstreams are independently validated.

## Boundary

Validation read nine repository documents and opened no operator JSON,
Parquet, row-level, confirmatory, or holdout data. It submitted no API request.
It does not authorize the binding change in G0 v2, a holdout seal, another
quantitative value, section completion, G0 sealing, model fitting, or G4.
