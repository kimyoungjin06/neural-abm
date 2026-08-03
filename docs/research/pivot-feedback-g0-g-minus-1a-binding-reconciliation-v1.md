# Pivot-feedback G0 G-1A binding reconciliation v1

Date: 2026-07-31

Status:
`PASS_METADATA_ONLY_G_MINUS_1A_BINDING_RECONCILIATION_FUTURE_G0_V2_ONLY_NO_RELABEL`

## Decision

The existing passing G-1A receipt can serve as bounded prerequisite evidence
for a future versioned G0 successor, but it must not be relabeled as a legacy
G-1 pass.

The future successor must use a distinct `g_minus_1a_binding` field. If it
retains the old field for lineage, that field becomes
`legacy_g_minus_1_binding` with state
`PENDING_NOT_PASSED_IMMUTABLE_LINEAGE`.

## Exact binding

The future binding is valid only against:

- schema `pivot_feedback_g_minus_1a_receipt.v1`;
- the exact bounded G-1A pass status;
- gate ID `G_MINUS_1A`;
- receipt SHA
  `8383e1884c3679df225b743f775832f02e4d767ac5a6b047dbde8aacf9e12802`;
- the receipt config SHA; and
- the independent verifier SHA.

Schema aliasing, status relabeling, and legacy-pass inference are prohibited.
An exact match can satisfy only the future G0 v2 G-1A bounded-evidence
prerequisite. It cannot pass G0, authorize model fitting, or open G4.

## Preserved limitations

All 14 unresolved receipt boundaries are copied exactly. In particular:

- author IDs remain current-snapshot tokens, not validated natural persons;
- historical author, topic, and citation observability remain unidentified;
- candidate eligibility, work types, and fixed horizon remain pending G0;
- the final cutoff panel remains pending G0.5; and
- no empirical or scientific effect is established.

The complete receipt claim boundary is also preserved exactly.

## Future G0 v2 boundary

A future binding-only successor must preserve eight G0 v1 subtrees:
foundation parents, holdout binding, fixed design, resolution, inherited
digests, completion contract, failure policy, and authorization.

Its only permitted semantic changes are retaining the legacy G-1 binding as
pending lineage and adding the exact G-1A binding. The holdout binding remains
null, all 17 values remain unresolved, all 15 sections remain not
freeze-ready, and `pass_g0` remains false.

This validation opened no operator data, confirmatory data, or holdout data and
made no API request. It does not itself create or modify G0 v2.
