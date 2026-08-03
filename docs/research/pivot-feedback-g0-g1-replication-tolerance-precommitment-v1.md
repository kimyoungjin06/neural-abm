# Pivot-feedback G0 G1 replication-tolerance precommitment v1

Date: 2026-08-03

Status: `COMPLETE_INDEPENDENTLY_VERIFIED_CLOSED_FIXTURE_NO_G0_PROMOTION`

## Decision

Gate v61 authorizes exactly one outcome-blind package before any G1 access:
`g1_replication_tolerance`. The versioned contract is
[`pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.yaml`](../../experiments/configs/pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.yaml),
SHA-256
`1bcf91d197a9b927286d43f3505528f3f3f7fb022ffd4fe89d95cb24c041184d`.

The frozen policy value is one cohort-year. It is defined on the additive H3
timing-gap scale, not on an odds-ratio, probability, or standardized-effect
scale. For episode `j`, the existing protocol defines

```text
delta_j = earliest entry-hazard peak k
          - earliest three-year-persistence peak k,
where k is 0, 1, or 2.
```

For G1 field `f`, let `theta_f` be the equal-weighted mean of `delta_j` across
that field's independent episodes. Let `theta_-f` be the equal-weighted mean of
the other two G1 field values. The replication contrast is

```text
rho_f = theta_f - theta_-f.
```

Larger values are more favorable. Every field's registered one-sided 95%
lower confidence bound for `rho_f` must be strictly greater than `-1.0`
cohort-year. Equality at `-1.0` fails.

## Why one cohort-year

The H3 episode clock has annual resolution and the primary point criterion is
already one cohort-year. One annual step therefore supplies a transparent,
scale-native maximum tolerated downward field-versus-other-fields degradation.
This is a conservative judgment-based policy chosen before G1 access. It is
not a literature-mandated constant, estimated heterogeneity, replication
probability, power calculation, or precision guarantee.

The margin does not permit a strong field to hide a reversed or absent signal.
The package additionally requires:

- exactly three G1 fields and at least three independent episodes per field;
- every field-specific point estimate to be strictly positive;
- every leave-one-field-out point estimate to be strictly positive;
- every pre-specified alternative-pivot field estimate to be strictly
  positive;
- the primary field-balanced point estimate to be at least one cohort-year;
  and
- the primary one-sided 95% lower bound to be strictly greater than zero.

Neither branch rescues the other. A failed primary cannot be promoted by the
replication rule, and a passed primary cannot override a field-replication
failure.

## Interval boundary

This package freezes the interval decision criterion, not the interval
estimator. It consumes a future registered field-balanced, episode-clustered
one-sided 95% lower bound. The exact estimator, resampling or small-sample
procedure, missing-cohort treatment, and related uncertainty details remain
unresolved in the future `estimators_and_uncertainty` G0 section. This package
therefore does not complete any of the fifteen required sections.

## Closed synthetic validation

The application tool is
[`apply_pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.py`](../../scripts/apply_pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.py),
SHA-256
`30d9fe0c6f443bf11eccf9d5eb3b19dd279478538f37624dd58f0b8d5126357a`.
It accepts only a closed synthetic fixture with no discovery, G1,
confirmatory, holdout, production, model-score, or real field/episode content.

The independent verifier is
[`verify_pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.py`](../../scripts/verify_pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.py),
SHA-256
`7e5a7a770e0dc6f9043e552d011a7e624dabd73b6945f5bd0dc36cd223ea7cd7`.
It binds gate v61, G0 v4, the closure plan, and protocol byte-for-byte; replays
the rule without importing the application tool; then executes the application
tool and compares its complete output to the frozen expected fixture.

Seven synthetic cases cover:

1. a complete pass;
2. failure at the exact `-1.0` interval boundary;
3. failure when a field direction equals zero;
4. failure when a leave-one-field-out direction equals zero;
5. failure when an alternative-pivot direction is negative;
6. non-rescue of a primary point estimate below one cohort-year; and
7. failure below the three-independent-episode floor.

The independent result is
[`pivot_feedback_g0_g1_replication_tolerance_precommitment_20260803.summary.json`](../../experiments/evidence/results/pivot_feedback_g0_g1_replication_tolerance_precommitment_20260803.summary.json),
SHA-256
`72b653528d9cf5c8f1dd823f304934b48ac21a0dd4a50011ac3d02ef9079edb3`.
It reports one intended pass and six intended failures with zero non-synthetic
row access.

## Preserved boundary

This package does not edit G0 v4, bind the value into a G0 successor, apply the
rule to discovery or G1 data, choose an interval estimator, resolve any G2 or
G3 value, complete a required G0 section, select or materialize G4, create a
holdout preflight or seal, pass G0, fit a model, or establish an empirical,
transport, causal, or scientific effect.

Gate v61 therefore remains the current machine-readable authority. A separate
successor gate is required before any further binding or application.
