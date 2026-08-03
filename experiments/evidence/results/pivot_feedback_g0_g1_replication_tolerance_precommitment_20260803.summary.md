# Pivot-feedback G0 G1 replication-tolerance precommitment summary

Date: 2026-08-03

Status: `PASS_ONE_OUTCOME_BLIND_G1_REPLICATION_TOLERANCE_AND_CLOSED_FIXTURE`

- Choice: `g1_replication_tolerance` only.
- Policy value: `1.0` cohort-year.
- Estimand: field H3 timing gap minus the equal-weighted mean timing gap of the
  other two G1 fields.
- Favorable direction: larger is better.
- Interval decision: every registered one-sided 95% lower bound must be
  strictly greater than `-1.0`; the boundary fails.
- Direction guards: every field, leave-one-field-out, and alternative-pivot
  point estimate must be strictly positive.
- Primary guard: field-balanced point estimate at least `1.0` with one-sided
  95% lower bound strictly greater than zero; neither branch rescues the other.
- Fixture: seven synthetic cases, one intended pass and six intended failures.
- Access: zero discovery, G1, production, confirmatory, or holdout rows; no
  model fit; G4 unopened.

The one-cohort-year value is an outcome-blind annual-grid policy, not a
literature mandate, replication probability, power or precision guarantee,
interval-estimator freeze, empirical result, G0 pass, transport result,
causality, or scientific effect. G0 v4 remains unchanged and no successor
binding is authorized.
