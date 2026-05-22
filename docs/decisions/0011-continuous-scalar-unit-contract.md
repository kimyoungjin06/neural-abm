# Decision 0011: Continuous Scalar Unit Contract Boundary

## Status

Accepted.

## Date

2026-05-22

## Context

Gate 7B and Gate 7C migrated two existing compatible toys through the
unit-backed scalar social path:

- Toy8 uses scalar social mixing for activation hazard propensities.
- Toy9 uses scalar social mixing for heterogeneous action probabilities.

Both slices reuse `src/neural_abm/mixers.py::apply_scalar_output_average`, but
they are still probability-like social messages. Toy7 is different. Its central
action is continuous extraction intensity in `[0, 1]`; it is bounded, but it is
not a probability of choosing action 1. Routing Toy7 through
`SCALAR_PROBABILITY_CHANNEL` would make the migration easy while blurring the
unit contract.

This matters because Decision 0010 froze the NABM Unit v1 boundary around
lifecycle sequencing, typed exchange, backend dispatch, and diagnostics. The
unit can validate and mix values, but it must not hide resource, payoff, or
extraction semantics inside a generic helper.

## Decision

Toy7 should not be migrated through `SCALAR_PROBABILITY_CHANNEL` as a normal
existing-toy parity slice.

The project now treats Toy7 continuous extraction intensity as a contract gap:
it requires a bounded continuous scalar channel before implementation. The
preferred future API shape is:

- `BOUNDED_SCALAR_CHANNEL = "bounded_scalar"` or an equivalent channel kind;
- `mix_bounded_scalars(...)` or an equivalent helper that mixes finite bounded
  scalar values without naming them probabilities;
- adapter-owned bounds and labels so a domain can state whether the value is an
  intensity, hazard, resource allocation, or other continuous scalar;
- diagnostics that preserve channel name, commit mode, peer ids, losses, and
  update norms without interpreting the scalar.

The generic unit may validate that values are finite and within declared bounds.
The domain must still own:

- resource stock dynamics;
- payoff and utility calculation;
- extraction-intensity semantics;
- noisy intensity sampling;
- propensity or policy update rules;
- evidence criteria.

## Implementation Rule

Toy7 migration can proceed only after bounded-scalar contract tests exist
outside Toy7.

The next implementation gate should first add the semantic-free bounded scalar
surface and tests, then migrate only the Toy7 social intensity mixing slice as a
parity check. It should not promote Toy7 to a full NABM evidence case and should
not claim performance improvement.

## Non-Goals

This decision does not:

- make Toy7 a full NABM claim path;
- change Toy7 resource or payoff logic;
- add a continuous-action policy-learning algorithm;
- reinterpret extraction intensity as a probability;
- change evidence gates or manuscript claims.

## Acceptance Criteria for Gate 7E

Gate 7E can be considered complete when:

1. A bounded continuous scalar channel/helper exists and is tested without
   Toy7-specific semantics.
2. The helper rejects non-finite values and out-of-bounds values using declared
   bounds.
3. Toy7 social intensity mixing has a parity test against the previous path for
   mixed intensities, peer ids, losses, and update norms.
4. Toy7 runner artifacts keep their aggregate and micro field contracts.
5. Documentation states that the result is bounded-scalar migration parity, not
   evidence that Toy7 is a full NABM or that continuous NABMs are solved.
