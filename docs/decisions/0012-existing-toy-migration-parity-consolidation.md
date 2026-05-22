# Decision 0012: Existing-Toy Migration Parity Consolidation

## Status

Accepted.

## Date

2026-05-22

## Context

Decision 0010 froze the NABM Unit v1 boundary around lifecycle sequencing,
typed exchange, backend dispatch, and diagnostics. Decision 0011 added the
bounded scalar contract after Toy7 exposed that continuous extraction intensity
should not be routed through scalar probability semantics.

Gates 7B through 7G then migrated one bounded social-exchange slice from each
compatible Toy6-10 runner:

| Gate | Toy | Unit channel surface | Migrated slice |
| --- | --- | --- | --- |
| 7B | Toy8 | scalar probability-style social value | activation-propensity hazard mixing |
| 7C | Toy9 | scalar probability-style social value | heterogeneous action-probability mixing |
| 7E | Toy7 | bounded scalar | extraction-intensity social mixing |
| 7F | Toy10 | bounded scalar, one channel at a time | price-expectation and conservation-norm mixing |
| 7G | Toy6 | probability distribution | strategy-distribution categorical mixing |

The common question is now different from the earlier Toy2/Toy4 performance
question. The engineering claim is whether existing compatible ABM runners can
reuse the unit social-exchange surface without moving their domain equations
into the generic layer.

## Decision

The project treats Gates 7B-7G as **existing-toy migration parity**, not as
new performance evidence and not as promotion of Toy6-10 to full NABM status.

The supported engineering claim is:

> Toy6-10 can route their primary social exchange slices through typed NABM
> Unit surfaces while keeping event scheduling, categorical payoff semantics,
> resource dynamics, heterogeneous local rules, market/ecology equations,
> dynamic rewiring, action sampling, and evidence criteria toy-owned.

The consolidated typed social-exchange surfaces are scalar probability,
bounded scalar, and probability distribution.
This should be cited as typed social-exchange reuse, not as evidence that
Toy6-10 are full NABM models.

The unsupported claims remain:

- Toy6-10 are full NABM evidence cases.
- The unit owns categorical strategy meaning, extraction intensity meaning,
  event-time scheduling, heterogeneous group rules, market price, resource
  dynamics, or dynamic graph rewiring.
- The migrations demonstrate algorithmic superiority over hand-coded ABM
  baselines.
- The codebase is a complete general-purpose ABM framework.

## Consolidated Boundary

| Toy | Unit-owned migrated surface | Toy-owned boundary |
| --- | --- | --- |
| Toy6 | Row-stochastic strategy distribution mixing via `PROBABILITY_DISTRIBUTION_CHANNEL`. | Strategy identity, cyclic payoff construction, local logit updates, action sampling, payoff EMA, strategy entropy, and evidence interpretation. |
| Toy7 | Extraction-intensity social mixing via `BOUNDED_SCALAR_CHANNEL`. | Resource dynamics, payoff calculation, noisy intensity sampling, propensity updates, and continuous-action evidence interpretation. |
| Toy8 | Activation-propensity social mixing through the scalar social path. | Event queue, stale-event invalidation, event scheduling, event application, hazard semantics, counters, and event-time RNG. |
| Toy9 | Heterogeneous action-probability social mixing through the scalar social path. | Group assignment, group-specific local rules, coordination gates, action sampling, payoff computation, propensity learning, and payoff EMA. |
| Toy10 | Price-expectation and conservation-norm social mixing via `BOUNDED_SCALAR_CHANNEL`, applied per channel. | Composite market/ecology similarity, harvest construction, market price, resource transition, payoff updates, dynamic rewiring, channel aggregation, and evidence interpretation. |

## Engineering Consequence

The existing compatible-toy migration work is now broad enough to stop adding
more parity-only slices until a new domain shape requires a new typed exchange
contract.

Future structural work should choose one of these paths:

1. Consolidate the migrated surfaces into manuscript architecture claims.
2. Move runner lifecycle ownership into generic infrastructure where duplication
   is still real and semantics remain callback-owned.
3. Add a new typed exchange contract only when at least one real domain cannot
   be represented by scalar probability, bounded scalar, probability
   distribution, tensor, or state-dict channels.

## Non-Goals

This decision does not:

- add new evidence manifests;
- change Toy6-10 result interpretation;
- make Toy6-10 evidence-default;
- move local learning or environment transition rules into the unit;
- claim that typed social exchange reuse is sufficient for a full NABM
  architecture.
