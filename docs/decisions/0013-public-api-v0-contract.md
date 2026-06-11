# Decision 0013: Public API v0 Contract Boundary

## Status

Accepted.

## Date

2026-06-04

## Context

Decision 0010 froze the NABM Unit v1 boundary around lifecycle sequencing,
typed exchange, backend dispatch, and diagnostics. Decisions 0011 and 0012 then
extended the typed social-exchange surface across compatible Toy6-10 migration
parity slices. Gate 8D added artifact-contract tests for Toy6-Toy10 CSV schemas.

The project now has enough internal structure to start API design, but the
release target is triple-use:

- an internal reusable module;
- a paper evidence package;
- a public Python package.

Those targets should share a core, but they must not share the same public API
surface. Publishing the whole research repository as the API would expose
unstable evidence machinery and toy-owned domain semantics as if they were
framework guarantees.

## Decision

The project will create a narrow v0 API facade only after classifying the
existing surface. The initial facade should lock reusable mechanics, not
research semantics.

The preferred stable import path is `neural_abm.api`. The current broad
`neural_abm.__init__` export list should be treated as a lazy compatibility
surface for existing module-path imports. It should not be interpreted as the
public v0 contract.

## Stable v0 Responsibilities

The v0 API may expose:

| Surface | Stable responsibility |
| --- | --- |
| Unit lifecycle | `NABMUnit`, `NABMStep`, `NABMLocalStep`, reports, specs, and callback/value-builder types. |
| Typed social exchange | `SocialChannel`, `SocialBlock`, `SocialMixResult`, peer selection, channel constants, and scalar/bounded/distribution mixing helpers. |
| Compatible-toy runner shell | `DomainToyRunner`, `DomainToyAdapter`, `DomainRunSettings`, run-directory and metadata helpers. |
| Semantic-free diagnostics | Peer/social aggregate and micro row mapping helpers. |
| Result envelopes | Stable result objects and summary-artifact helpers where they do not encode claim semantics. |
| Readiness aggregation | Generic binary peer-readiness propagation after domains provide readiness values. |

## Experimental Responsibilities

The following may be exposed only through an experimental namespace or explicit
module paths:

- binary policy-learning lifecycle helpers;
- binary revision lifecycle helpers;
- accelerator, batched, and tensor-runtime helpers;
- mobility and reputation utilities;
- evidence manifest loaders and profile builders.

Experimental imports can support internal and paper work, but they are not
stable public contracts.

## Non-API Responsibilities

The public v0 API must not own:

- payoff, reward, resource, market, event, group, threshold, categorical
  strategy, teacher, basin-credit, readiness-meaning, or revision-pressure
  construction;
- individual toy `step(...)` phase order;
- evidence criteria, paper claim judgment, or gate pass/fail interpretation;
- generated experiment manifests, generated configs, run directories, or result
  profiles as package data;
- runtime cache mutation or optimizer-state internals.

Those surfaces can remain in the repository for research and paper
reproducibility, but they are not framework promises.

## Toy Runner Policy

Toy runners remain importable from their module paths for scripts, tests, and
paper reproduction. They should not be promoted into the stable facade unless a
runner is explicitly converted into an example or template with a documented
contract.

Toy6-Toy10 keep their `compatible` status. Toy2/Toy4/Toy5 remain primary
binary NABM research runners. None of those labels turns toy-owned semantics
into stable public API.

## Evidence Package Boundary

The paper evidence package should be organized around manifests, result
profiles, findings files, and claim matrices. Its contract is reproducibility
and bounded claim wording, not general library behavior.

Evidence tooling may later get a separate namespace, but it should be named as
paper or experimental tooling. It should not be mixed into the stable core
facade.

## Implementation Consequence

The next implementation slice should:

1. add a small `neural_abm.api` facade;
2. export only the stable v0 candidates from `docs/api-surface-audit.md`;
3. add import-smoke and public-surface tests;
4. leave `neural_abm.__init__` unchanged unless a compatibility migration is
   explicitly planned;
5. update this decision if the facade needs to include any domain-semantic or
   evidence-semantic object.

## Non-Goals

This decision does not:

- publish the current package as a finished general-purpose ABM framework;
- claim Toy6-Toy10 are full NABM evidence cases;
- make paper evidence manifests stable library APIs;
- remove existing broad top-level exports;
- add new simulation features.
