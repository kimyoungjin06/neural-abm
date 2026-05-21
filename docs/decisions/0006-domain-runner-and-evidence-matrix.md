# Decision 0006: Domain Runner And Evidence Matrix

## Status

Accepted.

## Context

The toy suite now has two shared lifecycle runners:

- `BinarySpatialRunner` for the full NABM binary spatial lifecycle used by
  Toy2, Toy4, and Toy5.
- `DomainToyRunner` for the compatible domain lifecycle used by Toy6 through
  Toy10.

These runners are not a universal ABM engine. They standardize repeatable run
setup, logging, summary artifacts, and result envelopes around the current toy
suite while leaving domain equations, event laws, payoff/resource rules, and
state transitions inside each toy.

## Decision

Keep the runner boundary at the common lifecycle layer.

`BinarySpatialRunner` owns the binary spatial loop: run directory creation,
metadata writing, state initialization, initial aggregate row, revision mask
sampling, local update, peer selection, social coordination, action commit,
post-step reputation/mobility updates, aggregate/micro CSV writing, and binary
summary writing.

`DomainToyRunner` owns the Toy6-10 compatible lifecycle: run directory
creation, metadata writing, state initialization, epoch iteration, aggregate
and micro CSV writing, fallback row handling for zero-step domains, domain
summary writing, and the common `DomainToyResult` envelope. The toy adapter now
only supplies domain-specific lifecycle callbacks. `DomainRunSettings` carries
the run-level pieces that had previously been repeated by every adapter.

New compatible domain toys should be added in this order:

1. Validate a config model with the public `run`, `simulation`, `model`,
   `domain`, and `logging` shape.
2. Initialize domain state from the config and seed.
3. Implement one domain step and any fallback step needed for zero-event
   runs.
4. Emit an aggregate row with stable `domain_*` fields and
   `fragmentation_components`.
5. Emit micro rows with stable per-agent or per-event fields.
6. Map final rows into `domain_metrics`.
7. Fill `DomainRunSettings.metadata` with artifact metadata and capability
   fields through the shared metadata writer.

The evidence matrix is a small reproducible benchmark artifact, not a paper
claim by itself. It runs a fixed set of Toy1-5 comparison cases, writes
run-level metric rows, and summarizes direction-aware effects with mean, std,
and 95% confidence intervals. Toy6-10 are compatible with the shared domain
runner but are not default evidence-matrix cases until their neural path is the
validated experiment mechanism.

## Consequences

Adapter boilerplate is reduced without promoting Toy6-10 to `full` NABM status.
Run artifacts stay inspectable because generated configs, run directories, CSV
rows, summary JSON, and effect summaries are preserved.

The evidence matrix complements, but does not replace, the broader validation
suite. It is for repeated claim checks across a small set of mechanism-focused
comparisons.
