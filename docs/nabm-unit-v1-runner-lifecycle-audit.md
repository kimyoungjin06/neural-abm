# NABM Unit v1 Runner Lifecycle Audit

Date: 2026-05-22

## Purpose

This audit records what runner lifecycle behavior is already shared after the
Toy6-10 typed social-exchange parity slices, and what should remain
toy-owned. It is an engineering consolidation check, not new performance
evidence and not a promotion of Toy6-10 to full NABM evidence cases.

The main conclusion is conservative: `DomainToyRunner` already owns the common
outer run lifecycle. The next useful extraction is a small social diagnostics
mapper, not a full runner rewrite.

## Current Shared Runner Surface

`src/neural_abm/domain_runner.py` provides the common runner shell used by the
Toy6-10-compatible domain runners:

- `DomainRunSettings` owns run-level configuration: toy id, config object,
  config path, output directory, run name, seed, CSV fields, metadata, logging
  controls, no-step error text, and strict-capability behavior.
- `DomainToyAdapter` defines the toy-owned callbacks: `initialize`,
  `step_epochs`, `step`, `fallback_step`, `aggregate_row`, `micro_rows`,
  `final_epoch`, and `domain_metrics`.
- `DomainToyRunner` owns the timestamped run directory, metadata artifact
  writing, adapter initialization, epoch loop, fallback step, `micro_state.csv`,
  `aggregate_metrics.csv`, final summary artifact, and `DomainToyResult`
  envelope.

This means the project already has a reusable outer lifecycle for compatible
domain toys. The remaining duplication is mostly inside the toy-specific
`step(...)`, aggregate-row, and micro-row adapters.

## Existing Common Enough Pieces

These pieces are already common enough for the v1 engineering boundary:

- Run directory and metadata artifact lifecycle.
- CSV writer lifecycle for `aggregate_metrics.csv` and `micro_state.csv`.
- Epoch loop, stop-on-`None`, fallback, final-row lookup, final summary, and
  result envelope.
- Config/result shape through `DomainRunSettings` and `DomainToyResult`.

The generic runner should not take over toy-specific phase ordering just to
reduce line count.

## Toy6-10 Inner Lifecycle Audit

| Toy | Inner lifecycle shape | Unit-owned slice today | Toy-owned semantics that must stay local |
| --- | --- | --- | --- |
| Toy6 categorical game | local policy readout, action sampling, cyclic payoff, logit update, social distribution mix, final action sample | `strategy_distribution` mixing through the probability-distribution path | strategy identity, cyclic payoff construction, categorical action sampling, local logit learning, payoff EMA, strategy metrics |
| Toy7 resource intensity | peer selection, bounded scalar social intensity mix, noisy clipped intensity, payoff/resource update, propensity update | extraction-intensity mixing through the bounded scalar path | resource stock dynamics, payoff construction, noisy intensity sampling, propensity learning, evidence interpretation |
| Toy8 async event process | pop valid event, max-time guard, apply event/counters, schedule future events, rate snapshot | scalar activation-propensity social mix inside rate snapshot | event queue, stale-event invalidation, event application, scheduling, counters, asynchronous time semantics |
| Toy9 heterogeneous adoption | action-probability readout, group-gated social mix, action sampling, payoff/neighbor-rate computation, propensity update, payoff EMA | heterogeneous action-probability mixing through the scalar path | group assignment, group-specific local rules, coordination gating, payoff construction, action sampling, propensity learning |
| Toy10 market/ecology | graph snapshot, composite peer selection, two bounded scalar channel mixes, harvest, market price, payoff/resource update, channel update, dynamic rewire, local means | price-expectation and conservation-norm mixing through the bounded scalar path | multi-channel aggregation, market price, resource dynamics, payoff updates, channel learning, dynamic graph rewiring |

Do not unify Toy6-10 step order. The phase order carries domain semantics in
each toy.

## Candidate Extraction Surfaces

### 1. Social Exchange Wrapper

This is already mostly done. The `apply_scalar_output_average`,
`apply_bounded_scalar_output_average`, and `apply_distribution_output_average`
helpers route reusable social selection/mixing/commit diagnostics through
typed unit surfaces.

Further wrapper work should happen only when a new typed exchange surface is
needed by more than one domain. It should not create a large multi-toy runner
abstraction just to call existing helpers.

### 2. Social Diagnostics Mapper

This is the best next implementation target. Several Toy6-10 adapters repeat
small row-mapping work around peer counts, social losses, and social update
norms. A small helper can map semantic-free social fields while leaving domain
fields in the toy:

- per-agent fields such as `peer_ids`, `peer_count`, `social_loss`, and
  `social_update_norm`;
- aggregate fields such as `mean_peer_count`, `mean_social_loss`, and
  `mean_social_update_norm`;
- optional run context such as toy name, seed, epoch, and channel name only
  when those values are already domain-supplied.

This should be Gate 8B: Social Diagnostics Mapper Prototype. The helper should
be tested outside any one toy first, then migrated into one or two adapters with
artifact-field parity checks.

### 3. Adapter Protocol Refinement

`DomainToyAdapter` can grow optional standard phase labels or a small
`StepDiagnostics` carrier later, but only after a diagnostics mapper shows
repeated value. Adding a broader protocol now would risk hiding domain order
inside generic names.

### 4. Runner Lifecycle Extraction

A full runner rewrite is not the next step. `DomainToyRunner` already owns the
outer lifecycle. Moving more logic into it would mostly move toy-specific
`step(...)` ordering, payoff/resource/event/market semantics, or row schemas
into a generic layer, which would weaken the NABM unit boundary.

## Non-Extraction Rules

Do not move any of the following into `DomainToyRunner`, `NABMUnit`, or shared
diagnostic helpers:

- payoff, reward, resource, market, event, group, threshold, or categorical
  strategy semantics;
- Toy8 event-time queue ownership, event validity, event application, or
  scheduling;
- Toy10 dynamic graph rewiring, market-price construction, or resource
  dynamics;
- Toy6-10 `step(...)` phase ordering;
- evidence criteria or performance claims.

Shared helpers may carry numbers and field names supplied by toy adapters, but
they must not decide what those numbers mean.

## Recommendation

Gate 8A is complete as an audit. The next implementation slice should be
Gate 8B: Social Diagnostics Mapper Prototype.

Acceptance criteria for Gate 8B:

- Add a semantic-free helper for common peer/social metric row mapping.
- Cover the helper with toy-independent tests.
- Migrate Toy6-10 adapters only where the output fields stay identical.
- Keep `peer_count`, `mean_peer_count`, `mean_social_loss`, and
  `mean_social_update_norm` as diagnostics, not domain objectives.
- Do not make this a full runner rewrite.

## Gate 8B Result

Gate 8B implemented the small helper path recommended by this audit:

- `src/neural_abm/domain_social_diagnostics.py` provides
  `aggregate_social_diagnostic_fields(...)` and
  `micro_social_diagnostic_fields(...)`.
- `tests/test_domain_social_diagnostics.py` covers the mapper outside any one
  toy.
- Toy6-Toy10 now use the mapper for `peer_count`, `mean_peer_count`,
  `mean_social_loss`, and `mean_social_update_norm` row fields.
- Toy6-Toy10 routing tests guard that their row builders call the shared
  mapper while keeping domain fields toy-owned.

This is not a full runner rewrite. The mapper keeps artifact fields identical
and does not absorb event, group, market, resource, categorical, or
dynamic-graph semantics.

## Gate 8C Result

Gate 8C reduced the remaining safe run-artifact boilerplate:

- `src/neural_abm/domain_runner.py` now exposes `make_domain_run_dir(...)` and
  `write_domain_run_metadata(...)` over `DomainRunSettings`.
- `DomainToyRunner.run()` uses the same helpers as the thin Toy6-Toy10
  compatibility wrappers.
- Toy6, Toy7, Toy8, Toy9, and Toy10 keep their public `make_run_dir(...)` and
  `write_run_metadata(...)` wrapper functions, but those wrappers now delegate
  to the shared settings-based helpers.
- `tests/test_domain_runner.py::test_domain_run_artifact_helpers_use_settings`
  guards the shared run-directory and metadata artifact behavior.

This is the safe extraction boundary for adapter thinness. The audit still
rejects moving adapter `step(...)`, `aggregate_row(...)`, `micro_rows(...)`,
`final_epoch(...)`, or `domain_metrics(...)` into a generic helper because
those methods carry toy-specific phase order, row schemas, final-time meaning,
and domain metrics.
