# NABM Unit v1 Migration Candidate Audit

Date: 2026-05-22

## Purpose

Gate 5 and Gate 6 show that source-free adapter holdouts can use the public
binary policy lifecycle without changing `src/neural_abm`. Those holdouts are
still compact scripted domains. This audit selects the next migration target
from existing Toy6-10 runners so the next structural claim tests an already
implemented toy rather than another purpose-built holdout.

The goal is not to promote Toy6-10 into the primary evidence path. The goal is
to identify one bounded partial migration that can test whether the NABM Unit
contract helps an existing runner without absorbing domain equations.

## Selection Criteria

| Criterion | Meaning |
| --- | --- |
| Existing-toy pressure | The target already exists in `src/neural_abm`, with tests and run artifacts. |
| Beyond compact binary scripts | The target exercises a state/update shape not covered by Gate 5 or Gate 6. |
| Clear domain boundary | Reward, event, resource, threshold, or transition meaning can stay in the toy. |
| Unit-surface fit | A small lifecycle slice can route through `NABMUnit`, `NABMStep`, or an existing unit-backed helper without a full rewrite. |
| Parity-test feasibility | The migration can be checked against the current path using a quick deterministic or bounded stochastic contract test. |
| Claim safety | Passing evidence would support migration compatibility, not performance superiority. |

## Candidate Summary

| Candidate | Current shape | Migration pressure | Risk | Recommendation |
| --- | --- | --- | --- | --- |
| Toy8 async event ABM | Event queue with activation/failure/recovery hazards and stale-event invalidation. | High: event-time lifecycle is not covered by the scripted adapter holdouts. | Medium: preserving event queue semantics and RNG order requires a parity-first slice. | Primary Gate 7B target. |
| Toy9 heterogeneous binary adoption | Binary actions with group-specific local rules, payoff learning, and coordination gates. | Medium: group heterogeneity is useful, but the action surface is close to existing binary paths. | Low: current adapter is small and already has clear probabilities/actions/payoffs. | Fallback target if Toy8 parity proves too invasive. |
| Toy7 continuous resource intensity | Continuous extraction intensity with endogenous resource dynamics. | High: continuous scalar action exposes a real unit-surface gap beyond binary probability lifecycle. | High: it likely needs a continuous policy lifecycle or scalar commit unit, not only a migration wrapper. | Defer until after one lower-risk existing-toy migration. |

## Toy8 Audit

Source paths:

- `src/neural_abm/toy_async.py`
- `tests/test_toy8_runner.py`
- `tests/test_toy8_sweep.py`

Current ownership:

- `DomainToyRunner` owns run directory setup, metadata, epoch iteration, CSV
  writing, fallback handling, summary writing, and `DomainToyResult`.
- `Toy8Adapter` owns graph construction, initial state fractions, event queue,
  stale-event invalidation, event application, event counters, and final
  metrics.
- `compute_rate_snapshot(...)` owns hazard construction:
  neighbor-active fractions, local activation propensities, output-average
  mixing, activation/failure/recovery rates, peer ids, social losses, and
  update norms.

Best migration slice:

1. Keep `DomainToyRunner`, event queue scheduling, event validity, state
   transition, counters, and event-time RNG in Toy8.
2. Extract only the scalar social hazard-mixing slice inside
   `compute_rate_snapshot(...)`.
3. Route that slice through a unit-backed scalar social helper or
   `NABMStep`-backed adapter, preserving the current mixed activation
   propensities, peer ids, social losses, and update norms.
4. Add a parity test that compares current Toy8 output-average hazard mixing
   with the unit-backed path on a fixed tiny state.
5. Only after parity passes, optionally wire the unit-backed path behind an
   opt-in config flag or helper call used by Toy8.

Why Toy8 first:

- It is structurally different from Gate 5 and Gate 6 because event scheduling
  and stale events are central to the domain.
- A useful slice exists that does not require the unit to own event semantics:
  social hazard mixing is separable from event scheduling.
- Success would show that the unit contract can help an existing event ABM
  without turning the generic unit into an event scheduler.

Risk controls:

- Do not move `ScheduledEvent`, `valid_event(...)`, `apply_event(...)`, or
  `schedule_all_events(...)` into the unit.
- Do not change event-time RNG order in the first slice.
- Do not change Toy8 aggregate or micro CSV fields.
- Do not claim performance improvement. The first claim is parity plus
  diagnostics boundary.

## Toy9 Fallback Audit

Source paths:

- `src/neural_abm/toy_heterogeneous.py`
- `tests/test_toy9_runner.py`
- `tests/test_toy9_sweep.py`

Current ownership:

- `Toy9Adapter.step(...)` computes base action probabilities, selects peers,
  mixes probabilities, samples binary actions, computes neighbor action rates,
  computes payoffs, updates propensities, updates payoff EMA, and emits
  diagnostics.
- Group assignment, group-specific local rules, coordination enablement, and
  payoff-learning rules are already toy-owned.

Best migration slice if Toy8 is too invasive:

1. Keep group assignment, local-rule meaning, payoff computation, and
   propensity learning in Toy9.
2. Route the output-average probability-mixing slice through the existing
   binary or scalar unit-backed social path.
3. Add parity tests for mixed probabilities, peer ids, losses, and update norms.

Why fallback only:

- The migration is likely easier, but it is close to prior binary
  probability/adoption work.
- It would support existing-toy migration, but it would not pressure event-time
  or continuous-action unit boundaries.

## Toy7 Deferred Audit

Source paths:

- `src/neural_abm/toy_resource.py`
- `tests/test_toy7_runner.py`
- `tests/test_toy7_sweep.py`

Current ownership:

- `Toy7Adapter.step(...)` mixes continuous propensities, samples noisy
  intensities, computes payoffs, updates resource stock, updates payoff EMA,
  and updates propensities.
- The central action is continuous scalar extraction intensity, not a binary
  probability.

Why defer:

- Toy7 is the best pressure test for a future continuous-action unit, but it is
  probably a contract expansion rather than a migration-only slice.
- Doing Toy7 first would mix two questions: whether the existing unit contract
  supports an existing toy, and whether v1 needs a new continuous scalar
  policy/commit contract.

Acceptable future Toy7 path:

1. Record a contract gap for continuous scalar policy lifecycle.
2. Add a small toy-independent scalar-unit test.
3. Only then migrate Toy7 social intensity mixing or local scalar commit.

## Gate 7B Recommendation

Proceed with **Toy8 async social-hazard parity** as the next implementation
slice.

Gate 7B should produce:

- a Toy8 unit-backed scalar hazard mixing helper or adapter path;
- a parity test on a tiny deterministic Toy8 state comparing peer ids, mixed
  activation propensities, social losses, and update norms;
- a Toy8 runner contract test showing aggregate/micro fields remain stable;
- no changes to event scheduling, event validity, event application, event
  counters, or evidence gates;
- a checklist update marking the slice as migration parity, not new
  performance evidence.

Fallback rule:

- If Toy8 parity requires generic unit changes beyond scalar social mix
  plumbing, pause and record the contract gap. Then use Toy9 binary
  heterogeneous probability mixing as the lower-risk Gate 7B migration target.
