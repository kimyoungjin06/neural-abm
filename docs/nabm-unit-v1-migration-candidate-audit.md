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

## Gate 7B Result

Status: complete.

Implemented slice:

- `src/neural_abm/mixers.py::apply_scalar_output_average` now exposes a
  semantic-free scalar social helper backed by `NABMStep`.
- `src/neural_abm/toy_async.py::apply_output_average` routes Toy8
  activation-propensity social mixing through that helper.
- Toy8 still owns `ScheduledEvent`, `valid_event(...)`, `apply_event(...)`,
  `schedule_all_events(...)`, event counters, event-time RNG, and evidence
  interpretation.

Parity checks:

- `tests/test_social_block.py::test_scalar_output_average_unit_helper_matches_common_block`
- `tests/test_toy8_runner.py::test_toy8_output_average_matches_unit_scalar_parity`
- `tests/test_toy8_runner.py::test_toy8_output_average_routes_through_unit_scalar_helper`

Interpretation:

- The first existing-toy migration slice is parity-only. It shows that Toy8
  social-hazard mixing can pass through the unit-backed scalar social surface
  without turning `NABMUnit` into an event scheduler.
- It is not Toy8 performance evidence and does not promote Toy8 from
  `compatible` to `full`.

## Gate 7C Result

Status: complete.

Implemented slice:

- `src/neural_abm/toy_heterogeneous.py::apply_output_average` now routes Toy9
  heterogeneous action-probability social mixing through
  `src/neural_abm/mixers.py::apply_scalar_output_average`.
- Toy9 still owns group assignment, group-specific local-rule semantics,
  coordination gating, action sampling, payoff computation, propensity
  learning, payoff EMA, and evidence interpretation.

Parity checks:

- `tests/test_toy9_runner.py::test_toy9_output_average_matches_unit_scalar_parity`
- `tests/test_toy9_runner.py::test_toy9_output_average_routes_through_unit_scalar_helper`

Interpretation:

- The scalar social unit surface now has two existing-toy parity users: Toy8
  event-hazard mixing and Toy9 heterogeneous action-probability mixing.
- This supports reuse of lifecycle plumbing across existing compatible toys. It
  is not evidence that Toy9 is now a full NABM claim path.
- Additional scalar migrations should be justified by real duplication
  reduction; Toy7 remains a contract-gap decision because its central action is
  continuous extraction intensity.

## Gate 7D Result

Status: contract decision complete.

Decision artifact:

- `docs/decisions/0011-continuous-scalar-unit-contract.md`

Interpretation:

- Toy7 continuous extraction intensity is a bounded scalar, but it is not a
  probability.
- Toy7 should not be routed through `SCALAR_PROBABILITY_CHANNEL` as a normal
  parity migration.
- The next implementation needs a semantic-free bounded continuous scalar
  contract first, preferably `BOUNDED_SCALAR_CHANNEL` plus
  `mix_bounded_scalars(...)` or equivalent.
- Toy7 remains deferred until that contract has toy-independent tests.

Future Toy7 path:

1. Add bounded-scalar helper/channel tests independent of Toy7.
2. Route only Toy7 social intensity mixing through the new bounded-scalar path.
3. Preserve Toy7 resource dynamics, payoff semantics, noisy intensity sampling,
   and runner artifact fields.
4. Interpret the result as migration parity, not as full Toy7 NABM evidence.

## Gate 7E Result

Status: parity slice complete.

Implemented slice:

- `src/neural_abm/social.py::BOUNDED_SCALAR_CHANNEL` records bounded scalar as
  a separate social channel kind from `SCALAR_PROBABILITY_CHANNEL`.
- `src/neural_abm/social.py::mix_bounded_scalars` and
  `src/neural_abm/social.py::select_bounded_scalar_output_peers` validate
  finite values inside declared bounds before mixing or selecting peers.
- `src/neural_abm/mixers.py::apply_bounded_scalar_output_average` exposes the
  bounded scalar path through `NABMStep`.
- `src/neural_abm/toy_resource.py::apply_output_average` uses the bounded
  scalar helper for Toy7 extraction-intensity social mixing.

Parity checks:

- `tests/test_social_block.py::test_bounded_scalar_output_average_unit_helper_matches_common_block`
- `tests/test_toy7_runner.py::test_toy7_output_average_matches_unit_bounded_scalar_parity`
- `tests/test_toy7_runner.py::test_toy7_output_average_routes_through_unit_bounded_scalar_helper`

Interpretation:

- Toy7 now uses a bounded scalar unit surface for social extraction-intensity
  mixing without calling that value a probability.
- Toy7 still owns resource dynamics, payoff semantics, noisy intensity
  sampling, propensity learning, runner artifacts, and evidence interpretation.
- This does not promote Toy7 to full NABM evidence. It closes the bounded
  scalar contract gap for one parity slice.

## Gate 7F Result

Status: parity slice complete.

Audit finding:

- Toy10 does not require a vector-valued multi-channel contract for the first
  migration slice.
- Its social price expectation and conservation norm channels are separate
  bounded scalar values in `[0, 1]`.
- The composite peer-selection score is Toy10-owned market/ecology similarity,
  not a probability and not a new generic social-message schema.

Implemented slice:

- `src/neural_abm/toy_market.py::select_peer_ids` uses
  `select_bounded_scalar_output_peers(...)` for the Toy-owned composite
  similarity score.
- `src/neural_abm/toy_market.py::mix_channel` routes both
  `price_expectation` and `conservation_norm` through
  `apply_bounded_scalar_output_average(...)`.
- Toy10 still owns harvest construction, market price, resource transition,
  payoff updates, dynamic graph rewiring, channel-loss aggregation, and
  evidence interpretation.

Parity checks:

- `tests/test_toy10_runner.py::test_toy10_output_similarity_selects_bounded_scalar_composite`
- `tests/test_toy10_runner.py::test_toy10_mix_channel_matches_unit_bounded_scalar_parity`
- `tests/test_toy10_runner.py::test_toy10_mix_channel_routes_through_unit_bounded_scalar_helper`

Interpretation:

- Toy10 now reuses the bounded scalar unit surface for per-channel social
  mixing without introducing a generic multi-channel vector mixer.
- This is existing-toy migration parity only. It does not promote Toy10 to full
  NABM evidence and does not claim a general multi-channel message contract.

## Gate 7G Result

Status: parity slice complete.

Audit finding:

- Toy6 categorical social output is already a probability distribution over
  strategies.
- A new categorical-policy channel is not needed for the first migration slice.
- The generic unit should treat Toy6 values as row-stochastic distributions and
  leave strategy identity, cyclic payoff meaning, and evidence interpretation
  in Toy6.

Implemented slice:

- `src/neural_abm/mixers.py::apply_distribution_output_average` exposes a
  `NABMStep`-backed distribution social mix helper.
- `src/neural_abm/toy_categorical.py::apply_output_average` routes
  `strategy_distribution` through that helper with commit mode
  `categorical_probability_commit`.
- Toy6 still owns cyclic payoff construction, local logit updates, action
  sampling, payoff EMA, strategy entropy metrics, and runner artifacts.

Parity checks:

- `tests/test_social_block.py::test_distribution_output_average_unit_helper_matches_common_block`
- `tests/test_toy6_runner.py::test_toy6_output_average_matches_unit_distribution_parity`
- `tests/test_toy6_runner.py::test_toy6_output_average_routes_through_unit_distribution_helper`

Interpretation:

- Toy6 now reuses the probability-distribution unit surface for categorical
  output averaging without adding category-specific semantics to the unit.
- This is existing-toy migration parity only. It does not promote Toy6 to full
  NABM evidence and does not claim a general categorical ABM mechanism.
