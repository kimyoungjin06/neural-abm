# Decision 0015: Researcher Scenario Lite Contract

## Status

Accepted for the `main` / next-alpha candidate.

## Date

2026-07-16

## Context

Decision 0013 keeps evidence criteria and paper-claim judgment outside the
stable public API. Decision 0014 establishes `neural_abm.api_lite` as the
torch-free package floor. The researcher-pivot work adds a reusable need within
those boundaries: users should be able to declare baseline and counterfactual
scenarios, run the same bounded-scalar lifecycle for each scenario, and receive
deterministic or seed-paired comparison records without importing `torch`.

The comparison layer necessarily carries names such as research question,
outcome, success direction, and minimum delta. Those fields can look like a
framework-owned scientific evaluation unless their ownership is explicit.

## Decision

The `main` / next-alpha candidate adds the bounded-scalar scenario surface to
`neural_abm.api_lite`. The public facade may export:

- `ScenarioDefinition` and `BoundedScalarScenarioSpec`;
- `ScenarioComparison` and `BoundedScalarScenarioResult`;
- `ReplicationSpec` and `ScenarioReplicateContext`;
- `ReplicatedScenarioComparison` and `ReplicatedScenarioResult`;
- `run_bounded_scalar_scenarios(...)` and
  `run_replicated_bounded_scalar_scenarios(...)`.

`neural_abm.scenario_lite` owns generic orchestration and result envelopes.
`neural_abm.workflow_lite` is its torch-free bounded-scalar implementation
layer; it is not separately promoted through the facade by this decision.
Neither module owns a domain model.

Callers continue to own:

- the scientific question and scenario interpretation;
- agent construction, topology, local updates, and domain transitions;
- the outcome field and the meaning of a scientifically relevant effect;
- calibration, identification assumptions, robustness design, and claim
  wording.

## Success-Criterion Boundary

`success_direction` and `success_min_delta` are user-provided comparison
metadata. The resulting `success_criterion`, `success`, and
`success_fraction` fields report only whether the simulated delta mechanically
satisfies that caller-supplied threshold. They do not determine whether a
scientific claim is supported, whether an intervention is effective in the
world, or whether the scenario model is valid.

Likewise, replicate pairing is an execution and variance-control mechanism.
The framework reuses a seed by replicate, but strict common random numbers
still require caller callbacks to align component-level draws rather than
consume one mutable stream through scenario-dependent branches.

The reported `delta_ci95` / `mean_effect_ci95` is a normal-approximation
confidence interval for the paired mean delta. The separate
`delta_empirical_percentile_interval95` describes the central spread of
replicate-level deltas and is not a confidence interval for the mean. The
framework does not interpret it as a hypothesis test. Neither interval
guarantees finite-sample coverage, chooses a causal estimand, or validates the
caller's data-generating assumptions.

A mean confidence interval requires at least two replicates. For a one-
replicate smoke run, the mean-interval fields are `null` and the method is
`unavailable_requires_at_least_2_replicates`; the framework must not report the
single value as a zero-width 95% interval. `round_digits` is an audit-row
presentation control only. Effect estimates, replicate outcomes, and
distribution summaries retain the unrounded values used for analysis, and the
result payload labels both precision scopes explicitly.

Docs and examples may explain what a particular caller chose as its threshold,
but must not describe `success=true` as framework adjudication of a paper claim.

## Dependency and Release Boundary

The scenario surface remains within the default dependency policy: it uses
NumPy and the existing torch-free `social_core` and `unit_core` paths, and must
not import or require `torch`.

This is an unreleased `main` / next-alpha candidate contract. The existing
`v0.1.0a5` tag predates `scenario_lite` and must not be documented as containing
these exports or examples.

## Implementation Consequence

The implementation and package documentation must:

1. re-export the listed scenario objects through `neural_abm.api_lite`;
2. test that the facade remains torch-free and that its export set is explicit;
3. keep domain callbacks and outcome meaning caller-owned;
4. distinguish mean-effect confidence intervals from empirical replicate
   intervals and label comparison booleans as mechanical report fields rather
   than scientific verdicts;
5. distinguish the released `v0.1.0a5` surface from the unreleased
   `main` / next-alpha candidate.

## Non-Goals

This decision does not:

- provide a general causal-inference or statistical-testing framework;
- calibrate a scenario model or certify external validity;
- turn a user-supplied success threshold into paper-claim judgment;
- promote `workflow_lite` as an independent stable facade;
- add torch-backed lifecycle objects to `api_lite`;
- claim that `v0.1.0a5` contains the scenario surface.
