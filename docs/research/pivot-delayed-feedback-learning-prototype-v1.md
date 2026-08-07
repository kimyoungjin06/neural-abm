# Pivot delayed-feedback learning prototype v1

Date: 2026-08-07

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_delayed_feedback_learning_prototype_v1.yaml`](../../experiments/configs/pivot_delayed_feedback_learning_prototype_v1.yaml)

## Purpose

This is the first executable successor to the Pivot-to-NABM flagship bridge.
It asks whether the existing NABM lifecycle produces materially different
adaptive-entry dynamics when successful and failed pivot outcomes become
publicly available on different schedules.

The object is deliberately small. It does not fit empirical data, change the
generic NABM API, select a model, or claim that delayed feedback is the real
researcher-pivot mechanism. Its purpose is to learn whether Pivot exposes a
useful lifecycle gap and whether the case generates a nontrivial phase map
worth developing.

## Fixed comparison

The simulation retains the predecessor's sampled researcher population,
stage-assortative network, `hot_field_hype` scenario, one-time pivot event,
productive-fit rule, readiness channel, and three learning policies:
`frozen`, `imitative`, and `cautionary`.

Only public outcome availability changes. The five registered regimes are:

| Regime | Success delay | Failure delay | Success visible | Failure visible |
|---|---:|---:|---:|---:|
| `immediate` | 0 | 0 | 1.0 | 1.0 |
| `symmetric_lag_2` | 2 | 2 | 1.0 | 1.0 |
| `failure_lag_2` | 0 | 2 | 1.0 | 1.0 |
| `failure_lag_4` | 0 | 4 | 1.0 | 1.0 |
| `selective_failure_lag_4` | 0 | 4 | 1.0 | 0.5 |

No unregistered delay, probability, scenario, policy, or hyperparameter sweep
may be added after inspecting the output of this tranche.

## Timing and state contract

The Pivot commit adapter realizes an outcome privately in step `s`. The domain
event queue then decides whether that outcome is eligible for public release
and schedules it for step `s + 1 + d`, where `d` is the outcome-specific
delay. A zero delay therefore reproduces the predecessor: a pivot realized in
one step can first train a neighbor in the next step.

Private realization and public availability are separate records. An eligible
message scheduled beyond the simulation horizon is reported as horizon
censored. An outcome excluded by the visibility probability is reported as
permanently unavailable. Public availability does not assert that every
neighbor received, understood, or believed the outcome.

Visibility uses a component-keyed uniform variate indexed by base seed,
replicate, agent, realization step, and outcome type. Regime and learning-arm
identifiers are excluded, allowing nested regimes to share a visibility draw
whenever the same outcome event occurs.

## Outputs

The runner retains raw replicate rows and creates a phase-map JSON, a compact
Markdown table, and a hash receipt. It reports final pivot composition,
private realization and public-release counts, release trajectories, paired
delay-minus-immediate contrasts, and paired learning-minus-frozen contrasts.

This tranche has no success threshold and no single-argmax peak label. The
registered outcomes include amplification, suppression, cautionary trade-offs,
mechanism equivalence, and no nontrivial phase behavior. Numeric differences
are descriptive behavior of the frozen synthetic model, not effects in the
world.

## Precommit and execution

`plan` validates the contract and source bindings without generating results.
`run` requires the full 40-character Git commit SHA containing the exact
contract, runner, tests, package-boundary rules, and design note. It refuses
dirty bound paths or source-binding mismatches. This precommit protects against
result-contingent rule changes; it is not an additional scientific gate or an
independent review.

The development profile fixes 80 agents, 14 steps, and 16 paired replicates.
The smaller smoke profile exists only for implementation validation. Result
interpretation begins only after the development execution has produced a
valid receipt.

## Framework boundary

The delayed queue remains owned by the Pivot prototype. The generic API is not
changed in this tranche. A reusable lifecycle abstraction should be proposed
only if a second model family needs the same private-to-public transition or a
concrete diagnostic shows that domain ownership prevents fair composition.

All prototype files and generated evidence remain repository-only. They do not
replace the hash-pinned public Pivot example or enter the source distribution.
