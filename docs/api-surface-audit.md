# API Surface Audit

Date: 2026-06-04

## Purpose

This audit separates the reusable core, paper evidence layer, and public package
surface before adding a new facade API. The goal is to make the stable surface
smaller, not to publish every currently importable object.

The current package has useful reusable infrastructure, but
`neural_abm.__init__` is broader than the intended public v0 API. Treat the
current top-level exports as a lazy compatibility surface for existing
module-path imports.

## Release Targets

| Target | API implication |
| --- | --- |
| Internal reusable module | Stabilize lifecycle, typed exchange, runner, schema, and result surfaces that other local projects can reuse. |
| Paper evidence package | Keep manifests, scripts, result profiles, and claim matrices reproducible without making them general library APIs. |
| Public Python package | Expose a compact facade, examples, and installable metadata while keeping research internals out of the stable namespace. |

## Stable Core Candidates

These are the best candidates for a future `neural_abm.api` facade because they
own lifecycle or typed exchange mechanics rather than domain semantics.

| Module | Candidate surface | Reason |
| --- | --- | --- |
| `neural_abm.unit` | `NABMUnit`, `NABMStep`, `NABMLocalStep`, commit reports, social specs, value builders, diagnostics | Generic lifecycle and typed callback plumbing. |
| `neural_abm.social` | `SocialChannel`, `SocialBlock`, `SocialMixResult`, `PeerSelectionResult`, channel constants, scalar/bounded/distribution mix helpers | Typed social exchange and validation. |
| `neural_abm.domain_runner` | `DomainRunSettings`, `DomainToyRunner`, `DomainToyAdapter`, `make_domain_run_dir`, `write_domain_run_metadata` | Compatible-toy outer lifecycle and artifact plumbing. |
| `neural_abm.domain_social_diagnostics` | `aggregate_social_diagnostic_fields`, `micro_social_diagnostic_fields` | Semantic-free row mapping for peer/social diagnostics. |
| `neural_abm.results` | `DomainToyResult` and stable artifact summary helpers | Result envelope and artifact output contract. |
| `neural_abm.readiness` | `BinaryReadinessPropagationUnit` and report types | Generic peer-readiness aggregation after domains define readiness. |
| `neural_abm.scenario_lite` | Scenario definitions/specs, deterministic and replicated result envelopes, replicate context, and bounded-scalar scenario runners | Torch-free baseline/counterfactual orchestration whose domain callbacks, outcome meaning, and comparison threshold remain caller-owned. |

## Experimental Core Candidates

These are reusable enough for internal work, but should be marked experimental
until their semantics and caller expectations are narrower.

| Module | Surface | Reason to keep experimental |
| --- | --- | --- |
| `neural_abm.spatial_binary` | `BinaryPolicyLearningUnit`, `run_binary_policy_learning_step`, binary aggregate/micro helpers | Useful lifecycle plumbing, but tightly coupled to Toy2/Toy4/Toy5 binary runner shape. |
| `neural_abm.binary_revision` | `BinaryRevisionLearningUnit`, stay/switch helpers, revision probability mappers | Structurally useful, but evidence is still prototype-level and gate-sensitive. |
| `neural_abm.accelerator` and `neural_abm.binary_neural` | backend resolution, tensor runtime, batched update helpers | Important engineering internals, but too runtime-specific for first public facade. |
| `neural_abm.mobility` and `neural_abm.reputation` | local mobility and reputation helpers | Reusable in toys, but domain interpretation can leak into public use. |

## Internal or Paper-Only Surfaces

These should not be added to a stable facade in the first pass.

| Surface | Classification | Boundary |
| --- | --- | --- |
| Toy2/Toy4/Toy5 objective, basin, bootstrap, teacher, threshold, resource, and revision-pressure code | Internal | These modules own domain semantics and paper diagnostics. |
| Toy6-Toy10 runner modules | Internal or example-compatible | They are compatible migration cases, not stable framework APIs. Their run functions can remain importable through module paths. |
| `experiments/evidence/*.yaml` | Paper-only | Manifests support named claims and diagnostics, not general library behavior. |
| `scripts/` evidence and sweep commands | Paper-only or developer tooling | Reproducibility commands belong in evidence docs, not stable package imports. |
| `paper/` and `experiments/results/` artifacts | Paper-only | They document claim support and limitations. |
| `basin_phase_critic`, learned diagnostics, time-to-ceiling diagnostics | Paper-only or experimental | Useful research tools, but not public core behavior. |

## Do Not Export as Stable

Do not expose these as stable v0 API responsibilities:

- payoff, reward, resource, market, event, group, threshold, categorical strategy,
  teacher, basin-credit, readiness-meaning, or revision-pressure construction;
- evidence criteria, pass/fail logic, or paper claim interpretation;
- generated experiment manifests or result directories;
- backend cache internals, tensor-runtime state mutation, or accelerator-specific
  optimizer storage;
- individual toy `step(...)` phase order.

## Current Export Gap

`neural_abm.__init__` currently exports useful examples and internals in one
place. It includes stable candidates such as `NABMUnit`, `NABMStep`, and
`SocialChannel`, but it also includes accelerator and runtime helpers that should
not define the public v0 contract.

Do not remove those exports in this audit slice. The package root can remain a
lazy compatibility layer, but it should not define the public v0 contract.

## Recommended v0 Facade Shape

When implementation starts, prefer a small facade rather than widening
`__init__.py`.

| Namespace | Contents |
| --- | --- |
| `neural_abm.api` | Stable lifecycle, typed social exchange, domain runner, diagnostics, result envelope, and toy feature-taxonomy helpers. |
| `neural_abm.api_lite` | Torch-free seed surface for compatible runner, diagnostics, result envelope, toy feature-taxonomy helpers, readiness utilities, NumPy-only social primitives, lifecycle reports/local-step primitives, and the `main` / next-alpha bounded-scalar scenario candidate. |
| `neural_abm.experimental` | Binary policy lifecycle, binary revision lifecycle, accelerator/tensor runtime helpers if needed. |
| `neural_abm.paper` or evidence docs | Manifest loading, gate summaries, and reproducibility commands if a library namespace is justified. |

The first facade should support internal reuse and examples. It should not claim
that the project is a finished general-purpose ABM framework.

`neural_abm.api_lite` is not a replacement for the full stable v0 facade. It is
the first import-time split for package profiles: it intentionally excludes
`NABMUnit`, `NABMStep`, `SocialBlock`, typed tensor/state-dict social lifecycle
helpers, and other torch-backed surfaces while allowing torch-free
scalar/bounded scalar `SocialChannel` metadata, peer-selection, validation,
similarity, NumPy scalar mix helpers through `neural_abm.social_core`, and
report/diagnostic/local-step helpers through `neural_abm.unit_core`.
Distribution helpers are available as standalone helpers in the lite facade;
distribution mix channels, tensor channels, and state-dict channels stay on the
torch-backed API path.
Toy feature-taxonomy helpers are shared by the full and lite facades because
they are metadata-only and do not require torch.

## Main / Next-Alpha Scenario Candidate

The unreleased `main` / next-alpha candidate extends `neural_abm.api_lite`
with `ScenarioDefinition`, `BoundedScalarScenarioSpec`, deterministic and
replicated comparison/result records, `ReplicationSpec`,
`ScenarioReplicateContext`, and the two bounded-scalar scenario runners. The
underlying `workflow_lite` module remains an implementation layer rather than a
separately promoted facade.

This surface standardizes torch-free execution and report shape, not research
semantics. Agent construction, topology, local updates, domain transitions,
outcome meaning, and scientific interpretation remain caller-owned.
`success_direction` and `success_min_delta` are user-provided comparison
metadata. A returned `success` value only records mechanical satisfaction of
that threshold; it is not framework judgment that a scientific claim is
supported. Seed pairing and percentile intervals likewise do not supply causal
identification, hypothesis testing, calibration, or external-validity claims.

See
[Decision 0015](decisions/0015-researcher-scenario-lite-contract.md) for the
complete contract. The released `v0.1.0a5` tag predates this scenario surface.
