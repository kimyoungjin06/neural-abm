# Decision 0014: Package Dependency Policy for Lightweight API Readiness

## Status

Accepted.

## Date

2026-06-04

## Context

Decision 0013 selected `neural_abm.api` as the narrow stable v0 facade. Gate 8F
added that facade and Gate 8G proved it with a minimal example plus a
wheel-style import smoke.

That smoke also exposed the next packaging problem: the current package is not
lightweight. `pyproject.toml` installs the research, visualization, analysis,
and torch runtime stack as default dependencies. The current stable facade also
imports `unit.py` and `social.py`, and both modules import `torch` at module
load time. As a result, `neural_abm.api` is a narrow API boundary but not yet a
lightweight no-torch API boundary.

The project still targets three outputs:

- an internal reusable module;
- a paper evidence package;
- a public Python package.

Those outputs can share code, but they should not force the same dependency
profile on every user.

## Decision

The public package has a small default dependency set. The default install is
the `api_lite` floor and keeps only dependencies needed by the torch-free facade:

- `numpy`
- `pyyaml`

Research, analysis, visualization, CLI, and accelerator-heavy dependencies move
behind explicit extras in `project.optional-dependencies`:

| Extra | Dependency profile | Intended use |
| --- | --- | --- |
| `config` | `pydantic` | configuration schemas and config loaders. |
| `torch` | `torch` | torch-backed lifecycle, tensor social messages, accelerator helpers, and runtime-backed examples. |
| `research` | `networkx`, `pandas`, `pyarrow`, `pydantic`, `scikit-learn`, `scipy`, `torch`, `tqdm` | toy runners, evidence workflows, scientific analysis utilities, graph generation, and paper result processing. |
| `plot` | `matplotlib` | plotting and figure generation. |
| `cli` | `tqdm` | progress-reporting conveniences for long-running scripts. |
| `full` | all research, plot, CLI, config, and torch dependencies | full local research and compatibility environment. |

The dev dependency group also includes the full research stack so `uv sync` can
continue to prepare the local environment for the repository's full test suite.

The full `neural_abm.api` facade remains torch-backed because it imports
`unit.py` and `social.py`, and those modules import `torch` at module load time.
That is acceptable as long as package-facing copy starts no-torch users from
`neural_abm.api_lite` and tells lifecycle/tensor users to install a torch-backed
extra.

The first import-time split is `neural_abm.api_lite`. It is a torch-free facade
seed for compatible runner, diagnostics, result, readiness utilities,
torch-free social primitives such as scalar/bounded scalar `SocialChannel`
metadata, peer validators, peer selection helpers, and NumPy scalar/bounded
scalar mixing, plus lightweight lifecycle reports/local-step primitives.
Distribution validators and peer selection helpers remain available as
standalone helpers, but `api_lite.SocialChannel` accepts only scalar/bounded
scalar mix channel kinds. It is not a replacement for `neural_abm.api`, because
it intentionally excludes `NABMUnit`, `NABMStep`, `SocialBlock`,
tensor/state-dict social messages, and other torch-backed lifecycle surfaces.

The broad package root remains available as a compatibility surface for
existing module-path imports, but it must be lazy. `import neural_abm` must not
import `torch`; only requesting a torch-backed symbol such as `NABMUnit` should
load the corresponding torch-backed module.

## Transition Rules

Dependency removal must follow these rules:

1. Do not remove a default dependency only because it looks unused from the
   stable facade. First prove the no-dependency or reduced-dependency import
   path in an isolated environment.
2. Do not remove `torch` from the `torch`, `research`, or `full` extras while
   `neural_abm.api`, `unit.py`, and `social.py` remain torch-backed.
3. Do not remove toy-runner dependencies from `research` or `full` unless the
   paper and evidence workflows still have a documented install profile that
   runs their existing commands.
4. Do not expose toy-owned semantics, evidence criteria, generated manifests,
   or paper claim judgment as stable API just to justify a dependency.
5. Any future `pyproject.toml` dependency-profile change must update this
   decision, the package README guidance, and the dependency policy tests in the
   same patch.

## Implementation Consequence

The built-wheel profile smokes prove that:

- the default profile imports the package root and `neural_abm.api_lite` without
  installed or loaded `torch`;
- the `torch` extra imports the torch-backed `neural_abm.api` facade;
- the `research` extra imports representative research dependencies, config
  schemas, evidence-manifest types, and Toy6 runner symbols;
- the `full` extra imports plotting, research, torch-backed API, config, and
  Toy10 runner symbols.

The dependency profile split is now in `pyproject.toml`, and
`scripts/smoke_package_profiles.py` is the release-smoke command for these
profiles.

The next implementation slice should:

1. decide whether the lifecycle side of v0 remains explicitly torch-backed or
   whether additional lifecycle surfaces should be refactored into torch-free
   modules;
2. update developer/test installation guidance so full research tests still run
   under `uv`.

The default package profile can be marketed as the lightweight no-torch
`api_lite` profile. The full stable v0 facade should not be marketed as a
no-torch API while `NABMUnit` and `SocialBlock` remain torch-backed.

## Amendment: Torch-Free Social Core

The first social import split moved NumPy-only social exchange primitives into
`neural_abm.social_core` and re-exported them through `neural_abm.api_lite`.
This includes `SocialChannel`, `PeerSelectionResult`, `SocialMixResult`, scalar
and bounded-scalar channel constants, peer-id utilities, NumPy validators,
similarity helpers, peer selection helpers, and scalar/bounded scalar mix
helpers.
The `neural_abm.api_lite` facade narrows `SocialChannel` to scalar/bounded
scalar mix channel kinds so distribution helpers stay standalone and tensor or
state-dict channel lifecycles remain torch-backed.

`neural_abm.social` remains a compatibility module and still imports `torch`
because it owns `PeerIndexCache`, probability-distribution tensor mixing,
tensor-channel mixing, state-dict mixing, and `SocialBlock`. The full
`neural_abm.api` facade remains torch-backed for the same reason.

## Amendment: Torch-Free Lifecycle Reports

The first lifecycle import split moved torch-free lifecycle reports and
diagnostics into `neural_abm.unit_core` and re-exported them through
`neural_abm.api_lite`. This includes `CommitReport`, `SocialDiagnostics`,
`social_diagnostics(...)`, `CommitAdapter`, `LocalUpdateReport`,
`LocalUpdateAdapter`, `NABMLocalStep`, `NABMStepResult`, `PeerSelector`, and
`SocialValueBuilder`.

`neural_abm.unit` remains a compatibility module and still imports `torch`
because it owns `ObservationSpec`, `SocialMessageSpec`, tensor value builders,
torch-backed distillation adapters, `NABMStep`, and `NABMUnit`. A future no-torch
full lifecycle API would need a separate contract rather than rebranding the
current `NABMUnit` surface.

## Non-Goals

This decision does not:

- remove torch-backed functionality or research dependencies from the explicit
  extras/dev environment;
- claim the current package is product-ready;
- change the stable public API selected by Decision 0013;
- change evidence manifests, paper artifacts, or toy-runner semantics;
- define final semantic-versioning rules.
