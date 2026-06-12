# Neural ABM

[![CI](https://github.com/kimyoungjin06/neural-abm/actions/workflows/ci.yml/badge.svg)](https://github.com/kimyoungjin06/neural-abm/actions/workflows/ci.yml)

Neural ABM is a pre-release Python package for Neural Agent-Based Models
(NABM): simulations where neural local updates, explicit social exchange, and
ABM-style state logging are kept separate and inspectable.

The current alpha is distributed from Git tags before PyPI publication. The
default install is intentionally lightweight: it installs only `numpy` and
`pyyaml`, supports Python 3.11 or newer, and exposes the torch-free
`neural_abm.api_lite` surface.

In this repository, a model is inside the NABM claim when it has:

- neural agents or policies whose local update changes the simulated behavior;
- an ABM time loop with explicit scheduling, state transition, and logging;
- a separated local update and social update path;
- social message, peer selection, typed mixing, and commit stages;
- ABM-style aggregate and micro-state outputs for inspection.

The current framing is conservative:

- Do not claim a general Transformer replacement.
- Do not claim a universal simulator or general-purpose ABM framework.
- Treat the core unit as a Neural ABM Node with learnable social mixing.
- Position the system as a temporal heterogeneous GNN-style simulator with
  neural agents, explicit social update rules, and ABM logging.

## Project Map

- [docs/release-readiness.md](docs/release-readiness.md): remaining release
  gates before reserving final `0.1.0`.
- [docs/pypi-publishing.md](docs/pypi-publishing.md): Trusted Publishing and
  TestPyPI/PyPI workflow setup.
- [docs/toy-models/README.md](docs/toy-models/README.md): capability-first
  model-family roadmap.
- [docs/toy-models/capability-matrix.md](docs/toy-models/capability-matrix.md):
  current capability taxonomy and package catalog fields.
- [docs/package-release-boundary.md](docs/package-release-boundary.md):
  product-facing entry points, install profiles, and package checklist.
- [docs/git-distribution-flow.md](docs/git-distribution-flow.md): pre-PyPI
  Git commit/tag installation flow.
- [docs/pre-release-artifact-flow.md](docs/pre-release-artifact-flow.md):
  alpha artifact, wheel/sdist, and install-command validation flow.
- [docs/api-surface-audit.md](docs/api-surface-audit.md): stable,
  experimental, internal, and paper-only API boundary.
- [examples/README.md](examples/README.md): lightweight package-facing examples.

## Quick Start

Install the current Git alpha:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a2"
```

Query the torch-free toy catalog:

```python
from neural_abm.api_lite import toy_catalog, toys_by_taxonomy

catalog = toy_catalog()
binary_probability_toys = toys_by_taxonomy("output_family", "binary_probability")
```

Install the torch-backed lifecycle API only when needed:

```bash
uv pip install "neural-abm[torch] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a2"
```

```python
from neural_abm.api import NABMUnit, SocialBlock, SocialChannel
```

## Package Outputs

- Reusable simulation modules in `src/neural_abm/`.
- Package-facing examples in `examples/`.
- Release and profile-smoke scripts in `scripts/`.
- Public package docs in `docs/`.

## Public API and Package Status

Reusable code should start from the narrow stable facade:

```python
from neural_abm.api import NABMUnit, SocialBlock, SocialChannel
```

The package root is a lazy compatibility surface for existing module-path
imports. New code should use `neural_abm.api` for the stable torch-backed v0
contract. The first torch-free profile seed is `neural_abm.api_lite`; it exposes
runner, result, diagnostics, readiness utilities, NumPy-only social primitives,
and lightweight lifecycle reports/local-step primitives that can be imported
without loading torch. Its `SocialChannel` metadata is limited to
scalar/bounded scalar mix channels; distribution helpers remain standalone, and
tensor/state mixing requires the torch-backed API. Both `neural_abm.api` and
`neural_abm.api_lite` also expose feature-taxonomy helpers for mapping stable
model IDs to user-facing families.

The default package profile is now a lightweight torch-free install for the
`api_lite` surface. Full `NABMUnit` lifecycle work, `SocialBlock` tensor/state
mixing, and research workflows require explicit extras such as
`neural-abm[torch]`, `neural-abm[research]`, or `neural-abm[full]`.
Decision 0014 records the dependency policy and transition rules.
[Package release boundary](docs/package-release-boundary.md) records the
product-facing entry points and release checklist.

Torch-free catalog lookup:

```python
from neural_abm.api_lite import toy_catalog, toys_by_taxonomy

catalog = toy_catalog()
binary_probability_toys = toys_by_taxonomy("output_family", "binary_probability")
```

## Development

Use `uv` for all Python work:

```bash
uv sync
uv run pytest
```

`uv sync` installs the dev dependency group used by the full research test
suite. For package-profile checks, use `uv build` plus isolated installs of the
wheel or explicit extras. The release-smoke helper is:

```bash
uv run python scripts/smoke_package_profiles.py
```

The pre-release artifact inspection command is:

```bash
uv run python scripts/inspect_release_artifacts.py --build
```

The no-torch catalog example is:

```bash
uv run python examples/toy_catalog.py
```

Before PyPI, install from a committed Git ref or tag:

```bash
uv pip install "neural-abm @ git+file:///home/kimyoungjin06/Desktop/Workspace/1.4.6.Neural_ABM@<commit-or-tag>"
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a2"
```

The package currently supports Python 3.11 or newer.
