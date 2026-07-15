# Neural ABM

[![CI](https://github.com/kimyoungjin06/neural-abm/actions/workflows/ci.yml/badge.svg)](https://github.com/kimyoungjin06/neural-abm/actions/workflows/ci.yml)

Neural ABM is a pre-release Python package for Neural Agent-Based Models
(NABM): simulations where neural local updates, explicit social exchange, and
ABM-style state logging are kept separate and inspectable.

The current alpha is clone-first. Start from a Git clone, use `uv`, and stay on
the default torch-free `neural_abm.api_lite` surface unless you explicitly need
the torch-backed lifecycle API. The current distribution path is the repository
and Git tags.

## Quick Start

Clone the current verified alpha tag and run the torch-free first-run check:

```bash
git clone --depth 1 --branch v0.1.0a5 https://github.com/kimyoungjin06/neural-abm.git neural-abm
cd neural-abm
uv run --no-dev python examples/first_run.py
```

The first-run output should report `status=ok`, `toy_count=10`, and
`torch_loaded=false`. Then inspect the full toy capability catalog:

```bash
uv run --no-dev python examples/toy_catalog.py
```

Query the lightweight API from the clone:

```bash
uv run --no-dev python - <<'PY'
from neural_abm.api_lite import toy_catalog, toys_by_taxonomy

catalog = toy_catalog()
binary_probability_toys = toys_by_taxonomy("output_family", "binary_probability")
print(len(catalog), binary_probability_toys)
PY
```

Use the default branch only when you intentionally want unreleased changes from
`main`. For early-user reports, reproduce against `v0.1.0a5` first.

Install from the current Git alpha tag only when you want to consume it as a
dependency:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
```

Then query the torch-free toy catalog from another project:

```python
from neural_abm.api_lite import toy_catalog, toys_by_taxonomy

catalog = toy_catalog()
binary_probability_toys = toys_by_taxonomy("output_family", "binary_probability")
```

Install the torch-backed lifecycle API only when needed:

```bash
uv pip install "neural-abm[torch] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
```

```python
from neural_abm.api import NABMUnit, SocialBlock, SocialChannel
```

## Main / Next-Alpha Candidate

`examples/research_pivot_scenario_lite.py` is not present in `v0.1.0a5`.
It is currently available on `main` as a candidate for the next alpha. Test it
in a separate checkout so that its results are not confused with the verified
tag path:

```bash
git clone --depth 1 --branch main https://github.com/kimyoungjin06/neural-abm.git neural-abm-next-alpha
cd neural-abm-next-alpha
uv run --no-dev python examples/research_pivot_scenario_lite.py
```

The example compares a science-of-science PIVOT question across baseline and
counterfactual environments through local adaptation, typed peer exchange,
domain-owned transition, and audit evidence.

## What You Just Ran

- `examples/first_run.py` imports only from `neural_abm.api_lite`.
- The default clone-first path should report `torch_loaded=false`.
- The package currently exposes 10 toy capability entries.
- The first toy families to inspect are binary-probability social-learning
  models; the full taxonomy is printed by `examples/toy_catalog.py`.
- On the optional `main` / next-alpha candidate path,
  `examples/research_pivot_scenario_lite.py` shows a no-torch
  science-of-science PIVOT scenario: baseline vs interdisciplinary seed grants
  vs hot-field hype, with productive-pivot outcomes and aggregate/micro audit
  rows.
- Python 3.11 or newer is supported.

## Troubleshooting

If the clone-first path fails, capture the command output before changing the
environment or installing optional extras. Use the same command order as the
Quick Start:

```bash
uv --version
python --version
git rev-parse --short HEAD
uv run --no-dev python examples/first_run.py
uv run --no-dev python examples/toy_catalog.py
```

Open an issue with the failed command, full output, operating system, Python
version, `uv --version`, the attempted Git ref, and whether you expected
torch-backed behavior. The early-user checklist is in
[docs/early-git-user-handoff.md](docs/early-git-user-handoff.md).

## Project Map

- [docs/release-readiness.md](docs/release-readiness.md): clone-first readiness
  and remaining gates before final `0.1.0`.
- [docs/git-distribution-flow.md](docs/git-distribution-flow.md): Git clone
  and commit/tag distribution flow.
- [docs/early-git-user-handoff.md](docs/early-git-user-handoff.md): stable,
  torch-backed, and experimental surfaces for early Git users.
- [docs/package-release-boundary.md](docs/package-release-boundary.md):
  product-facing entry points, install profiles, and package checklist.
- [examples/README.md](examples/README.md): lightweight package-facing examples.
- [docs/case-studies/researcher-pivot/README.md](docs/case-studies/researcher-pivot/README.md):
  end-to-end replicated research example (question, hypotheses, seed-paired
  replication, sensitivity sweeps, figures, limitations) on the torch-free
  surface.
- [docs/classical-reductions.md](docs/classical-reductions.md): exact DeGroot
  and Granovetter special cases, plus explicitly labeled FJ-like anchored and
  self-excluding HK variants, as settings of the same lifecycle.
- [docs/toy-models/README.md](docs/toy-models/README.md): capability-first
  model-family roadmap.
- [docs/toy-models/capability-matrix.md](docs/toy-models/capability-matrix.md):
  current capability taxonomy and package catalog fields.
- [docs/api-surface-audit.md](docs/api-surface-audit.md): stable,
  experimental, internal, and paper-only API boundary.
- [docs/pre-release-artifact-flow.md](docs/pre-release-artifact-flow.md):
  alpha artifact, wheel/sdist, and install-command validation flow.

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

## Scope

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
uv run --no-dev python examples/first_run.py
uv run --no-dev python examples/toy_catalog.py
```

The primary user path is a Git clone. Direct Git URL installs are also
supported from committed refs or tags:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
```

The package currently supports Python 3.11 or newer.
