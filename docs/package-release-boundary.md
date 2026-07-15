# Package Release Boundary

This release boundary fixes the product-facing entry points for the current v0
package shape. It is a packaging contract, not a new simulation claim.

## Released Tag: `v0.1.0a5`

Default Git tag install:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
```

The default profile is the lightweight `api_lite` floor. It supports package
metadata, toy feature-taxonomy lookup, compatible runner/result helpers,
readiness utilities, NumPy-only scalar/bounded-scalar social helpers, and
torch-free lifecycle reports/local-step helpers. It must not import or require
`torch`.

Torch-backed API:

```bash
uv pip install "neural-abm[torch] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
```

Use this profile for `neural_abm.api`, `NABMUnit`, `NABMStep`, `SocialBlock`,
tensor/state-dict social messages, and the full tensor-backed lifecycle.

Research workflows:

```bash
uv pip install "neural-abm[research] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
uv pip install "neural-abm[full] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
```

Use these profiles for toy runners, evidence matrices, paper workflows,
plotting, and the full development/research stack.

## Main / Next-Alpha Candidate

The unreleased `main` / next-alpha candidate extends the default torch-free
`api_lite` floor with `scenario_lite` bounded-scalar scenario comparison
helpers, including seed-paired replicated runs with outcome distributions.
Its package metadata uses `0.1.0a6.dev0` so artifacts from `main` cannot be
mistaken for the verified `v0.1.0a5` release.
The `v0.1.0a5` tag predates `scenario_lite`; do not use that tag when testing
these researcher-facing exports or examples.
Install that candidate explicitly when evaluating this surface:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@main"
```

This candidate does not change the dependency boundary: `api_lite` and the
scenario helpers must not import or require `torch`.

## Public Entry Points

Use `neural_abm.api_lite` for no-torch metadata and lightweight package
operations available from the released tag:

```python
from neural_abm.api_lite import toy_catalog, toys_by_taxonomy

binary_toys = toys_by_taxonomy("output_family", "binary_probability")
catalog = toy_catalog()
```

On `main` / the next-alpha candidate, bounded-scalar researcher scenarios use
the same no-torch facade:

```python
from neural_abm.api_lite import (
    BoundedScalarScenarioSpec,
    ScenarioDefinition,
    run_bounded_scalar_scenarios,
)
```

The scenario spec's `success_direction` and `success_min_delta` are
user-provided comparison metadata. Returned `success`, `success_fraction`,
paired-mean confidence intervals, and separate empirical replicate intervals
are mechanical simulation reports, not framework judgment that a scientific
claim is supported. Domain semantics, calibration, identification assumptions,
and claim interpretation remain caller-owned; see
[Decision 0015](decisions/0015-researcher-scenario-lite-contract.md).

Use `neural_abm.api` for the stable torch-backed v0 lifecycle:

```python
from neural_abm.api import NABMUnit, SocialBlock, SocialChannel
```

Keep `neural_abm.__init__` as a lazy compatibility layer for existing
module-path imports. Product-facing examples and docs should prefer `api_lite`
or `api`.

## Toy Catalog Boundary

Stable model IDs such as `toy1` are reproducibility IDs for configs, artifacts,
tests, and paper references. User-facing selection should use feature taxonomy
fields:

```text
display_name
domain_family
state_family
output_family
topology_family
coordination_family
unit_surface
evidence_role
```

Run artifacts include these fields in `metadata.json` and `summary.json`. CSV
and sweep field contracts keep established NABM fields for compatibility.

## Release Checklist

Run this checklist before tagging a release candidate:

```bash
uv run ruff check src tests scripts
uv run pytest -q
git diff --check
uv run python scripts/inspect_release_artifacts.py --build
uv run python scripts/smoke_package_profiles.py
```

For a narrower default-profile check during iteration:

```bash
uv run python scripts/smoke_package_profiles.py --profiles default
uv run --no-dev python examples/toy_catalog.py
```

For the `main` / next-alpha candidate, also run:

```bash
uv run --no-dev python examples/research_pivot_scenario_lite.py
```

The artifact inspector should show that default wheel metadata does not require
`torch`, and the default-profile smoke should show `torch_loaded=false` while
blocking torch imports.

See [pre-release-artifact-flow.md](pre-release-artifact-flow.md) for the
metadata, version, Python-floor, wheel/sdist, and install-command review.

Use [git-distribution-flow.md](git-distribution-flow.md) for commit/tag based
installation from `https://github.com/kimyoungjin06/neural-abm`.

## Non-Goals

- Do not market `neural_abm.api` as no-torch while `NABMUnit`, `NABMStep`, and
  `SocialBlock` remain torch-backed.
- Do not move toy-owned payoff, resource, event, market, or evidence semantics
  into stable API metadata.
- Do not rename stable model IDs in artifacts or configs; feature names are the
  display layer.
- Do not treat a scenario comparison's `success=true` as scientific claim
  adjudication.
- Do not document the `v0.1.0a5` tag as containing the unreleased
  `scenario_lite` surface.
