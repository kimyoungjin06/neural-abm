# Release Readiness

This document separates current alpha distribution from the conditions for a
final `0.1.0` release.

## Current Verdict

Do not reserve final `0.1.0` yet. The repository has a validated Git alpha
(`v0.1.0a2`) and the current `main` branch has a validated clone-first alpha
path with a lightweight default package profile. Final public release still
needs index-publishing and release-operations checks.

The current product-facing path before PyPI is:

```bash
git clone https://github.com/kimyoungjin06/neural-abm.git
cd neural-abm
uv run --no-dev python examples/toy_catalog.py
```

That path is now checked in CI by the `Smoke clone-first default environment`
step.

## Required Before Final 0.1.0

1. CI is present and green on `main`.
2. CI is green on the release tag or the same release commit.
3. TestPyPI upload succeeds from the built wheel and sdist.
4. TestPyPI install smoke confirms:
   - `Requires-Python: >=3.11`;
   - default install pulls only `neural-abm`, `numpy`, and `pyyaml`;
   - `torch` is neither installed nor loaded by the default profile;
   - `neural_abm.__version__` matches package metadata.
5. PyPI project ownership, token or trusted-publishing setup, and package-name
   availability are confirmed.
6. README install commands are switched from Git-tag installs to PyPI installs.
7. GitHub Release notes identify the alpha/final status, install commands,
   validation evidence, and known boundaries.
8. The Python floor is re-checked against source syntax and dependency wheels.
9. The public API boundary remains limited to `api_lite` and the torch-backed
   `api` facade unless a new decision record expands it.

## Alpha Policy

Use alpha tags while any of these are true:

- PyPI/TestPyPI publishing has not been exercised.
- CI has not passed on the release commit.
- The package-facing README or install commands are still changing.
- The release notes still describe the package as pre-public.
- Remaining release-owner decisions are unresolved.

## Next Operational Gate

Keep PyPI deferred until the Git-based alpha path is the stable onboarding
surface. The next operational gate is to decide whether the current `main`
state should be promoted to a new Git alpha tag, expected to be `v0.1.0a3`.

Before tagging that alpha, confirm:

1. README Quick Start begins with clone-first usage.
2. `docs/git-distribution-flow.md` documents the fresh clone smoke.
3. CI is green on the candidate commit.
4. A fresh remote clone runs the no-dev catalog smoke without loading torch.
5. `pyproject.toml` is intentionally bumped before any new alpha tag is cut.
6. Git tag install commands point at the new alpha tag after it exists.

After the clone-first alpha tag is cut and verified, follow
[pypi-publishing.md](pypi-publishing.md) to configure pending Trusted
Publishers for TestPyPI and PyPI. Do not change README install commands to PyPI
or reserve final `0.1.0` before the TestPyPI workflow has passed.

## Final Release Rule

Reserve `0.1.0` only when a user can install from PyPI, run the no-torch catalog
example, opt into torch-backed APIs with extras, and understand the package's
current boundaries from README and release notes without relying on internal
project context.
