# Release Readiness

This document tracks the clone-first release path.

## Current Verdict

Do not reserve final `0.1.0` yet. The repository has a validated clone-first
alpha path and the next release target is `v0.1.0a4`.

The product-facing path is:

```bash
git clone https://github.com/kimyoungjin06/neural-abm.git
cd neural-abm
uv run --no-dev python examples/first_run.py
uv run --no-dev python examples/toy_catalog.py
```

That path is checked in CI by the `Smoke clone-first default environment` step.

## Required Before Final 0.1.0

1. CI is green on `main`.
2. CI is green on the release tag or the same release commit.
3. A fresh remote clone runs `examples/first_run.py` and
   `examples/toy_catalog.py` without installing the dev dependency group.
4. A direct Git tag install reports matching `neural_abm.__version__` and
   package metadata.
5. The default profile remains lightweight and does not install or load `torch`.
6. The README starts from clone-first usage and explains the no-torch default
   surface before advanced API or research context.
7. [early-git-user-handoff.md](early-git-user-handoff.md) identifies stable,
   experimental, and intentionally torch-backed surfaces for early users.
8. The public API boundary remains limited to `api_lite` and the torch-backed
   `api` facade unless a new decision record expands it.
9. No local filesystem paths or internal-only release planning instructions leak
   into README or package-facing docs.

## Alpha Policy

Use alpha tags while any of these are true:

- The clone-first README or example flow is still changing.
- CI has not passed on the release commit.
- The release notes still describe the package as pre-public.
- Early-user handoff, issue routing, or public API boundaries are unresolved.

## Completed Git Alpha Gate

The `v0.1.0a3` Git alpha gate is complete:

1. README Quick Start begins with clone-first usage.
2. `docs/git-distribution-flow.md` documents the fresh clone smoke.
3. CI is green on the release tag.
4. A fresh remote clone runs the no-dev catalog smoke without installing or
   loading torch.
5. `pyproject.toml` and `neural_abm.__version__` both report `0.1.0a3`.
6. Git tag install commands point at `v0.1.0a3`.
7. The GitHub Release is marked as a prerelease.

## Current Operational Gate

The current operational gate is `v0.1.0a4`, a clone-first alpha that captures:

1. `examples/first_run.py` as the first user command after clone.
2. README Quick Start before project-map and internal scope material.
3. GitHub Actions on Node 24 compatible action versions.
4. Early Git user handoff guidance.
5. Fresh remote clone and direct Git tag install smoke for the `v0.1.0a4` tag.

## Final Release Rule

Reserve `0.1.0` only when a user can clone the repository, run the torch-free
first-run and catalog examples, optionally consume a Git tag as a dependency,
and understand the package's current boundaries from README and handoff docs
without relying on internal project context.
