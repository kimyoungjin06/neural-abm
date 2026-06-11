# Release Readiness

This document separates current alpha distribution from the conditions for a
final `0.1.0` release.

## Current Verdict

Do not reserve final `0.1.0` yet. The repository has a validated Git alpha
(`v0.1.0a2`) with a lightweight default package profile, but final public
release still needs index-publishing and release-operations checks.

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

## Final Release Rule

Reserve `0.1.0` only when a user can install from PyPI, run the no-torch catalog
example, opt into torch-backed APIs with extras, and understand the package's
current boundaries from README and release notes without relying on internal
project context.
