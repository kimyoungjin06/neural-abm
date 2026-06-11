# Pre-Release Artifact Flow

This flow treats the package as a pre-public alpha artifact. The goal is to
produce a clean installable package without carrying internal gate history into
the public-facing distribution.

## Reviewed Items

### 1. Distribution Metadata

Current status:

- `pyproject.toml` has package name, alpha version, description, README,
  Apache-2.0 license metadata, authors, keywords, classifiers, Python
  requirement, default dependencies, optional extras, build backend, and an
  explicit sdist include list.
- `project.urls` points at the GitHub repository and issue tracker.

Decision:

- The package uses Apache-2.0 for code, package-facing docs, examples, and
  release-smoke scripts.
- Git repository URLs are resolved for Git-based alpha distribution.

### 2. Version Policy

Current status:

- The project version is `0.1.0a1`.

Decision:

- Use alpha versions while the package is pre-public and still changing shape.
- Reserve final `0.1.0` for the first intentional public release.

### 3. Python Requirement

Current status:

- `requires-python = ">=3.14"`.

Decision:

- Keep Python 3.14 as the current research-runtime floor.
- Treat it as an adoption constraint, not a packaging defect.
- Do not lower the Python requirement without a separate compatibility test
  pass.

### 4. Wheel and Sdist Contents

Current status:

- The wheel contains the importable `neural_abm` package and the v0/lite API
  boundary modules.
- The sdist is intentionally public-facing: README, pyproject, selected docs,
  examples, release-smoke scripts, and source package.
- Internal audit/checklist, paper, experiment, archive, and generated-result
  surfaces are excluded from the sdist flow.

Artifact command:

```bash
uv run python scripts/inspect_release_artifacts.py --build
```

This script builds a wheel and sdist, inspects package metadata, checks default
dependencies and extras, verifies key wheel modules, verifies required source
files, and rejects internal-history paths in the sdist.

### 5. Install Commands

Before PyPI publication, use local built artifacts or Git refs.

Local artifact check:

```bash
uv build --out-dir dist
uv run --isolated --with dist/neural_abm-0.1.0a1-py3-none-any.whl python -c "import neural_abm.api_lite"
```

Git ref install after a remote/tag exists:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a1"
```

After PyPI publication, package docs may use:

```bash
uv pip install neural-abm
uv pip install 'neural-abm[torch]'
uv pip install 'neural-abm[research]'
uv pip install 'neural-abm[full]'
```

## Pre-Release Checklist

Run:

```bash
uv run ruff check src tests scripts
uv run pytest -q
git diff --check
uv run python scripts/inspect_release_artifacts.py --build
uv run python scripts/smoke_package_profiles.py
uv run python examples/toy_catalog.py
```

Pass criteria:

- No blocking issues from `inspect_release_artifacts.py`.
- Default dependency metadata remains limited to `numpy` and `pyyaml`.
- Required extras include `torch`, `research`, and `full`.
- Artifact metadata reports no default `torch` requirement, and default wheel
  smoke reports `torch_loaded=false` while blocking torch imports.
- `examples/toy_catalog.py` runs without loading `torch`.
- The sdist contains public-facing docs/examples/scripts and excludes internal
  gate, paper, experiment, archive, and generated-result paths.

Release-owner decisions:

- Decide whether the Python 3.14 floor is acceptable for the first public
  release.

## Non-Goals

- Do not broaden the API surface during pre-release artifact hardening.
- Do not change toy semantics, artifact IDs, or evidence claims.
- Do not mark the package public-ready until the remaining release-owner policy
  is explicitly resolved.
