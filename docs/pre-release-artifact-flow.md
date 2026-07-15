# Pre-Release Artifact Flow

This flow treats the package as a Git-distributed alpha artifact. The goal is to
produce a clean installable package without carrying internal gate history into
the user-facing distribution.

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

- The verified Git alpha remains `v0.1.0a5`; unreleased `main` uses the
  distinct next-alpha development identity `0.1.0a6.dev0`.

Decision:

- Use alpha versions while the package is still changing shape.
- Reserve final `0.1.0` for the first release that should be treated as stable.

### 3. Python Requirement

Current status:

- `requires-python = ">=3.11"`.

Decision:

- Use Python 3.11 as the package support floor because the source syntax,
  `tomllib` usage, default dependencies, and package smoke checks are
  compatible with Python 3.11+.
- Treat newer local research runtimes as allowed, not required.
- Do not change the Python requirement without a separate compatibility test
  pass.

### 4. Wheel and Sdist Contents

Current status:

- The wheel contains the importable `neural_abm` package and the v0/lite API
  boundary modules.
- The sdist is intentionally public-facing: README, pyproject, selected docs,
  examples, release-smoke scripts, source package, and the claim-bearing
  researcher-pivot data and rendered figures advertised by README.
- Internal audit/checklist, paper, experiment, archive, and unrelated generated
  result surfaces are excluded from the sdist flow.

Artifact command:

```bash
uv run python scripts/inspect_release_artifacts.py --build
```

This script builds a wheel and sdist, inspects package metadata, checks default
dependencies and extras, verifies the runtime and metadata versions agree,
verifies key wheel modules and required source files, checks internal Markdown
link targets, and rejects internal-history paths in the sdist.

### 5. Install Commands

Use local built artifacts or Git refs.

Local artifact check:

```bash
uv build --out-dir dist
uv run --isolated --with dist/neural_abm-0.1.0a6.dev0-py3-none-any.whl python -c "import neural_abm.api_lite"
```

Git ref install after a remote/tag exists:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
```

## Pre-Release Checklist

Run:

```bash
uv run ruff check src tests scripts
uv run pytest -q
git diff --check
uv run python scripts/inspect_release_artifacts.py --build
uv run python scripts/smoke_package_profiles.py
uv run --no-dev python examples/toy_catalog.py
```

Pass criteria:

- No blocking issues from `inspect_release_artifacts.py`.
- Default dependency metadata remains limited to `numpy` and `pyyaml`.
- Required extras include `torch`, `research`, and `full`.
- Artifact metadata reports no default `torch` requirement, and default wheel
  smoke reports `torch_loaded=false` while blocking torch imports.
- `examples/toy_catalog.py` runs without loading `torch`.
- The sdist contains the advertised public docs/examples/scripts and their
  claim-bearing case-study artifacts, while excluding internal gate, paper,
  experiment, archive, and unrelated generated-result paths.

Release-owner decisions:

- Re-check the Python 3.11 floor before final `0.1.0` if dependency wheels or
  syntax requirements change.
- Use [release-readiness.md](release-readiness.md) as the final `0.1.0`
  reservation checklist.

## Non-Goals

- Do not broaden the API surface during pre-release artifact hardening.
- Do not change toy semantics, artifact IDs, or evidence claims.
- Do not mark the package stable until the remaining release-owner policy is
  explicitly resolved.
