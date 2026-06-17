# Git Distribution Flow

This flow covers Git alpha distribution. The release unit is a Git commit or
annotated tag.

## Current Mode

The primary user path is a fresh clone of the current verified alpha tag. It
should exercise the default package profile without installing the dev
dependency group or loading torch:

```bash
git clone --depth 1 --branch v0.1.0a4 https://github.com/kimyoungjin06/neural-abm.git neural-abm
cd neural-abm
uv run --no-dev python examples/first_run.py
uv run --no-dev python examples/toy_catalog.py
uv run --no-dev python - <<'PY'
import sys

from neural_abm.api_lite import toy_catalog

assert len(toy_catalog()) == 10
assert "torch" not in sys.modules
PY
```

Use the default branch only when intentionally checking unreleased changes from
`main`; support reports should reproduce the verified tag path first.

The repository can also be used locally before pushing a release tag:

```bash
uv pip install -e .
uv run --no-dev python examples/toy_catalog.py
```

For a Git-style install from the local repository, use a committed ref. Git URL
installs read committed content, so uncommitted working-tree changes are not
part of the install:

```bash
uv pip install "neural-abm @ git+file:///path/to/neural-abm@<commit-or-tag>"
```

## Remote Git Mode

The remote repository is:

```text
https://github.com/kimyoungjin06/neural-abm
```

If it is not already configured, add it and push the release branch:

```bash
git remote add origin https://github.com/kimyoungjin06/neural-abm.git
git push -u origin main
```

Before tagging an alpha release, run the local release checks:

```bash
uv run ruff check src tests scripts
uv run pytest -q
git diff --check
uv run python scripts/inspect_release_artifacts.py --build
uv run python scripts/smoke_package_profiles.py --wheel dist/neural_abm-0.1.0a4-py3-none-any.whl
uv run --no-dev python examples/first_run.py
uv run --no-dev python examples/toy_catalog.py
```

Then create and push an annotated tag:

```bash
git tag -a v0.1.0a4 -m "neural-abm 0.1.0a4"
git push origin v0.1.0a4
```

Users can install the default package profile from the tag:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a4"
```

Extras use the same direct URL shape:

```bash
uv pip install "neural-abm[torch] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a4"
uv pip install "neural-abm[research] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a4"
uv pip install "neural-abm[full] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a4"
```

## Metadata

`project.urls` points at this repository and its issue tracker. Use a
`Documentation` URL only after documentation has a stable public location.

## Remaining Release Decisions

Git distribution does not remove these release-owner decisions:

- keep the package Python support floor at `requires-python = ">=3.11"` unless
  a compatibility pass justifies changing it;
- decide when to reserve final `0.1.0` instead of alpha tags.
