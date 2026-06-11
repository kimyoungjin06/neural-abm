# Git Distribution Flow

This flow covers pre-PyPI alpha distribution. The release unit is a Git commit
or annotated tag, not a package-index upload.

## Current Mode

The repository can be used locally before pushing a release tag:

```bash
uv pip install -e .
uv run python examples/toy_catalog.py
```

For a Git-style install from the local repository, use a committed ref. Git URL
installs read committed content, so uncommitted working-tree changes are not
part of the install:

```bash
uv pip install "neural-abm @ git+file:///home/kimyoungjin06/Desktop/Workspace/1.4.6.Neural_ABM@<commit-or-tag>"
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
uv run python scripts/smoke_package_profiles.py --wheel dist/neural_abm-0.1.0a1-py3-none-any.whl
uv run python examples/toy_catalog.py
```

Then create and push an annotated tag:

```bash
git tag -a v0.1.0a1 -m "neural-abm 0.1.0a1"
git push origin v0.1.0a1
```

Users can install the default package profile from the tag:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a1"
```

Extras use the same direct URL shape:

```bash
uv pip install "neural-abm[torch] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a1"
uv pip install "neural-abm[research] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a1"
uv pip install "neural-abm[full] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a1"
```

## Metadata

`project.urls` points at this repository and its issue tracker. Use a
`Documentation` URL only after documentation has a stable public location.

## Remaining Non-Git Decisions

Git distribution does not remove these release-owner decisions:

- decide whether `requires-python = ">=3.14"` is acceptable for the first
  public release;
- decide when to reserve final `0.1.0` instead of alpha tags.
