# PyPI Publishing

This document prepares TestPyPI and PyPI publishing through Trusted Publishing.
It does not mark the package as final.

## Workflow

The repository publish workflow is:

```text
.github/workflows/publish.yml
```

It is intentionally manual-only through `workflow_dispatch`.

Publish targets:

- `testpypi`: uploads to TestPyPI and then runs a TestPyPI install smoke.
- `pypi`: uploads to PyPI and then runs a PyPI install smoke.

The PyPI target has two extra guards:

- the workflow must run from `refs/tags/v<version>`;
- the `confirm_pypi` input must exactly match
  `publish neural-abm <version> to pypi`.

## Trusted Publisher Setup

Configure pending Trusted Publishers before the first upload.

TestPyPI:

- URL: `https://test.pypi.org/manage/account/publishing/`
- PyPI project name: `neural-abm`
- Owner: `kimyoungjin06`
- Repository name: `neural-abm`
- Workflow filename: `publish.yml`
- Environment name: `testpypi`

PyPI:

- URL: `https://pypi.org/manage/account/publishing/`
- PyPI project name: `neural-abm`
- Owner: `kimyoungjin06`
- Repository name: `neural-abm`
- Workflow filename: `publish.yml`
- Environment name: `pypi`

TestPyPI and PyPI use separate accounts and separate Trusted Publisher
registrations.

## GitHub Environments

The GitHub repository uses two environments:

- `testpypi`
- `pypi`

Keep `testpypi` unprotected for alpha iteration. Require manual approval for
the `pypi` environment before any final publish. This matches PyPA guidance for
production PyPI publishing and keeps TestPyPI iteration separate from final
release operations.

## TestPyPI Run

Use the GitHub Actions UI:

1. Open **Actions**.
2. Select **Publish Python package**.
3. Click **Run workflow**.
4. For a tagged alpha, run from the matching tag, for example `v0.1.0a2`.
5. Select `target=testpypi`.
6. Set `version` to the current alpha version, for example `0.1.0a2`.

The workflow builds wheel/sdist, validates package metadata, uploads to
TestPyPI, and then runs a default install smoke from TestPyPI.

The install smoke confirms:

- `Requires-Python: >=3.11`;
- `neural_abm.__version__` matches package metadata;
- the default install does not install or load `torch`;
- the toy catalog contains 10 entries.

## PyPI Run

Use this only after TestPyPI has passed and the final release version is chosen.

1. Push the release tag, for example `v0.1.0`.
2. Open **Actions**.
3. Select **Publish Python package**.
4. Run the workflow from the release tag.
5. Select `target=pypi`.
6. Set `version` to the exact package version.
7. Enter `confirm_pypi` as:

```text
publish neural-abm <version> to pypi
```

For `0.1.0`, the confirmation must be:

```text
publish neural-abm 0.1.0 to pypi
```

## Current Status

As of `v0.1.0a2`, the project is ready for TestPyPI setup but has not been
published to TestPyPI or PyPI.

Do not switch README install commands from Git tags to PyPI until the TestPyPI
smoke and the final PyPI publish path are both proven.
