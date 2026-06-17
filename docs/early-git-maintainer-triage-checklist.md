# Early Git Maintainer Triage Checklist

Use this checklist when a report arrives through the early Git issue template.
The goal is to reproduce once, classify once, and apply the smallest correction.

## First Pass

1. Confirm the issue used `.github/ISSUE_TEMPLATE/early-git-user-report.yml`.
2. Confirm it includes:
   - path used;
   - Git ref or package version;
   - failed command;
   - diagnostic output;
   - OS, Python, shell, and uv version;
   - torch expectation.
3. Add `early-git`.
4. Add exactly one primary taxonomy label.
5. Ask for missing diagnostics before attempting a fix if the command or
   environment is ambiguous.

## Primary Labels

| Label | Use When |
| --- | --- |
| `docs` | Commands work, but README, handoff, examples, or release notes are unclear. |
| `clone-smoke` | Fresh clone commands fail in a clean environment. |
| `git-install` | Direct Git tag install fails or reports mismatched metadata. |
| `dependency-profile` | Default profile pulls or loads torch, or an extra misses a dependency. |
| `api-boundary` | Documented `api_lite` or `api` surface breaks. |
| `environment` | Failure depends on OS, Python, shell, uv, network, or cache state. |
| `unsupported-surface` | Report targets internals outside the stable user contract. |

## Reproduction

Preferred helper:

```bash
uv run python scripts/reproduce_early_git.py --ref v0.1.0a5
```

This runs the fresh clone path and the direct Git tag install path in temporary
directories, then verifies version, package metadata, `toy_count=10`, and
torch-free default behavior.

Manual fresh clone fallback:

```bash
tmpdir=$(mktemp -d)
git clone --depth 1 --branch v0.1.0a5 https://github.com/kimyoungjin06/neural-abm.git "$tmpdir/neural-abm"
cd "$tmpdir/neural-abm"
uv run --no-dev python examples/first_run.py
uv run --no-dev python examples/toy_catalog.py
```

Manual direct Git tag install fallback:

```bash
tmpdir=$(mktemp -d)
UV_CACHE_DIR="$tmpdir/cache" uv run --isolated --no-project --python 3.11 \
  --with "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5" \
  python - <<'PY'
import importlib.metadata
import importlib.util
import json
import sys

import neural_abm
from neural_abm.api_lite import toy_catalog

print(json.dumps({
    "version": neural_abm.__version__,
    "metadata_version": importlib.metadata.version("neural-abm"),
    "toy_count": len(toy_catalog()),
    "torch_installed": importlib.util.find_spec("torch") is not None,
    "torch_loaded": "torch" in sys.modules,
}, sort_keys=True))
PY
```

## Fix Decision

Use a docs-only commit when the observed behavior already matches the current
contract and the issue is wording or discoverability.

Use a behavior fix when a clean reproduction contradicts README, handoff,
release-readiness, or CI baseline behavior.

Use a new alpha tag only when the fix changes:

- clone-first command behavior;
- direct Git tag install behavior;
- package metadata or dependency profile;
- stable `api_lite` or `api` imports;
- first-run or toy catalog output shape.

Close as unsupported when the report targets toy internals, paper workflows,
generated artifacts, or modules outside `api_lite`, `api`, and documented
examples. Point the user to `docs/early-git-user-handoff.md`.

## Regression Requirement

- README, handoff, issue template, label manifest, or triage checklist changes
  need coverage in `tests/test_clone_first_distribution.py`.
- Default dependency profile changes need
  `scripts/smoke_package_profiles.py` coverage.
- Artifact boundary changes need `scripts/inspect_release_artifacts.py`
  coverage.
- Public API behavior changes need package-level import or behavior tests.
