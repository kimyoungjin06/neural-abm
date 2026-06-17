# Early Git Feedback Loop Design

This design defines how to turn early Git user reports into reproducible fixes
without widening the package scope or drifting away from the clone-first release
path.

## Goal

Make `v0.1.0a4` and later Git alpha reports actionable within one triage pass:

1. identify which entry path failed;
2. reproduce the failure with a bounded command;
3. classify whether the issue is docs, packaging, API boundary, dependency
   profile, or user environment;
4. decide whether the fix needs a patch commit, an alpha tag, or only a docs
   clarification.

## Inputs

The feedback loop starts from these surfaces:

- README `Troubleshooting`, which tells users what to capture first.
- `.github/ISSUE_TEMPLATE/early-git-user-report.yml`, which collects the
  failed path, command, diagnostics, environment, and torch expectation.
- `docs/early-git-user-handoff.md`, which defines stable, intentionally
  torch-backed, and experimental surfaces.
- CI `Smoke clone-first default environment`, which is the baseline pass
  signal for clone-first usage.

## Supported Entry Paths

### Fresh Clone

Primary path:

```bash
git clone https://github.com/kimyoungjin06/neural-abm.git
cd neural-abm
uv run --no-dev python examples/first_run.py
uv run --no-dev python examples/toy_catalog.py
```

Expected default result:

- `status=ok`;
- `toy_count=10`;
- `torch_loaded=false`.

### Direct Git Tag Install

Dependency-style path:

```bash
uv run --isolated --no-project --python 3.11 \
  --with "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a4" \
  python -c "from neural_abm.api_lite import toy_catalog; print(len(toy_catalog()))"
```

Expected default result:

- package metadata matches `neural_abm.__version__`;
- `toy_count=10`;
- `torch` is not installed or loaded.

### Torch-Backed Extras

Torch-backed reports are valid when the user intentionally installed
`neural-abm[torch]`, `neural-abm[research]`, or `neural-abm[full]`. These should
not be mixed with default-profile failures during triage.

## Triage Taxonomy

Use one primary category per issue:

- `docs`: the command works, but README, handoff, or examples made the path
  unclear.
- `clone-smoke`: fresh clone commands fail in a clean environment.
- `git-install`: direct Git tag install fails or reports mismatched metadata.
- `dependency-profile`: default install pulls or loads torch, or an extra omits
  an expected dependency.
- `api-boundary`: user used a documented stable surface and hit an import or
  behavior break.
- `environment`: failure depends on OS, Python, shell, uv, network, or cache
  state and cannot be reproduced from the repository alone.
- `unsupported-surface`: report uses toy internals, paper workflows, generated
  artifacts, or modules outside `api_lite`, `api`, and documented examples.

## Triage Procedure

1. Confirm the report includes the issue template fields.
2. Reproduce the same entry path in a temporary directory or isolated uv run.
3. Compare the result against CI baseline behavior.
4. Assign one primary category from the taxonomy.
5. Decide the smallest correction:
   - docs-only clarification;
   - example or handoff correction;
   - packaging/profile fix;
   - public API boundary fix;
   - close as unsupported with a pointer to the stable surface.
6. Add or update a regression test before closing any bug that changes behavior.

## Reproduction Commands

Fresh clone:

```bash
tmpdir=$(mktemp -d)
git clone --depth 1 --branch v0.1.0a4 https://github.com/kimyoungjin06/neural-abm.git "$tmpdir/neural-abm"
cd "$tmpdir/neural-abm"
uv run --no-dev python examples/first_run.py
uv run --no-dev python examples/toy_catalog.py
```

Direct Git tag install:

```bash
tmpdir=$(mktemp -d)
UV_CACHE_DIR="$tmpdir/cache" uv run --isolated --no-project --python 3.11 \
  --with "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a4" \
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

## Regression Rules

- Docs-only fixes need a test in `tests/test_clone_first_distribution.py` when
  they affect README, handoff, issue template, or release readiness contracts.
- Default-profile behavior fixes need coverage in
  `tests/test_package_dependency_policy.py` or `scripts/smoke_package_profiles.py`.
- Artifact content fixes need coverage in
  `scripts/inspect_release_artifacts.py` and
  `tests/test_release_artifact_inspection.py`.
- Public API fixes need direct import or behavior tests in the relevant package
  test file before release notes are updated.

## Release Decision

Use a new alpha tag when a fix changes one of these user-visible surfaces:

- clone-first command behavior;
- direct Git tag install behavior;
- package metadata or dependency profile;
- stable `api_lite` or `api` imports;
- first-run or toy catalog output shape.

Use a normal main commit without a new tag for:

- issue template wording;
- docs that only clarify already-working behavior;
- triage taxonomy refinements;
- internal test wording.

## Non-Goals

- Do not make package upload part of this loop.
- Do not promote toy internals or paper workflows to stable user contracts.
- Do not make `neural_abm.api` a no-torch surface while it remains
  intentionally torch-backed.
- Do not broaden supported Python versions without a separate compatibility
  pass.

## Next Implementation Steps

1. Create repository labels matching the triage taxonomy.
2. Add a maintainer triage checklist to pull requests or issue comments.
3. Add a small script that runs the fresh-clone and direct-Git reproduction
   commands for the latest alpha tag.
4. Decide whether the next alpha should include that script as a package-facing
   support tool or keep it maintainer-only.
