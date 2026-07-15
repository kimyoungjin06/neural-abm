# Early Git User Handoff

Use this handoff for users trying `neural-abm` from a Git clone or Git tag.
The current release path is repository-first; assume `uv` and Python 3.11 or
newer.

## Current Use Path

Start with the current verified alpha tag:

```bash
git clone --depth 1 --branch v0.1.0a5 https://github.com/kimyoungjin06/neural-abm.git neural-abm
cd neural-abm
uv run --no-dev python examples/first_run.py
uv run --no-dev python examples/toy_catalog.py
```

The first-run output should report `status=ok`, `toy_count=10`, and
`torch_loaded=false`.

Use `main` only when intentionally checking unreleased changes. If a user
reports a failure from `main`, ask them to reproduce the same commands against
`v0.1.0a5` before treating it as a release-path regression.

Use a Git tag install only when consuming the package from another project:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
```

## Main / Next-Alpha Candidate

`examples/research_pivot_scenario_lite.py` is not present in `v0.1.0a5`.
It is a candidate for the next alpha and currently must be tested from `main`,
separately from the verified-tag support path:

```bash
git clone --depth 1 --branch main https://github.com/kimyoungjin06/neural-abm.git neural-abm-next-alpha
cd neural-abm-next-alpha
uv run --no-dev python examples/research_pivot_scenario_lite.py
```

The researcher scenario should report `surface=neural_abm.scenario_lite`,
`base_surface=neural_abm.api_lite`, `default_profile=torch-free`, and
`torch_loaded=false`.

## Stable Surfaces

- `neural_abm.api_lite`: torch-free package metadata, toy taxonomy, runner and
  result helpers, readiness utilities, scalar social primitives, and
  lightweight lifecycle reports/local-step primitives.
- `examples/first_run.py`: the first clone smoke for new users.
- `examples/toy_catalog.py`: the current capability catalog and taxonomy
  browser.
- `docs/toy-models/capability-matrix.md`: user-facing toy family map.

## Intentionally Torch-Backed Surfaces

- `neural_abm.api`: stable v0 lifecycle facade for `NABMUnit`, `NABMStep`,
  `SocialBlock`, and `SocialChannel`.
- `neural_abm.unit` and `neural_abm.social`: compatibility modules that may
  load torch at import time.
- The `torch`, `research`, and `full` extras.

Install these only when the torch-backed lifecycle or research stack is needed:

```bash
uv pip install "neural-abm[torch] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5"
```

## Experimental or Internal Surfaces

- Toy runner internals outside the documented examples.
- Evidence matrices, basin diagnostics, paper workflows, and generated result
  artifacts.
- Any module not routed through `neural_abm.api_lite`, `neural_abm.api`, or the
  documented examples.

Do not treat these as stable user contracts unless a later decision record or
release note promotes them.

## What To Report

When opening an issue, include:

- OS and Python version.
- `uv --version`.
- The command that failed.
- The Git tag or `git rev-parse HEAD`.
- The full output from `examples/first_run.py` or `examples/toy_catalog.py`.
- Whether torch was expected for the attempted command.

## Minimal Diagnostic Bundle

For a fresh clone failure, run these commands from the repository root and paste
the full output into the issue:

```bash
uv --version
python --version
git rev-parse --short HEAD
uv run --no-dev python examples/first_run.py
uv run --no-dev python examples/toy_catalog.py
```

The researcher-scenario command is intentionally excluded from this verified
tag diagnostic bundle. For a report about the `main` / next-alpha candidate,
record `git branch --show-current` and run
`uv run --no-dev python examples/research_pivot_scenario_lite.py` separately.

For a direct Git tag install failure, include the exact install command and a
short import smoke:

```bash
uv run --isolated --no-project --python 3.11 \
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

Expected default-profile values are `version=0.1.0a5`,
`metadata_version=0.1.0a5`, `toy_count=10`, `torch_installed=false`, and
`torch_loaded=false`.
