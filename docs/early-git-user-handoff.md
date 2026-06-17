# Early Git User Handoff

Use this handoff for users trying `neural-abm` from a Git clone or Git tag.
The current release path is repository-first; assume `uv` and Python 3.11 or
newer.

## Current Use Path

Start with a fresh clone:

```bash
git clone https://github.com/kimyoungjin06/neural-abm.git
cd neural-abm
uv run --no-dev python examples/first_run.py
uv run --no-dev python examples/toy_catalog.py
```

The first-run output should report `status=ok`, `toy_count=10`, and
`torch_loaded=false`.

Use a Git tag install only when consuming the package from another project:

```bash
uv pip install "neural-abm @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a4"
```

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
uv pip install "neural-abm[torch] @ git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a4"
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
