# Neural ABM

This project studies Neural Agent-Based Models (NABM) as a research toolkit and
prototype suite for agent-based modeling.

In this repository, a model is inside the NABM claim when it has:

- neural agents or policies whose local update changes the simulated behavior;
- an ABM time loop with explicit scheduling, state transition, and logging;
- a separated local update and social update path;
- social message, peer selection, typed mixing, and commit stages;
- ABM-style aggregate and micro-state outputs for inspection.

The current framing is conservative:

- Do not claim a general Transformer replacement.
- Do not claim a universal simulator or general-purpose ABM framework.
- Treat the core unit as a Neural ABM Node with learnable social mixing.
- Position the system as a temporal heterogeneous GNN-style simulator with
  neural agents, explicit social update rules, and ABM logging.

The starting reference is [ref/deep-research-report.md](ref/deep-research-report.md).

## Project Map

- [docs/project-structure.md](docs/project-structure.md): long-term repository
  structure and artifact policy.
- [docs/process-design.md](docs/process-design.md): research process, claims,
  decision gates, and experiment flow.
- [docs/implementation-plan.md](docs/implementation-plan.md): uv-based
  environment policy, module shape, and immediate pre-implementation plan.
- [docs/guidelines/neural-abm-node-guidelines.md](docs/guidelines/neural-abm-node-guidelines.md):
  design guidelines for the reusable Neural ABM Node.
- [docs/guidelines/agent-internal-design.md](docs/guidelines/agent-internal-design.md):
  internal agent contract and testing ladder.
- [docs/toy-models/README.md](docs/toy-models/README.md): initial toy model
  designs.
- [docs/decisions/0001-project-positioning.md](docs/decisions/0001-project-positioning.md):
  first architecture/research decision record.
- [docs/decisions/0003-social-update-pipeline.md](docs/decisions/0003-social-update-pipeline.md):
  social update decomposition into compatibility, peer selection, alignment,
  and typed mixing.
- [docs/decisions/0004-binary-spatial-runner-hooks.md](docs/decisions/0004-binary-spatial-runner-hooks.md):
  binary spatial runner hook lifecycle and Toy 2/4/5 responsibility split.
- [docs/decisions/0005-nabm-definition.md](docs/decisions/0005-nabm-definition.md):
  conservative NABM definition, claim boundary, and toy-suite classification.
- [docs/decisions/0006-domain-runner-and-evidence-matrix.md](docs/decisions/0006-domain-runner-and-evidence-matrix.md):
  common runner boundary and evidence-matrix contract.
- [docs/decisions/0007-basin-centric-relational-nabm-roadmap.md](docs/decisions/0007-basin-centric-relational-nabm-roadmap.md):
  basin-centric relational NABM target, current gap review, and improvement
  roadmap.

## Expected Outputs

- Paper drafts and figures in `paper/`.
- Internal research notes and design documents in `docs/`.
- Reusable simulation modules in `src/neural_abm/`.
- CLI or batch scripts in `scripts/`.
- Experiment configs, runs, and derived results in `experiments/`.
- Periodic snapshots and quick-search indexes in `archive/`.

## Development

Use `uv` for all Python work:

```bash
uv sync
uv run pytest
```

The project targets Python 3.14 or newer. Python 3.14.4 is the latest official
3.14 maintenance release as of 2026-04-29, while `uv` may resolve a managed
3.14.x interpreter depending on what its Python distribution index provides.
