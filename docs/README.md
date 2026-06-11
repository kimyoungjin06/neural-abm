# Docs Index

Use this directory for internal project knowledge.

## Core Docs

- [project-structure.md](project-structure.md): repository layout and artifact
  policy.
- [process-design.md](process-design.md): research process and decision gates.
- [implementation-plan.md](implementation-plan.md): uv environment policy,
  module shape, and immediate pre-implementation checklist.
- [archive-and-indexing.md](archive-and-indexing.md): snapshot and quick-search
  archive design.
- [nabm-unit-v1-boundary-audit.md](nabm-unit-v1-boundary-audit.md): concrete
  Toy2/Toy4 ownership audit for the reusable NABM Unit v1 contract.
- [nabm-unit-v1-completeness-checklist.md](nabm-unit-v1-completeness-checklist.md):
  operational completion map for the NABM Unit v1 engineering, evidence, and
  paper-readiness gates.
- [api-surface-audit.md](api-surface-audit.md): classification of stable,
  experimental, internal, and paper-only surfaces before a v0 public facade.
- [package-release-boundary.md](package-release-boundary.md): product-facing
  package entry points, install profiles, toy catalog boundary, and release
  checklist.
- [git-distribution-flow.md](git-distribution-flow.md): pre-PyPI Git
  commit/tag installation and metadata flow.
- [pre-release-artifact-flow.md](pre-release-artifact-flow.md): package
  metadata, version, Python floor, wheel/sdist contents, and install-command
  pre-release artifact flow.

## Guidelines

- [guidelines/neural-abm-node-guidelines.md](guidelines/neural-abm-node-guidelines.md):
  reusable Neural ABM Node design guidelines.
- [guidelines/agent-internal-design.md](guidelines/agent-internal-design.md):
  internal agent contract and test ladder.

## Toy Models

- [toy-models/README.md](toy-models/README.md): capability-first model-family
  roadmap.
- [toy-models/neural-hk-classification.md](toy-models/neural-hk-classification.md):
  detailed Toy 1 specification.
- [toy-models/neural-spatial-pd.md](toy-models/neural-spatial-pd.md):
  detailed Toy 2 specification.

## Decisions

- [decisions/0001-project-positioning.md](decisions/0001-project-positioning.md):
  accepted positioning of the project.
- [decisions/0002-uv-and-module-shape.md](decisions/0002-uv-and-module-shape.md):
  accepted Python environment and module layout decision.
- [decisions/0003-social-update-pipeline.md](decisions/0003-social-update-pipeline.md):
  accepted decomposition of social update into peer selection, alignment, and
  typed mixing.
- [decisions/0004-binary-spatial-runner-hooks.md](decisions/0004-binary-spatial-runner-hooks.md):
  accepted hook lifecycle and runner/domain responsibility split for binary
  spatial toys.
- [decisions/0005-nabm-definition.md](decisions/0005-nabm-definition.md):
  conservative NABM definition, claim boundary, and toy-suite classification.
- [decisions/0006-domain-runner-and-evidence-matrix.md](decisions/0006-domain-runner-and-evidence-matrix.md):
  accepted common runner boundary and evidence-matrix contract.
- [decisions/0007-basin-centric-relational-nabm-roadmap.md](decisions/0007-basin-centric-relational-nabm-roadmap.md):
  proposed basin-centric relational NABM target and gap-closing roadmap.
- [decisions/0008-basin-critic-borrowed-concepts.md](decisions/0008-basin-critic-borrowed-concepts.md):
  borrowed-concept ledger and structural plan for a learned basin critic.
- [decisions/0009-general-nabm-unit-priority.md](decisions/0009-general-nabm-unit-priority.md):
  accepted priority and migration history for the general NABM unit.
- [decisions/0010-nabm-unit-v1-contract.md](decisions/0010-nabm-unit-v1-contract.md):
  accepted v1 contract, domain boundary, and holdout migration gate for the
  reusable NABM unit.
- [decisions/0011-continuous-scalar-unit-contract.md](decisions/0011-continuous-scalar-unit-contract.md):
  accepted bounded continuous scalar exchange contract for non-probability
  social values.
- [decisions/0012-existing-toy-migration-parity-consolidation.md](decisions/0012-existing-toy-migration-parity-consolidation.md):
  accepted consolidation of Toy6-Toy10 typed social-exchange parity as an
  engineering boundary, not performance evidence.
- [decisions/0013-public-api-v0-contract.md](decisions/0013-public-api-v0-contract.md):
  accepted public API v0 boundary separating stable core, experimental internals,
  and paper evidence tooling.
- [decisions/0014-package-dependency-policy.md](decisions/0014-package-dependency-policy.md):
  accepted package dependency policy for lightweight API readiness, including
  the torch-free social-core and unit-core splits, torch-backed full lifecycle
  boundary, and optional-extra transition rules.
