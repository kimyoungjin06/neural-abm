# NABM Unit v1 Completeness Checklist

Date: 2026-05-21

## Purpose

This checklist turns Decision 0010 and the boundary audit into an operational
completion map. It separates engineering completion from research evidence and
paper readiness so that future work does not confuse more implementation with a
stronger NABM claim.

Status terms:

- `Implemented`: code path exists and is used by at least one toy or backend.
- `Guarded`: focused tests prevent silent drift away from the unit contract.
- `Evidenced`: a quick manifest, holdout, or diagnostic artifact supports the
  claim under a named condition.
- `Paper-ready`: the claim is bounded enough to appear in a manuscript without
  overstating generality.

## V1 Completion Scorecard

| Surface | Implemented | Guarded | Evidenced | Paper-ready | Current judgment |
| --- | --- | --- | --- | --- | --- |
| Generic unit lifecycle | yes | yes | partial | no | `NABMUnit`, `NABMStep`, and `NABMLocalStep` exist, but full runner ownership is still partial. |
| Binary policy lifecycle | yes | yes | yes | partial | Toy2, Toy4, and Toy5 route non-revision neural local policy steps through shared policy plumbing. |
| Binary revision lifecycle | yes | partial | partial | no | Optional stay/switch unit exists for Toy2/Toy4, but evidence remains prototype-level and gate-sensitive. |
| Readiness propagation | yes | yes | yes | partial | Toy5 hard holdout supports threshold-aware readiness under named stress cases. |
| Social distillation | yes | yes | partial | partial | Output-distribution mix and commit diagnostics are unit-backed across loop, batched, and tensor paths. |
| Backend local commits | yes | yes | partial | no | `NABMLocalStep` wraps batched/tensor policy-gradient commits, but backend claims are engineering claims, not NABM novelty claims. |
| Domain diagnostics plumbing | yes | yes | partial | partial | Toy2/Toy4 shared diagnostic field plumbing reduces schema drift without moving semantics into the unit. |
| Holdout migration | yes | yes | yes | partial | Toy5 is a real holdout, but broader robustness needs a larger topology/threshold grid. |
| Evidence gate integration | yes | partial | yes | partial | Manifests and profile index exist, but some gate criteria remain brittle for stochastic final-epoch failures. |
| Manuscript narrative | partial | no | partial | no | The claim boundary exists in docs, but paper outline and figures do not yet absorb the current evidence. |

## What Is Complete Enough

The following pieces are complete enough to treat as v1 infrastructure:

- `src/neural_abm/unit.py::NABMUnit`, `NABMStep`, and `NABMLocalStep` as generic
  lifecycle primitives.
- `src/neural_abm/spatial_binary.py::BinaryPolicyLearningUnit` as the binary
  neural policy lifecycle owner.
- `src/neural_abm/spatial_binary.py::run_binary_policy_learning_step` as
  semantic-free callback plumbing for policy learning.
- `src/neural_abm/binary_revision.py::BinaryRevisionLearningUnit` as an opt-in
  revision lifecycle primitive.
- `src/neural_abm/readiness.py::BinaryReadinessPropagationUnit` as peer
  readiness aggregation after a domain defines readiness.
- `src/neural_abm/domain_learning_diagnostics.py` as schema and diagnostic
  plumbing for domain-learning extras.

These surfaces should now be protected. New toys should adapt to them first,
and generic unit changes should be treated as contract changes rather than
ordinary toy implementation details.

## What Is Not Complete

The project should not yet claim that the full NABM architecture is finished.
The remaining gaps are:

- Runner ownership is still split. `BinarySpatialRunner` and domain classes
  still hold substantial orchestration, environment transition, and logging
  logic.
- Toy2/Toy4 evidence is not clean enough to claim general algorithmic
  superiority over hand-coded baselines.
- Revision-operator evidence is structurally useful but not final enough to
  serve as a primary mechanism claim.
- Toy5 hard holdout supports the unit lifecycle under meaningful stress, but
  only on a bounded grid of topology and threshold settings.
- Paper artifacts do not yet express the current boundary: unit lifecycle
  reuse, domain-owned semantics, and bounded robustness evidence.

## Claim Boundary

Current supported claim:

> The project has a reusable neural ABM unit contract that can run binary
> policy learning, social propagation, readiness propagation, backend commits,
> and diagnostics across Toy2, Toy4, and Toy5 without moving payoff, resource,
> threshold, teacher, or basin semantics into the generic layer.

Current unsupported claims:

- Neural ABMs generally outperform Fermi, RD, reputation imitation, or
  threshold baselines.
- Basin credit is a finalized learned critic.
- Revision operators are a solved structural mechanism.
- Toy6-10 are full NABM evidence cases.
- The codebase is ready to be presented as a general-purpose ABM framework.

## Next Completion Gates

### Gate 1: Unit Contract Freeze

Goal: prevent silent expansion of generic unit semantics.

Required work:

- Add a short contract note to `src/neural_abm/README.md` pointing to this
  checklist and the boundary audit.
- Keep Toy2/Toy4/Toy5 policy-unit guard tests active.
- Require a docs update whenever generic unit APIs gain new responsibilities.

Completion condition:

- New unit changes can be classified as lifecycle, typed exchange, backend
  dispatch, diagnostics, or explicit contract-gap remediation.

### Gate 2: Hard Holdout Expansion

Goal: turn Toy5 from a single hard holdout into bounded robustness evidence.

Required work:

- Add a small topology/threshold grid for the threshold-aware Toy5 path.
- Preserve no-seed safety cases.
- Report baseline, main, and negative-control results separately.

Completion condition:

- The holdout claim can state exactly which topology and threshold regimes are
  robust and which remain open.

### Gate 3: Toy2/Toy4 Evidence Triage

Goal: separate algorithmic failure from gate brittleness and baseline-fit
effects.

Required work:

- Keep final-epoch ceiling failures distinct from trajectory-level convergence.
- Report pure stochastic final flips separately from mechanism failures.
- Avoid adding more policy losses until the failure mode requires one.

Completion condition:

- Toy2/Toy4 results can be classified as success, stochastic gate brittleness,
  baseline-favored environment, or true mechanism failure.

### Gate 4: Manuscript Claim Matrix

Goal: turn the codebase into a paper-ready evidence package.

Required work:

- Create a claim-to-artifact table linking each claim to code path, manifest,
  result, figure, and limitation.
- Promote only bounded claims to the paper outline.
- Keep prototype mechanisms out of the primary claim path unless their evidence
  is upgraded.

Completion condition:

- A reviewer can trace each paper claim to a reproducible artifact and a stated
  limitation.

## Recommended Next Slice

The next implementation slice should be Gate 1: freeze the unit contract in
developer-facing docs. This is cheaper and higher leverage than another
mechanism sweep because it prevents the reusable unit from absorbing domain
semantics as the project grows.
