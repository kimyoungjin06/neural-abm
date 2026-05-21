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
| Holdout migration | yes | yes | yes | partial | Toy5 now has a small threshold-aware topology/threshold grid, but negative-control separation is strongest on safety rather than spread. |
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
- Toy5 hard holdout now supports the unit lifecycle under a small
  topology/threshold grid, but exposure-anchor controls also spread in seeded
  cases, so threshold-aware uniqueness is not established.
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

Status: first pass complete.

Artifacts:

- `src/neural_abm/README.md`
- `docs/decisions/0010-nabm-unit-v1-contract.md`
- `docs/nabm-unit-v1-boundary-audit.md`
- `docs/nabm-unit-v1-completeness-checklist.md`
- `tests/test_nabm_unit_docs.py`

Completed work:

- Add a short contract note to `src/neural_abm/README.md` pointing to this
  checklist and the boundary audit.
- Keep Toy2/Toy4/Toy5 policy-unit guard tests active.
- Require a docs update whenever generic unit APIs gain new responsibilities.

Result:

- New unit changes can be classified as lifecycle, typed exchange, backend
  dispatch, diagnostics, or explicit contract-gap remediation.
- The README now states that generic unit code must not construct rewards,
  payoffs, thresholds, teacher signals, basin credit, readiness meaning,
  revision pressure meaning, or evidence criteria.
- The active guard surface is explicit: `tests/test_spatial_binary_runner.py`
  for unit-level binary lifecycle tests, `tests/test_toy2_runner.py`,
  `tests/test_toy4_runner.py`, and `tests/test_toy5_runner.py` for
  policy-unit adoption, `tests/test_readiness.py` for readiness propagation,
  and `tests/test_nabm_unit_docs.py` for documentation boundaries.

Completion condition update:

- Gate 1 is complete enough for v1 infrastructure work.
- Any future generic unit API expansion should update Decision 0010, the
  boundary audit, and this checklist in the same patch.

### Gate 2: Hard Holdout Expansion

Goal: turn Toy5 from a single hard holdout into bounded robustness evidence.

Status: first pass complete.

Artifacts:

- `experiments/evidence/toy5_neural_threshold_target_threshold_aware_grid_quick.yaml`
- `experiments/evidence/results/toy5_neural_threshold_target_threshold_aware_grid_quick.summary.md`
- `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_threshold_aware_grid_quick_profile.md`
- `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_threshold_aware_grid_quick_findings.md`

Completed work:

- Add a small topology/threshold grid for the threshold-aware Toy5 path.
- Preserve no-seed safety cases.
- Report baseline, main, and negative-control results separately.

Result:

- Gate status: `pass`.
- Main threshold-aware path: `5/5` final ceiling hits on no-seed safety and all
  six spread cases.
- Baseline output-average path: safe in no-seed, `0/5` final ceiling hits in
  all six spread cases.
- Negative controls: non-directional no-seed control fails safety, but
  exposure-anchor controls also achieve `5/5` final hits in all seeded spread
  cases.

Completion condition update:

- The bounded robustness claim is now supported for lattice `k=4`, lattice
  `k=6`, and rewired `k=6, p=0.10` at high thresholds `0.85` and `0.95`.
- A stronger uniqueness claim for threshold-aware direction remains open.

### Gate 3: Toy2/Toy4 Evidence Triage

Goal: separate algorithmic failure from gate brittleness and baseline-fit
effects.

Status: first pass complete.

Artifacts:

- `experiments/results/nabm_effect_matrix/evidence_profile_index_gate3.md`
- `experiments/results/nabm_effect_matrix/toy24_revision_operator_quick_profile.md`
- `experiments/results/nabm_effect_matrix/toy24_basin_credit_objective_blend_quick_profile.md`
- `experiments/results/nabm_effect_matrix/toy24_revision_operator_precommitment_controls_quick_profile.md`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick_profile.md`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_reputation_fragility_stress_quick_profile.md`
- `experiments/results/nabm_effect_matrix/toy24_gate3_evidence_triage_findings.md`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_reputation_fragility_stress_quick_findings.md`
- refreshed gate summaries with `trajectory_status` and `failure_mode` fields
  for the same Toy2/Toy4 manifest family.

Completed work:

- Keep final-epoch ceiling failures distinct from trajectory-level convergence.
- Report pure stochastic final flips separately from mechanism failures.
- Avoid adding more policy losses until the failure mode requires one.

Result:

- Toy2/Toy4 results can be classified as success, stochastic gate brittleness,
  baseline-favored environment, or true mechanism failure.
- `toy24_revision_operator_quick` is classified as stochastic gate brittleness
  plus baseline-favored environment for both Toy2 and Toy4.
- `toy24_basin_credit_objective_blend_quick` separates Toy2 slow TtC gate lag
  from Toy4 success.
- Precommitment/peer-evidence variants recover stable final hits, but several
  cases remain baseline-favored because reputation imitation reaches the same
  ceiling faster.
- The combined reputation-fragility stress now gives a positive targeted
  contrast: noisy reputation, sparse initial action seeds, and open boundaries
  drop reputation imitation to `0/5` final and ever ceiling hits in both Toy2
  and Toy4, while the peer-evidence candidate stays at `5/5` with mean TtC
  `9.4` and `9.0`.
- Gate JSON/Markdown now reports trajectory outcome separately from pass/fail,
  so final-epoch brittleness is visible without changing the gate threshold.

Completion condition update:

- Gate 3 is complete enough for first-pass evidence triage.
- The next Toy2/Toy4 evidence step should move beyond reputation-targeted
  perturbations and test non-reputation fragility, such as delayed/sparse
  reward, heterogeneous observation quality, or topology bottlenecks.

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

The next implementation slice should avoid another broad parameter sweep. Two
paths are now useful:

- Gate 3: triage Toy2/Toy4 evidence so stochastic gate brittleness,
  baseline-favored environments, and true mechanism failures are separated.
- Toy5 control sharpening: add a small case where exposure-only anchoring is
  expected to over-spread or self-excite, while the threshold-aware adapter
  should preserve safety.
