# Decision 0009: General NABM Unit Priority

## Status

Accepted.

## Date

2026-05-18

## Context

The repository has strong evidence infrastructure, but Toy2/Toy4/Toy5 and the
compatible Toy6-10 implementations still repeat the same high-level lifecycle:

```text
observe -> local update -> social message -> peer selection -> social mix
  -> commit -> aggregate/micro logging
```

`SocialBlock`, `SocialChannel`, `NABMStep`, `BinarySpatialRunner`, and
`DomainToyRunner` already cover pieces of that lifecycle, but there was no
single reusable unit that represented the full agent-level step.

## Decision

The highest-priority architecture target is a general `NABMUnit` contract.
Paper consolidation and further basin-critic expansion should not displace this
work.

The first accepted scope is:

- `NABMUnit`: owns a sequence of `NABMAgent` instances, a `NABMStep`, a peer
  selector callback, and a social-value builder callback.
- `NABMUnit.run(...)`: executes local updates, validates social messages,
  selects peers, runs social mix/commit, collects logs, and returns
  `NABMUnitReport`.
- `NABMUnitReport`: exposes local losses, social messages, peer ids, social
  diagnostics, aggregate rows, and micro rows.
- Message-to-channel helpers: `scalar_message_values`, `tensor_message_values`,
  and `state_dict_values`.

## Migration Order

1. Keep domain payoff and environment transition code inside the current toy
   modules.
2. Move only the repeated agent lifecycle into `NABMUnit`.
3. First migrate Toy1-style classification social updates, because the agent
   already satisfies `NABMAgent`.
4. Then migrate one narrow Toy2/Toy4 social path, ideally output-distribution
   distillation, without changing basin-credit behavior.
5. Only after that should `toy_pd.py` and `toy_public_goods.py` be split into
   domain, policy, runner, and credit modules.

## 2026-05-18 Initial Migration Slice

Implemented the first runner-visible migration:

- Toy1 `output_average` still keeps its existing runner call site and public
  output contract.
- The helper now uses `NABMUnit` internally to validate social messages, use
  the injected peer selector, run `NABMStep`, and return the existing
  `NABMStepResult`.
- Aggregate and micro diagnostics still report
  `social_channel=probe_output_distribution` and
  `commit_mode=distillation_step`.

This intentionally avoids moving Toy1 domain data generation, metrics, or
logging into the generic unit. The next migration target is one Toy2/Toy4
output-distribution distillation path.

## 2026-05-18 Binary Distillation Migration Slice

Implemented the next narrow slice for binary social distillation:

- `spatial_binary.apply_binary_output_distribution_distillation` now owns the
  common `NABMUnit`, `NABMStep`, and `DistributionDistillationAdapter`
  construction for binary policy-distribution distillation.
- Toy2 `apply_output_average`, Toy4 `apply_output_average_distillation`, and
  Toy5 `apply_output_average_distillation` now call that shared primitive.
- The injected peer selector returns the runner-computed `peer_ids`, and the
  injected social-value builder returns the existing `previous_probs` teacher
  distribution. This keeps domain peer selection and teacher construction
  outside the generic unit for now.
- Existing loop-vs-batched distillation tests still pass for Toy2, Toy4, and
  Toy5, so the loop helper
  behavior remains aligned with the accelerated compatibility path.

This slice is intentionally limited to the loop backend. Tensor/batched
accelerators, payoff construction, basin-credit scoring, and CSV contracts are
unchanged.

## 2026-05-18 Binary Distillation Report Slice

Lifted the shared binary distillation primitive from loss-only output to a
runner-visible report contract:

- `run_binary_output_distribution_distillation` returns
  `BinaryOutputDistillationReport`, including social losses, aggregate
  diagnostics, micro diagnostics, and the optional `NABMUnitReport`.
- The existing `apply_binary_output_distribution_distillation` remains a
  loss-only compatibility wrapper.
- `distill_binary_policy_output_average` accepts either a loss vector or a
  `BinaryOutputDistillationReport`. When a report is returned, it stores
  `social_unit_aggregate` and `social_unit_micro` in
  `BinarySocialStepResult.extras`.
- Common binary aggregate/micro rows can now expose unit-owned fields such as
  `social_channel`, `commit_mode`, social update norms, and active social agent
  count.

This still does not move tensor/batched accelerators into `NABMUnit`. The
structural change is that the runner can now treat the loop social update as a
NABM unit result, rather than only a list of losses.

## 2026-05-18 Accelerated Backend Report Compatibility

Unified the runner-facing report shape across loop, batched, and tensor-batched
binary policy distillation:

- `BinaryOutputDistillationReport.from_accelerated_update_result` wraps accelerated
  `BatchedMLPUpdateResult` objects into the same aggregate/micro diagnostics
  contract used by the loop `NABMUnit` path.
- `from_batched_update_result` remains as a compatibility alias, but new code
  should use the accelerated name because this wrapper now mostly describes
  tensor-batched report compatibility rather than the batched unit path.
- Toy2, Toy4, and Toy5 `distill_policy` methods now return
  `BinaryOutputDistillationReport` for batched and tensor-batched branches.
- Aggregate CSV rows now receive `social_channel=policy_distribution` and
  `commit_mode=distillation_step` for all three neural update backends.
- Accelerated backends do not yet expose per-agent parameter delta norms, so
  `mean_social_update_norm`, `max_social_update_norm`, and micro
  `social_update_norm` are explicit `0.0` compatibility values for those
  branches.

This is a contract-unification step, not a claim that accelerated kernels now
execute inside `NABMUnit`. The later batched and tensor runtime adapter slices
supersede that boundary for policy-distillation social updates.

## 2026-05-18 Batched Commit Adapter Slice

Started moving accelerated execution ownership behind the generic unit without
making `NABMUnit` aware of accelerator caches:

- Added `BatchedDistributionDistillationAdapter`, a commit adapter that consumes
  `NABMStep` mixed distribution targets and applies batched MLP distillation
  through the existing accelerator primitives.
- Toy2, Toy4, and Toy5 `batched` policy-distillation branches now run through
  `run_binary_output_distribution_distillation` with that adapter, so social
  mix, social-message validation, commit, and report construction all occur
  inside the `NABMUnit` path.
- The adapter still owns accelerated details such as `BatchedMLPParameters`,
  `BatchedAdamStateCache`, model synchronization, optimizer-state
  synchronization, and active-agent subsets. This keeps those details out of
  `NABMUnit` itself.
- This superseded the batched portion of the prior report-compatibility note:
  after this slice, `0.0` update-norm compatibility values applied only to
  tensor-batched branches, while batched branches received social mix update
  norms from `NABMStep`.
- Direct adapter contract tests compare `BatchedDistributionDistillationAdapter`
  against the legacy batched update helper and assert that empty-peer social
  steps commit no agents and leave parameters unchanged.

This is the batched accelerated execution slice, not a full backend migration.

## 2026-05-18 Tensor Runtime Commit Adapter Slice

Moved tensor-batched social distillation behind the same generic unit boundary:

- Added `TensorRuntimeDistributionDistillationAdapter`, a commit adapter that
  consumes `NABMStep` mixed distribution targets, obtains trainable parameters
  from `TensorPolicyRuntime`, and commits losses through
  `runtime.apply_loss_gradients`.
- Toy2, Toy4, and Toy5 `tensor_batched` policy-distillation branches now call
  `run_binary_output_distribution_distillation` with that adapter instead of
  bypassing the unit through `apply_tensor_output_average_distillation_update`.
- Tensor runtime ownership remains outside `NABMUnit`: the adapter owns runtime
  parameter access, optimizer update calls, and timing hook forwarding, while
  runner code still controls when runtime state is flushed back to concrete
  agents.
- The shared binary distillation helper supplies minimal synthetic unit agents
  when a tensor runtime intentionally keeps no concrete per-agent objects in the
  runner state, as in Toy2. This preserves the `NABMUnit` lifecycle contract
  without forcing tensor runtimes to materialize dormant agents.
- Direct adapter tests compare the tensor runtime adapter against the legacy
  tensor update helper and assert empty-peer no-op behavior.

This completes the current Toy2/Toy4/Toy5 policy-distillation backend migration
behind `NABMStep`.

## 2026-05-18 Local Policy-Gradient Adapter Slice

Moved the backend commit portion of local policy-gradient updates behind a
generic local unit boundary:

- Added `LocalUpdateReport`, `LocalUpdateAdapter`, and `NABMLocalStep` to the
  reusable unit contract.
- Added `BatchedPolicyGradientLocalUpdateAdapter` and
  `TensorRuntimePolicyGradientLocalUpdateAdapter` for binary policy-gradient
  losses. These adapters own loss tensor construction and backend commit, while
  toy domains still own objective inputs.
- Toy2, Toy4, and Toy5 now compute domain-specific `actions`, `advantages`, and
  `active_agent_ids` first, then call the common local-step adapters for
  batched and tensor-batched gradient commits.
- This intentionally does not merge Toy2 counterfactual advantage, Toy4
  public-goods baseline/bootstrap logic, or Toy5 adoption utility into the
  generic unit. Those remain domain semantics; the common unit owns only the
  reusable learning commit protocol.
- Direct tests compare the batched and tensor local adapters against their
  legacy accelerator paths, and existing Toy2/Toy4/Toy5 batched local parity
  tests continue to compare against loop training.

## 2026-05-18 Toy5 Binary Policy Learning Unit Slice

Started lifting repeated policy-learning lifecycle wiring out of toy files:

- Added `BinaryPolicyLearningUnit` and `BinaryPolicyLearningResult` to the
  shared binary runner layer.
- Toy5 neural policy local steps now use that unit for pre-policy readout,
  decision probability construction, action sampling, local update commit,
  cache/runtime refresh, and post-local policy readout.
- Toy5 still owns observation construction, adoption sampling semantics,
  utility-proxy advantages, and backend-specific local update callbacks. This is
  lifecycle extraction, not domain-objective generalization.
- Direct tests verify callback ordering and returned policy-learning state, and
  Toy5 batched/tensor-batched runner parity tests continue to pass.

This slice is the template for Toy4 and Toy2 migration, but those domains should
move only after their bootstrap/basin and counterfactual objective hooks are
kept explicit.

## 2026-05-18 Toy4 Binary Policy Learning Unit Slice

Moved Toy4 neural policy lifecycle wiring behind the shared binary learning
unit while keeping public-goods semantics runner-owned:

- Toy4 neural local steps now use `BinaryPolicyLearningUnit` for pre-policy
  readout, decision probability construction, action sampling, local update
  commit, cache/runtime refresh, and post-local policy readout.
- Toy4 still owns observation construction, resource transition, payoff
  recomputation, decision bootstrap, distill bootstrap, state-continuation
  objective components, basin diagnostics handoff, and teacher-alignment
  diagnostics. Those remain explicit callbacks, not generic unit behavior.
- `BinaryPolicyLearningUnit` now accepts an optional post-readout callback so a
  domain can preserve raw pre-readout diagnostics while using temperature-
  adjusted post-local probabilities.
- Toy4 loop, batched, tensor-batched, and teacher-alignment regression slices
  pass after the migration.

This confirms the policy-learning unit can carry a more complex domain than
Toy5 without absorbing domain-objective semantics.

## 2026-05-19 Toy2 Binary Policy Learning Unit Slice

Completed the binary spatial policy-learning migration for the remaining full
NABM toy:

- Toy2 neural local steps now use `BinaryPolicyLearningUnit` for pre-policy
  readout, decision probability construction, action sampling, local update
  commit, cache/runtime refresh, and post-local policy readout.
- Toy2 still owns neural context peer selection, pairwise payoff recomputation,
  sampled-policy-gradient versus counterfactual-advantage target construction,
  decision bootstrap, distill bootstrap, basin handoff, and teacher-alignment
  diagnostics. These remain domain callbacks rather than generic unit logic.
- The pre-readout callback preserves Toy2's distinction between temperature-
  adjusted policy probabilities and raw decision-kernel probabilities.
- Toy2 loop, batched, tensor-batched, bootstrap, distill, and teacher-alignment
  regression slices pass after the migration.

With Toy2, Toy4, and Toy5 all using `BinaryPolicyLearningUnit`, the current
full-NABM binary spatial family shares the same neural policy lifecycle
contract. The next structural step should be extracting a typed callback
protocol only if it reduces duplication without hiding domain objectives.

## 2026-05-19 Binary Policy Learning Callback Boundary

After the Toy2/Toy4/Toy5 migration, the repeated constructor wiring was the next
source of structural duplication. The common part was not the objective logic;
it was the lifecycle callback shape:

- policy readout;
- decision probability construction;
- action sampling;
- local update commit;
- cache/runtime refresh;
- optional post-local policy readout.

Added `BinaryPolicyLearningCallbacks` plus typed callback protocols for those
six hooks, and updated Toy2, Toy4, and Toy5 to pass one callback container into
`BinaryPolicyLearningUnit`.

This is intentionally a boundary extraction, not a new objective abstraction.
Toy domains still own payoff context, resource dynamics, counterfactual
advantages, basin handoff, bootstrap, and teacher-alignment diagnostics. The
shared unit owns only lifecycle ordering and timing stages.

## Claim Boundary

This does not make the project a general-purpose ABM framework. It makes the
neural agent lifecycle reusable enough that Toy-specific code can stop owning
the common NABM update protocol.
