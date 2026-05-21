# Decision 0004: Binary Spatial Runner Hooks

Status: Accepted

Date: 2026-05-06

## Context

Toy 2, Toy 4, and Toy 5 all use binary spatial actions, revision masks,
optional output averaging, optional neural policy distillation, reputation, and
in some cases payoff EMA and mobility. Before the hook refactor, each toy runner
kept its own step orchestration, which made it easy for equivalent policy modes
to drift in subtle ways.

The shared behavior belongs in one runner. Toy domains should own only the
domain-specific parts: observations, payoff/resource/exposure state, candidate
probabilities, local learning, and terminal metrics.

## Decision

`BinarySpatialRunner` owns the binary spatial lifecycle for Toy 2, Toy 4, and
Toy 5. Hook domains implement `BinarySpatialDomain`; Toy 2/4/5 adapters inherit
from `BinaryToyDomainBase` for common peer selection, output-similarity
coordination, neural policy distillation, post-step payoff/reputation/mobility
updates, generic aggregate/micro row assembly, and `BinaryToyResult` summary
writing. The runner also validates hook return values at runtime, so new binary
domains fail at the contract boundary instead of writing bad rows or mutating
state with malformed arrays.

One epoch follows this order:

```text
build_step_context
-> local_step
-> select_peers
-> social coordination
-> action selection
-> commit_actions
-> common post-step updates
-> finalize_hook_step
-> BinaryPolicyStepResult
```

The runner owns:

- revision mask sampling and realized revision rate calculation
- `none` versus `output_average` coordination branching
- output-average probability mixing for classical binary paths
- neural social distillation dispatch
- deciding whether current actions are sampled by the runner or already fixed
  by the domain
- validation of binary action/probability/loss/peer/update payloads returned by
  hooks
- common payoff EMA, reputation, and mobility updates
- assembly of `BinaryPolicyStepResult`
- writer management and log cadence

The domain owns:

- observation tensors and local context construction
- candidate action probabilities
- domain-specific payoff, resource, exposure, and threshold transitions
- neural local training
- action commit side effects that are not common binary state updates
- domain-specific finalization such as Toy 4 collapse time and Toy 5 time to
  50 percent adoption
- domain-specific aggregate and micro-state field extensions

This decision is a breaking public API cleanup for Toy 2/4/5. YAML configs use
only `run`, `simulation`, `model`, `domain`, and `logging` at the top level.
`model` contains `policy`, `agents`, `coordination`, and `state`; `domain`
contains `toy` plus toy-specific environment/game/graph data. Legacy top-level
`policy`, `coordination`, and `state` fields are not loaded.

Shared row schemas use generic binary names such as `action_rate`,
`action_probability`, `mean_policy_action_probability`,
`policy_action_probability_pre_revision`, and
`realized_decision_action_probability`. Toy-specific row and summary fields use
`domain_*` names. Terminal summaries return `BinaryToyResult` with
`final_action_rate`, `final_mean_policy_action_probability`, common
reputation/payoff fields, and `domain_metrics`.

## Hook Payloads

The lifecycle hooks pass the same `BinarySpatialState` through the epoch and
return typed dataclass payloads:

- `build_step_context(epoch, state, revision_mask)` returns a
  `BinaryStepContext`. `revision_mask` must be a one-dimensional bool array with
  one value per agent. `extras` must be a mapping and is for domain-owned context
  shared with later hooks.
- `local_step(state, context)` returns a `BinaryLocalStepResult`.
  `candidate_action_probs` must be a finite one-dimensional vector in `[0, 1]`,
  `local_losses` must contain one finite numeric value per agent, and
  `extras` must be a mapping. `actions_after_revision` is optional; when present
  it must contain one binary integer/bool action per agent and the runner will
  commit it without sampling current actions.
- `select_peers(action_probs, state, context, local_result)` returns
  `list[list[int]]`. The outer list must have one entry per agent and every peer
  id must be in bounds. Domains should use `peer_ids_for_binary_mixer` when they
  need the shared `mixer=none` empty-peer convention and domain-specific error
  labels.
- Social coordination returns a `BinarySocialStepResult`. `final_action_probs`
  must be a finite one-dimensional vector in `[0, 1]`, `social_losses` must
  contain one finite numeric value per agent, `peer_ids` must pass the same peer
  validation, and `extras` must be a mapping.
- If actions were not fixed by `local_step`, `sample_actions(state,
  action_probs, revision_mask, context, local_result)` returns the current action
  vector. The runner validates one binary integer/bool value per agent before
  commit.
- `commit_actions(state, actions, context, local_result, social_result)` applies
  domain side effects such as assigning `state.actions` and payoffs. It returns a
  mapping of public step extras. Keys beginning with `_` are treated as private
  and are filtered out of `BinaryPolicyStepResult.extras`.
- `post_step_state_update(state, context, local_result, social_result)` returns a
  `BinaryPostStepStatePolicy`. The runner applies requested payoff EMA,
  reputation, and mobility updates after `commit_actions` and before
  `finalize_hook_step`. Mobility requests must include neighbors and an RNG.
- `finalize_hook_step(state, context, local_result, social_result,
  mobility_result)` returns a mapping. It may include `extras`, which must itself
  be a mapping and is merged into public step extras after private-key filtering.
  It may also include `post_social_probs` to override the result readout after
  final domain updates; that value must expose binary action-1 probabilities via
  `[:, 1]`, and those probabilities must be finite and in `[0, 1]`.

`BinaryPolicyStepResult` contains the local and social probability readouts,
loss vectors, active peer ids, the sampled revision mask, mobility result,
realized revision rate, and public extras collected from `local_step`, social
coordination, `commit_actions`, and `finalize_hook_step`.

## Social Modes

`BinaryLocalStepResult.social_mode` selects the current-action semantics.

`probability_mix` is the classical path. The domain returns candidate binary
action probabilities. If `coordination.mixer == "output_average"`, the runner
mixes those probabilities with selected peers, then samples the current epoch's
actions from the mixed probabilities. If the mixer is `none`, the runner samples
from the local probabilities.

`policy_distill` is the neural path. The domain has already sampled or chosen
the current epoch's actions and returns them as `actions_after_revision`. If
`coordination.mixer == "output_average"`, the runner calls the domain's
distillation hook. Distillation updates policy/readout state and
`post_social_probs`, but the runner does not resample the current epoch's
actions.

`mixer=none` bypasses social coordination. The runner derives
`final_action_probs` from the local policy readout and returns zero social losses.

## New Binary Toy Checklist

- Define a domain adapter that implements `BinarySpatialDomain` directly or
  inherits `BinaryToyDomainBase`, with `micro_state_fields` and
  `aggregate_fields` matching the generic binary CSV schema plus domain fields.
- Return `BinarySpatialState` from `initial_state`, including common binary
  arrays for actions, payoffs, payoff EMA, previous payoff EMA, and reputation.
- Provide initial and per-step probability readouts that expose action-1
  probabilities through the binary policy helpers in `spatial_binary.py`.
- Keep `candidate_action_probs`, `final_action_probs`, losses, actions, revision
  masks, peer ids, and extras within the validated shapes described above.
- Choose `social_mode="probability_mix"` when the runner should mix scalar
  candidate probabilities and sample actions, or `social_mode="policy_distill"`
  when the domain fixes current actions and owns neural distillation.
- Put domain-only commit/finalize outputs under private `_` keys or omit them;
  put public row/summary values in hook extras intentionally.
- Use common helpers such as `binary_loss_metrics`, `binary_peer_metrics`,
  `binary_peer_component_map`, `binary_reputation_metrics`,
  `binary_mobility_metrics`, `binary_micro_common_fields`,
  `binary_micro_mobility_fields`, `binary_policy_prob`, and
  `mean_binary_policy_prob` where they match existing row semantics.
- Add fake-domain contract tests for any new shared runner behavior, plus
  toy-specific regression tests for public metrics, logs, and summaries.

## Testing Contract

Runner-level contract tests use a fake binary domain in
`tests/test_spatial_binary_runner.py`. These tests fix the shared semantics
independently from Toy-specific golden metrics:

- `probability_mix` samples current actions from output-averaged probabilities.
- `mixer=none` bypasses social mixing and uses local probabilities.
- `policy_distill` commits local actions without resampling.
- common payoff EMA and reputation updates run after `commit_actions` and before
  `finalize_hook_step`.
- mobility runs before `finalize_hook_step` and its result is included in the
  step result.

Toy-specific tests verify that Toy 2, Toy 4, and Toy 5 preserve numerical
behavior while exposing the generic binary public API.

## Consequences

- Toy 2, Toy 4, and Toy 5 should not reintroduce full local `step()`
  orchestration.
- New binary spatial toys should implement the hook protocol rather than
  copying a toy runner.
- Toy 2/4/5 YAML, CSV, result, and summary names are intentionally not
  backwards compatible with the pre-`model/domain` shape.
- Toy-specific fields must be added under `domain_*`; common binary fields
  should stay generic across toys.
- Shared runner behavior should be changed only with fake-domain contract tests
  plus Toy-specific regression tests.
