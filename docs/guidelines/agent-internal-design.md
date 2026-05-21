# Agent Internal Design

This document records the internal structure expected from a reusable
Neural ABM Agent before the current Toy 1 implementation is generalized.

## Purpose

The current Toy 1 agent is intentionally small: an MLP classifier with a private
data shard and local optimizer. The long-term agent should remain compatible
with this simple case while adding memory, trust, message budgets, and explicit
update clocks only when experiments justify them.

## Target Agent Structure

```text
NeuralABMAgent
  identity:
    agent_id
    group
    role

  model:
    predictor or policy network

  internal_state:
    latent belief
    confidence
    optional trust state
    optional memory summary

  optimizer:
    local learning mechanism

  memory:
    recent observations
    recent outputs/actions
    recent rewards/losses
    recent social messages

  social_interface:
    emit_message()
    receive_messages()
    score_compatibility()
    select_peers()
    align_peer_state()
    mix_channel()
    commit_social_update()

  update_clock:
    fast: observe -> predict/act
    medium: local_update -> social_pipeline
    slow: trust/meta/mixer update

  diagnostics:
    log_state()
    summarize_state()
```

## Minimal Contract

Every concrete agent should expose these operations:

```text
observe(input_or_environment) -> observation
act_or_predict(observation) -> output
local_update(batch_or_reward) -> scalar loss/reward metric
social_message(context) -> bounded message dict
compatibility_score(peer_message, context) -> scalar
peer_select(candidate_messages, graph, threshold) -> peer ids and scores
align_or_translate(peer_state, channel) -> comparable peer state
social_mix(channel, aligned_peer_states, peer_weights) -> update summary
commit_social_update(update_summary) -> committed state change
log_state(context) -> flat dict
```

Toy-specific agents may keep these methods thin, but the method names and
expected semantics should remain stable across toy models.

## Current Toy 1 Mapping

| Contract | Toy 1 Implementation |
| --- | --- |
| `observe` | Receives a tensor of feature points. |
| `act_or_predict` | Produces class probabilities. |
| `local_update` | Runs supervised MLP updates on private shard batches. |
| `social_message` | Emits probe probabilities, latent summary, confidence, and parameter norm. |
| `compatibility_score` | Implemented by external peer-rule functions. |
| `peer_select` | Implemented by graph-neighbor filtering and similarity thresholds. |
| `align_or_translate` | Implemented for Toy 1 hidden-unit parameter alignment. |
| `social_mix` | Implemented by external mixer functions. |
| `commit_social_update` | Implemented by loading the mixed model state or applying a distillation step. |
| `log_state` | Emits a flat dict compatible with `micro_state.csv`. |

The social pipeline is still external because Toy 1 compares peer rules,
aligners, and mixer families independently. Later refactors may move dispatch
behind an agent-facing adapter, but peer selection, alignment, and mixer
implementations should remain separately testable.

## Test Ladder

### 1. Single-Agent Learning Test

Run one agent without social update.

Required checks:

- Local loss decreases over several updates or the final loss is finite and
  below a random-initialization sanity threshold.
- Accuracy is in `[0, 1]`.
- Fixed seed produces reproducible outputs.
- `log_state` contains required fields.

### 2. Agent Contract Test

Check the interface directly:

- `observe` preserves expected shape.
- `act_or_predict` returns valid probabilities or actions.
- `local_update` changes model parameters when learning is enabled.
- `social_message` stays within the configured message budget.
- `peer_select` returns stable peer ids under fixed seeds and thresholds.
- `align_or_translate` preserves behavior for known equivalent states when
  equivalence is expected.
- `social_mix` changes only the declared channel.
- `log_state` returns scalar or JSON-serializable fields.

### 3. Internal State Ablation

Add internal components one at a time:

```text
model only
model + latent summary
model + memory
model + confidence
model + trust
model + compressed social message
```

Only keep a component if it changes task performance, robustness, or social
dynamics in a measurable way.

### 4. Social Robustness Test

Introduce adversarial or low-quality peers:

- noisy-label agents
- low-accuracy agents
- overconfident wrong agents
- random-message agents

Check whether trust, peer weighting, or learned mixers reduce bad-peer
influence.

### 5. Multi-Agent Integration Test

After the single-agent and contract tests pass, run the full Toy 1 ablation:

```text
none
output_average
latent_average
parameter_average
parameter_aligned_average
aligned_state_similarity + parameter_aligned_average
trust_weighted_output
learned_edge_mixer
```

The full test should evaluate accuracy, consensus, polarization, fragmentation,
peer count, edge entropy, bad-peer sensitivity, and state drift.

## Adoption Rule

Do not add memory, trust, or learned mixer state to the core agent simply because
they are plausible. Add them only after a small test demonstrates what they
change and how the logs will expose the change.
