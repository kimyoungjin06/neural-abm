# Neural ABM Node Guidelines

These guidelines define the reusable unit that the toy models should test.

## Core Principle

A Neural ABM Node is not just an MLP agent. It is an ABM agent with:

- Private observation and action/prediction.
- Internal neural state.
- Local learning.
- Social messaging.
- Peer selection.
- Social compatibility scoring.
- Optional peer-state alignment or translation.
- Typed social mixing.
- Full micro-state logging.

## Time Scales

Keep three update scales separate.

| Scale | Target | Examples |
| --- | --- | --- |
| Fast | `internal_state` | observe, infer, act, update short memory |
| Medium | `model_params`, `peer_set`, `trust_state` | local learning, social update, peer refresh |
| Slow | `social_mixer` or meta-policy | learned edge function, calibration, meta-update |

Do not mix these updates implicitly. Every experiment should state which scales
are active.

## Common Interface

Every toy model should be expressible through this interface:

```text
observe(env, history) -> observation
act_or_predict(observation, internal_state) -> output
local_update(private_data_or_reward) -> updated local model
social_message(output, internal_state, metrics) -> bounded message
compatibility_score(self_message, peer_message, context) -> scalar
peer_select(messages, graph, threshold) -> peer set and edge scores
align_or_translate(peer_state, self_state, channel) -> comparable peer state
social_mix(channel, self_state, aligned_peer_states, edge_weights) -> update
commit_social_update(update) -> updated state/model/output
log_state() -> micro-state record
```

## Revision Operator Prototype

Binary game and contribution models should not assume that the neural output is
always `P(action=1)`. Classical ABM rules often operate as revision protocols:

```text
current action + domain pressure + social evidence + inertia
-> P(stay), P(switch_to_1), P(switch_to_0)
```

Use this as a separate prototype path, not as a silent replacement for the
policy-probability learner. The reusable unit owns only the lifecycle and typed
revision choices. Domain adapters still own the meaning of revision pressure,
social evidence, basin/objective terms, and inertia/readiness state.

Current Toy2/Toy4 adapter boundary:

- `coordination.revision_operator_enabled=false` keeps the original
  policy-probability path.
- `revision_operator_source=policy_probability` maps the existing decision
  probability into stay/switch probabilities as a structural hook. It is not
  yet a learned revision network.
- Aggregate and micro logs must expose revision probabilities and realized
  choices even when the operator is disabled, so comparisons do not depend on
  schema drift.

Minimum comparison rule:

- Run revision-operator variants without auxiliary commitment/precommitment
  gates first.
- Compare against the best policy-probability variant on final ceiling,
  time-to-ceiling, flip/reversal rate, premature cascade rate, and objective
  sign agreement.
- Stop the path if the toy-specific signal callbacks become larger than the
  shared revision lifecycle or if the learned operator only reproduces a named
  reference rule without additional diagnostics.

## Social Update Pipeline

Do not treat social update as a single opaque operation. The reusable NABM
design must keep these stages separate:

```text
agent state
-> bounded social message
-> compatibility scoring
-> peer selection
-> optional alignment or translation
-> typed social mixing
-> commit update
-> log micro-state
```

This separation is not cosmetic. Toy 1 showed that independent-init parameter
averaging can fail because raw parameter similarity fragments the peer graph,
while hidden-unit aligned similarity restores connectivity over a much wider
threshold range. The averaging operator and the peer-selection rule therefore
need separate names, configs, logs, and ablations.

## Peer Selection Taxonomy

Peer selection is the social graph construction step. It should be configurable
independently from the mixer.

| Rule | Compared Object | Main Use | Main Risk |
| --- | --- | --- | --- |
| `none` | Candidate graph only | No-social or unfiltered-contact baselines. | Can hide social selectivity. |
| `state_similarity` | Raw flattened state or parameters. | Simple parameter-path diagnostics. | Sensitive to incompatible bases. |
| `aligned_state_similarity` | State after an alignment map. | Independent-init parameter diagnostics. | Alignment cost and assumptions can dominate. |
| `latent_similarity` | Compact latent summaries. | Representation-level social neighborhoods. | Latents may not be comparable across agents. |
| `output_similarity` | Probe outputs or policy outputs. | Behavior-level neighborhoods. | Probe choice can define the result. |
| Learned edge scorer | Messages plus context. | Dynamic graph or TGN-style extensions. | Opaque without strong diagnostics. |

## Alignment and Translation

Alignment is a first-class part of the social interface when agents exchange or
compare non-output internal states.

Use an explicit map:

```text
A_{j -> i}(state_j, state_i, channel) -> aligned_state_j
```

Minimum alignment variants:

| Variant | Use | Notes |
| --- | --- | --- |
| Identity | Outputs, same-init homogeneous models, scalar messages. | Must be stated explicitly. |
| Hidden-unit permutation | Single-hidden-layer MLP parameter path. | Toy 1 baseline for independent-init agents. |
| Representation projection | Heterogeneous latent channels. | Needs a validation task. |
| Learned translator | Future heterogeneous agents. | Requires logging and regularization. |

Parameter-level experiments must state whether alignment is used for:

- peer selection only
- mixing only
- both peer selection and mixing
- neither

## Social Mixing Taxonomy

Use these variants as the first comparison set:

| Variant | Description | Main Risk |
| --- | --- | --- |
| No social | Agents learn independently. | Weak baseline if data shards are easy. |
| Output averaging | Mix predicted probabilities, logits, or scalar outputs. | Needs output-type-specific averaging. |
| Latent/message mixing | Mix compact hidden states or social messages. | May become hard to interpret. |
| Parameter averaging | Average model weights. | Weight permutation and incompatible representations. |
| Parameter averaging + alignment | Align weights before averaging. | Adds complexity and may dominate the result. |
| Learned edge mixer | Learn `W_ij` from messages and context. | Can become opaque without logging. |

## Typed Mixer Channels

Every mixer should declare its channel. The same peer set can have different
meaning depending on the channel being mixed.

| Channel | Typical State | Candidate Mixer |
| --- | --- | --- |
| Output | Probabilities, logits, scalar predictions. | Pooling, logit pooling, distillation. |
| Latent | Hidden summaries, belief vectors. | Weighted representation matching. |
| Parameter | Model weights. | FedAvg-style averaging, aligned averaging. |
| Action or policy | Discrete actions, policy probabilities. | Voting, policy pooling, policy distillation. |
| Memory or trust | History summaries, peer reliability. | Exponential update, learned trust state. |

Do not compare mixer families without recording the channel. For example,
`parameter_aligned_average + aligned_state_similarity` and
`parameter_aligned_average + state_similarity` are different social mechanisms,
even though their averaging operator is the same.

## Output Handling

Define the social mean by output type.

| Output Type | Candidate Social Update |
| --- | --- |
| Scalar regression | Weighted arithmetic mean. |
| Probability distribution | Probability pooling, logit pooling, or distillation target. |
| Discrete action | Voting, stochastic policy mixing, or policy distillation. |
| Latent vector | Weighted latent/message average with normalization. |

## Communication Budget

Every social message should have a size limit. This prevents the system from
quietly becoming full model sharing.

Candidate message fields:

- Prediction on a shared probe set.
- Confidence or entropy.
- Compact latent summary.
- Recent reward or payoff.
- Trust summary.
- Optional parameter summary, not full parameters unless explicitly tested.

## Synchronization

Each experiment must specify one synchronization mode:

- Synchronous: all agents update together.
- Asynchronous: agents update in random or scheduled order.
- Event-driven: updates occur only after interaction events.

ABM dynamics can change substantially under different synchronization modes, so
the mode belongs in every config and result table.

## Logging Requirements

At minimum, each simulation step should record:

```text
run_id
seed
t
agent_id
observation_summary
output
action
local_loss_or_reward
internal_state_summary
message_summary
peer_ids
edge_weights
update_type
graph_component_id
```

For parameter experiments, also record:

```text
param_norm
param_delta_norm
alignment_enabled
alignment_metric
alignment_scope
compatibility_rule
compatibility_threshold
```

## Metrics

Separate task metrics from social dynamics metrics.

Task metrics:

- Accuracy or loss.
- Calibration.
- Robustness to noisy agents.

Social metrics:

- Consensus.
- Polarization.
- Fragmentation.
- Peer graph connected components.
- Edge entropy.
- Cooperation rate and payoff for game models.
