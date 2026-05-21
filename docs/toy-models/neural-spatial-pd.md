# Toy 2: Neural Spatial Prisoner's Dilemma

Toy 2 moves the project from supervised correctness to game dynamics. The goal
is not to build a rich social simulator yet. The goal is to verify that the
NABM social pipeline from Toy 1 also works when outputs are policies, rewards
come from interaction, and the main metrics are cooperation, payoff, and spatial
clustering.

## Purpose

Test whether neural agents with local policy learning and explicit social
policy mixing produce different game dynamics from independent local learning.

The first Toy 2 implementation should answer:

- Can the same node/logging/config pattern run a non-supervised ABM?
- Does output-level policy mixing change cooperation rate or payoff?
- Do peer components and cooperation clusters provide useful social-dynamics
  metrics?
- What additional state is needed before adding trust or learned edge functions?

## Environment

Use a 2D toroidal grid so every agent has a stable local neighborhood.

Default topology:

```text
grid_width = 10
grid_height = 10
neighborhood = von_neumann
periodic = true
agent_count = grid_width * grid_height
```

Actions:

```text
0 = defect
1 = cooperate
```

Default payoff table:

| | Opponent C | Opponent D |
| --- | ---: | ---: |
| Agent C | `R = 3` | `S = 0` |
| Agent D | `T = 5` | `P = 1` |

Each round, every connected pair plays once. Agent payoff is the mean payoff
over its neighbors.

## Agent Observation

The first observation vector is intentionally small:

```text
own_previous_action
domain_neighbor_action_rate
own_payoff_ema / T
payoff_trend / T
neighbor_mean_payoff / T
bias = 1
```

This gives the policy local social context without adding memory or trust yet.

## Agent Model

Default policy network:

```text
MLP: 6 -> 16 -> 2
output: softmax over [defect, cooperate]
optimizer: Adam
```

Local learning uses a simple policy-gradient style update:

```text
loss = -advantage * log_prob(action) - entropy_beta * entropy(policy)
```

The first version uses an EMA payoff baseline for the advantage. This is a
minimal behavioral learner, not a claim about optimal reinforcement learning.

## Tensor-Batched Torch State

Toy 2 supports `policy.neural_update_backend: tensor_batched` for
`policy.rule: neural_policy`. In this path the core binary state stays on the
configured torch device:

```text
actions: torch.long
payoffs: torch.float64
payoff_ema: torch.float64
previous_payoff_ema: torch.float64
reputation: torch.float64
```

The public config and result contract are unchanged. CSV rows, summaries, and
micro-state fields are still emitted as scalar/NumPy-compatible values.

Torch-state coverage includes:

- sampled and counterfactual local neural updates
- `output_average` social distillation
- `output_similarity` peer-selection smoke coverage
- reputation observation features
- reputation EMA updates
- mobility state swaps

Fallbacks remain intentionally conservative:

- `loop` and `batched` keep the NumPy state path.
- Non-neural Toy2 policies keep the NumPy state path.
- Dynamic ragged peer contexts can still use list peer ids while converting
  state tensors only at the computation boundary.
- Mobility currently uses a NumPy temporary mirror and copies swapped values
  back into torch tensors.

Benchmark artifact:

- `experiments/results/toy2_torch_state_fast_path_consensus_opt_1024_2048_e5_r5_analysis.md`
- `experiments/results/toy2_torch_state_fast_path_finalize_initial_opt_1024_2048_e5_r5_analysis.md`
- `experiments/results/toy2_torch_state_fast_path_social_static_opt_1024_2048_e5_r5_analysis.md`
- `experiments/results/toy2_torch_state_fast_path_direct_init_opt_1024_2048_e5_r5_analysis.md`
- `experiments/results/toy2_torch_state_fast_path_aggregate_cluster_opt_1024_2048_e5_r5_analysis.md`
- `experiments/results/toy2_torch_state_fast_path_generator_init_opt_1024_2048_e5_r5_analysis.md`
- `experiments/results/toy2_torch_state_fast_path_trainable_view_opt_1024_2048_e5_r5_analysis.md`
- `experiments/results/toy2_torch_state_fast_path_active_ids_opt_1024_2048_e5_r5_analysis.md`
- `experiments/results/toy2_torch_state_fast_path_binary_loss_opt_1024_2048_e5_r5_analysis.md`
- `experiments/results/toy2_torch_state_fast_path_kl_loss_opt_1024_2048_e5_r5_analysis.md`

## NABM Unit Decomposition

Toy 2 now treats the neural agent unit as four explicit channels rather than a
single policy-sampling path:

```text
observation builder
-> policy readout
-> decision/action kernel
-> local learning kernel
-> coordination policy mixer
```

The action path is:

```text
observation -> policy head -> decision kernel -> realized action
```

`policy.temperature` belongs to policy readout. It controls the probability
distribution used for logging, peer selection, and coordination policy mixing.

`policy.decision` belongs to realized action selection:

```yaml
model:
  policy:
    rule: neural_policy
    temperature: 1.0
    decision:
      mode: sampled
      action_temperature: 1.0
      exploration_epsilon: 0.0
    domain:
      local_update_rule: counterfactual_advantage
  coordination:
    mixer: none
    peer_rule: none
    alpha: 0.0
    threshold: 0.0
  state:
    reputation:
      enabled: true
    mobility:
      enabled: false
domain:
  toy: toy2
  environment:
    initial_action_probability: 0.5
  game:
    family: prisoner_dilemma
```

`decision.mode=sampled` samples revised actions from the readout probabilities.
`decision.action_temperature` sharpens or softens that sampled action channel
without changing the logged policy readout. `decision.mode=argmax` selects the
maximum-probability action and ignores `action_temperature` and
`exploration_epsilon`.

The default action kernel remains stochastic
(`mode=sampled`, `action_temperature=1.0`). RD-like Stag-Hunt diagnostics use
lower action temperatures, especially around `0.5`, as a calibration probe
rather than as the global default.

## Social Pipeline

Toy 2 reuses the decomposed NABM social pipeline:

```text
social_message
-> compatibility_score
-> peer_select
-> align_or_translate
-> typed social_mix
-> commit_social_update
```

The first implementation uses identity alignment because the exchanged channel
is output policy probability, not model weights.

## Peer Selection

Candidate peers are spatial graph neighbors.

Initial peer rules:

| Rule | Meaning |
| --- | --- |
| `none` | Use all spatial neighbors. |
| `output_similarity` | Compare current `p(cooperate)` values. |

## Mixer Variants

Initial variants:

| Mixer | Channel | Description |
| --- | --- | --- |
| `none` | none | Independent local policy learning. |
| `output_average` | policy output | Mix peer cooperation probabilities and distill the local policy toward the mixed target. |

Deferred variants:

| Variant | Reason Deferred |
| --- | --- |
| Strategy imitation | Needs a clean baseline definition separate from neural policy mixing. |
| Latent/message mixing | Requires a stronger interpretation of policy embeddings. |
| Learned edge mixer | Should wait until the simple policy-output path is interpretable. |
| Trust state | Requires bad-peer or exploitation tests. |

## Metrics

Aggregate metrics use generic common binary names plus Toy-2-specific
`domain_*` fields:

- `action_rate`.
- Mean payoff.
- `mean_policy_action_probability`.
- Policy consensus.
- `domain_action_components`.
- `domain_largest_action_cluster_fraction`.
- Peer graph fragmentation.
- Mean peer count.
- Edge entropy.

Micro-state fields:

- `action`.
- `action_probability`: post-social policy action probability.
- `policy_action_probability_pre_revision`: policy readout before action
  revision and local learning.
- `policy_action_probability_post_local`: policy readout after local
  learning and before social mixing.
- `policy_action_probability_post_social`: policy readout after social
  mixing.
- `candidate_decision_action_probability_pre_revision`: decision-kernel action
  probability before revision masking.
- `realized_decision_action_probability`: decision-kernel action probability
  only for agents that revised in that epoch; blank means no decision event
  occurred.
- Payoff.
- Payoff EMA.
- `domain_neighbor_action_rate`.
- Neighbor mean payoff.
- Peer IDs.
- Peer count.
- Peer component ID.
- Local policy loss.
- Social distillation loss.

## First Success Gate

Toy 2 is ready for ablations if:

- A config-driven run writes aggregate and micro-state logs.
- `none` and `output_average` complete through the same runner.
- Cooperation and payoff metrics remain in valid ranges.
- Peer graph metrics are logged every epoch.
- The run is deterministic under a fixed seed except for intended policy
  sampling controlled by that seed.
