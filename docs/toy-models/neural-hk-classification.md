# Toy 1: Neural HK Classification

This toy model is the first executable test for the Neural ABM Node idea. It
keeps the environment supervised so the effect of social mixing can be measured
against a known target.

## Purpose

Test whether neural agents with biased private data can improve or reorganize
their predictions through explicit social mixing.

The first experiment should answer:

- Does social mixing change global test accuracy?
- Does peer selection by state similarity behave differently from peer selection
  by output similarity?
- Does social mixing increase consensus at the cost of diversity?
- Does parameter averaging fail under independent initialization more often than
  output or latent/message mixing?
- Does hidden-unit alignment change parameter-based peer selection and
  fragmentation?

## Scope

In scope for the first implementation:

- Binary classification in `R^2`.
- Small homogeneous MLP agents.
- Biased private data shards.
- Static graph.
- Synchronous updates.
- No-social, output, latent/message, and parameter averaging variants.
- Hidden-unit aligned parameter diagnostics after the first baseline.

Out of scope for the first implementation:

- Learned edge mixer.
- Asynchronous updates.
- Dynamic graph rewiring.
- Toy 2 game dynamics.

These are later ablations after the first logs and result summary exist.

## Environment

Each point is sampled from:

```text
x = (x1, x2), x1 ~ Uniform(-1, 1), x2 ~ Uniform(-1, 1)
```

The default nonlinear decision boundary is:

```text
y = 1[x2 > 0.35 * sin(3 * pi * x1)]
```

Label noise is optional:

```text
with probability label_noise: y = 1 - y
```

Default datasets:

| Split | Size | Purpose |
| --- | ---: | --- |
| `global_train_pool` | 20,000 | Source pool for private shards. |
| `global_probe` | 512 | Shared probe set for similarities and metrics. |
| `global_test` | 5,000 | Held-out global evaluation. |

The probe set is not used for gradient training. It is used for output
comparison, consensus, and logging.

## Agents

Default population:

```text
agent_count = 50
```

Default model:

```text
MLP: 2 -> 16 -> 2
activation: ReLU
output: softmax class probability
optimizer: Adam
learning_rate: 0.01
local_batch_size: 64
local_steps_per_epoch: 1
```

Each agent owns a private shard of the global training pool.

Default shard groups:

| Group | Count | Data Bias |
| --- | ---: | --- |
| `left_region` | 10 | Mostly `x1 < -0.2`. |
| `right_region` | 10 | Mostly `x1 > 0.2`. |
| `boundary_region` | 10 | Mostly points close to the decision boundary. |
| `noisy_labels` | 10 | Balanced region, higher label noise. |
| `small_balanced` | 10 | Balanced but smaller shard. |

Default shard sizes:

| Group | Samples per Agent |
| --- | ---: |
| `left_region` | 600 |
| `right_region` | 600 |
| `boundary_region` | 600 |
| `noisy_labels` | 600 |
| `small_balanced` | 150 |

The first run should use the same architecture for all agents. Initialization
mode is a separate ablation:

```text
same_init: all agents start from the same initial weights
independent_init: each agent starts from a different random initialization
```

## Graph

The first graph should be static to keep interpretation simple.

Default graph:

```text
type: watts_strogatz
n: 50
k: 6
rewire_probability: 0.1
```

The graph defines candidate social contacts. Peer selection can further filter
neighbors by similarity threshold.

## Epoch Order

Use synchronous updates.

For each epoch `t`:

1. Each agent trains locally for `local_steps_per_epoch`.
2. Each agent predicts on `global_probe`.
3. Each agent builds a bounded social message.
4. Compatibility scores are computed for graph neighbors.
5. Similarity thresholding selects the peer set.
6. Optional alignment translates peer state into the receiver basis.
7. A typed mixer computes the social update from the previous synchronized
   state.
8. All agents commit social updates together.
9. Metrics and micro-state logs are written.

The no-social baseline runs steps 1, 2, and 8 only.

## Social Message

Default message fields:

```text
probe_logits_or_probs
probe_entropy_mean
latent_summary
param_norm
local_loss
```

Default message budget:

```text
probe_predictions: 512 x 2 probabilities
latent_summary: 16 floats
scalar_summary: <= 8 floats
```

For the first run this is intentionally permissive. Later experiments should
reduce the probe set or compress the message.

## Peer Selection

Candidate peers are graph neighbors. Similarity is computed only among candidate
peers.

Peer rules:

```text
none:
  use all graph neighbors

state_similarity:
  sim(i, j) = cosine(flatten(theta_i), flatten(theta_j))

latent_similarity:
  sim(i, j) = cosine(latent_summary_i, latent_summary_j)

output_similarity:
  sim(i, j) = 1 - JS(mean_probe_distribution_i, mean_probe_distribution_j)

aligned_state_similarity:
  sim(i, j) = cosine(flatten(A_{j -> i}(theta_j)), flatten(theta_i))
```

Default threshold:

```text
threshold = 0.8
```

If no peer passes the threshold, the agent performs no social update for that
epoch.

## Mixer Variants

First comparison set:

| Mixer | Description |
| --- | --- |
| `none` | No social update. |
| `output_average` | Mix probe-level predicted probabilities, then distill locally. |
| `latent_average` | Mix latent summaries and use them as a state/message update. |
| `parameter_average` | Direct weight averaging with peers. |
| `parameter_aligned_average` | Align hidden units before averaging peer weights. |

Deferred variants:

| Mixer | Reason Deferred |
| --- | --- |
| `learned_edge_mixer` | Needs stable logs and simple baselines first. |

Default social strength:

```text
alpha = 0.25
```

For direct state updates:

```text
new_state_i = (1 - alpha) * state_i + alpha * peer_mean_i
```

For output averaging, use the mixed peer prediction as a distillation target on
the shared probe set:

```text
target_i = (1 - alpha) * p_i(probe) + alpha * mean_j p_j(probe)
loss = KL(target_i || p_i(probe))
```

This keeps the agent model executable after social output mixing.

## Metrics

Task metrics:

- Mean global test accuracy.
- Mean global test cross entropy.
- Per-shard-group global accuracy.
- Probe accuracy.

Social metrics:

- Consensus: mean pairwise agreement on probe predictions.
- Output divergence: mean pairwise Jensen-Shannon divergence on probe outputs.
- Polarization: number of prediction clusters on probe outputs.
- Fragmentation: number of connected components after peer filtering.
- Edge entropy: entropy of normalized peer influence weights.

Parameter metrics:

- Mean parameter norm.
- Mean parameter delta norm after social update.
- Pairwise parameter cosine similarity.

## Required Plots

Minimum first result summary:

1. Accuracy over epoch by mixer.
2. Consensus over epoch by mixer.
3. Accuracy versus consensus scatter by mixer.
4. Fragmentation over epoch for state-similarity and output-similarity peer
   rules.
5. Same-init versus independent-init comparison for parameter averaging.

## Config Requirements

The runner accepts the same public top-level sections as the other toys:
`run`, `simulation`, `model`, `domain`, and `logging`.

```yaml
run:
  name: toy1_baseline
  seed: 1
  output_dir: experiments/runs

simulation:
  epochs: 50
  sync_mode: synchronous
  device: cpu

model:
  agents:
    count: 50
    init_mode: same_init
    model:
      hidden_dim: 16
      activation: relu
    optimizer:
      name: adam
      learning_rate: 0.01
    shards:
      policy: five_group_bias
  coordination:
    mixer: output_average
    peer_rule: output_similarity
    alpha: 0.25
    threshold: 0.8
    communication_budget:
      probe_predictions: 512
      latent_dim: 16

domain:
  toy: toy1
  data:
    boundary: sine
    label_noise: 0.02
    train_pool_size: 20000
    probe_size: 512
    test_size: 5000
  graph:
    type: watts_strogatz
    k: 6
    rewire_probability: 0.1

logging:
  micro_state: true
  interval: 1
```

## Logging Schema

Write one agent row per epoch at minimum.

Required fields:

```text
run_id
seed
epoch
agent_id
domain_shard_group
coordination_mixer
coordination_peer_rule
model_init_mode
local_loss
domain_global_accuracy
domain_probe_accuracy
domain_probe_entropy
domain_confidence
peer_ids
edge_weights
peer_count
component_id
message_norm
latent_norm
param_norm
param_delta_norm
domain_output_js_to_population_mean
```

Write aggregate metrics separately per epoch:

```text
run_id
seed
epoch
coordination_mixer
coordination_peer_rule
model_init_mode
domain_mean_global_accuracy
domain_mean_probe_accuracy
domain_mean_consensus
domain_mean_output_js
domain_polarization_clusters
fragmentation_components
mean_peer_count
edge_entropy
```

The summary JSON exposes common fields such as `run_dir`, `toy`, and
`final_fragmentation_components`; Toy-specific final metrics live under
`domain_metrics` with `domain_*` keys, for example
`domain_final_mean_global_accuracy` and `domain_final_mean_consensus`.

## First Ablation Matrix

Keep the first matrix intentionally small.

| Axis | Values |
| --- | --- |
| `mixer` | `none`, `output_average`, `latent_average`, `parameter_average` |
| `peer_rule` | `none`, `state_similarity`, `output_similarity` |
| `init_mode` | `same_init`, `independent_init` |
| `seed` | `1, 2, 3, 4, 5` |

Not every combination is required. Minimum first batch:

```text
none + none + same_init
output_average + output_similarity + same_init
latent_average + state_similarity + same_init
parameter_average + state_similarity + same_init
parameter_average + state_similarity + independent_init
```

Follow-up parameter diagnostic:

```text
parameter_average + state_similarity + independent_init
parameter_aligned_average + state_similarity + independent_init
parameter_aligned_average + aligned_state_similarity + independent_init
```

This separates three effects:

- raw parameter averaging
- aligned parameter averaging with the same raw peer graph
- aligned peer selection plus aligned averaging

## Success Gate

Toy 1 is successful enough to proceed to Toy 2 if:

- Runs are reproducible from config and seed.
- Micro-state logs can reproduce aggregate metrics.
- At least one social mixer measurably changes accuracy, consensus,
  polarization, or fragmentation relative to no-social.
- Parameter averaging differs between same-init and independent-init, or the
  absence of a difference is documented as a result.
- Aligned and unaligned parameter paths are compared before making claims about
  parameter-level social learning.
- State-similarity and output-similarity peer rules produce distinguishable peer
  graph dynamics.
