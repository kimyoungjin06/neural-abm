# Toy 3: Neural Opinion Rewiring

Toy 3 tests whether Neural ABM can represent opinion polarization and graph
fragmentation when interaction structure is endogenous. Unlike Toy 1 and Toy 2,
the graph can change during the run: high-disagreement edges can be dropped and
replaced by more homophilous candidates.

## Purpose

Toy 3 is a research-validation model for:

- Bounded-confidence opinion clustering.
- Echo-chamber formation through homophilous peer selection.
- Fragmentation under endogenous rewiring.
- Classical baseline comparison against a neural social variant.

The first implementation target is not a complete social-media model. It is a
controlled vertical slice where HK, Deffuant, and neural opinion updates share
the same config, runner, logging, sweep, and summary structure.

## State Space

Each agent has one scalar opinion:

```text
opinion_i in [-1, 1]
```

The graph is undirected, simple, and dynamic when rewiring is enabled. The first
implementation keeps no self-loops and no duplicate edges.

Default initial condition:

```text
agent_count = 100
initial_opinion_mode = two_clusters
cluster_centers = [-0.4, 0.4]
cluster_std = 0.08
graph = Watts-Strogatz(n=100, k=6, p=0.1)
```

## Config Schema

Toy 3 uses these top-level blocks:

```yaml
run:
simulation:
model:
domain:
logging:
```

Core defaults:

```yaml
model:
  policy:
    update_rule: hk
    confidence_threshold: 0.35
    influence_rate: 1.0
    deffuant_mu: 0.5
    neural_delta_scale: 0.25
    neural_learning_rate: 0.01
  agents:
    count: 100
  coordination:
    mixer: none
    peer_rule: bounded_confidence
    alpha: 0.0

domain:
  toy: toy3
  environment:
    opinion_min: -1.0
    opinion_max: 1.0
  graph:
    type: watts_strogatz
    k: 6
    rewire_probability: 0.1
  rewiring:
    enabled: false
    threshold: 0.8
    rate: 0.0
    candidate_pool_size: 10
```

The baseline config is:

```text
experiments/configs/toy3_opinion_rewiring_baseline.yaml
```

Run it with:

```bash
scripts/run_toy3.py --config experiments/configs/toy3_opinion_rewiring_baseline.yaml
```

## Baselines

### HK Bounded Confidence

Each agent averages with graph neighbors whose opinion distance is within
`confidence_threshold`. The local target includes the agent's own current
opinion and compatible neighbor opinions.

### Deffuant Pairwise Update

Each epoch shuffles graph edges and applies the symmetric Deffuant update to
compatible pairs:

```text
o_i <- o_i + mu * (o_j - o_i)
o_j <- o_j + mu * (o_i - o_j)
```

When the pair is compatible, the pair mean is conserved.

### Neural Policy

The neural variant interprets the model output as an acceptance probability for
peer influence.

Default network:

```text
MLP: 6 -> 16 -> 1
output: sigmoid acceptance probability
```

Observation vector:

```text
own opinion
neighbor opinion mean
neighbor opinion standard deviation
mean local disagreement
recent opinion drift
bias
```

For compatible peers, the opinion moves toward compatible peer mean according
to:

```text
delta = influence_rate * acceptance_probability * (compatible_peer_mean - own_opinion)
```

The delta is bounded by `neural_delta_scale` before opinion clipping.

## Social Mixer

The first social mixer is `output_average` over neural acceptance outputs:

```text
p_accept_i <- (1 - alpha) * p_accept_i + alpha * mean(peer p_accept)
```

Peers are bounded-confidence neighbors. Classical HK and Deffuant paths log the
same social fields but do not use neural output mixing.

## Rewiring

When rewiring is enabled:

- Edges with opinion disagreement above `rewiring.threshold` are candidates.
- Each candidate edge rewires with probability `rewiring.rate`.
- The dropped edge is replaced by a non-neighbor candidate chosen from a sampled
  candidate pool by closest opinion distance.
- The graph remains simple, undirected, and loop-free.

This is intentionally local to Toy 3 until Toy 4 or Toy 5 need dynamic graphs.

## Metrics

Aggregate metrics:

- `domain_opinion_mean`.
- `domain_opinion_variance`.
- `domain_polarization_index`, normalized by the maximum possible variance in the
  opinion bounds.
- `domain_opinion_cluster_count` from sorted opinion gaps.
- `domain_mean_edge_disagreement`.
- `domain_high_disagreement_edge_fraction`.
- `fragmentation_components`.
- `domain_largest_connected_component_fraction`.
- Mean compatible peer count.
- `domain_rewired_edge_count` and `domain_cumulative_rewired_edge_count`.
- `domain_realized_rewiring_rate`, computed as rewired edges divided by the edge count
  considered at the start of that rewiring pass.
- `domain_opinion_assortativity` approximation.

Micro-state fields:

- `domain_opinion` and `domain_opinion_pre_update`.
- `domain_opinion_delta`.
- `domain_neighbor_opinion_mean` and `domain_neighbor_opinion_std`.
- `domain_local_disagreement`.
- `domain_degree`.
- Compatible peer IDs and peer count.
- Component ID in the compatible-peer graph.
- `domain_edge_disagreement`.
- `domain_acceptance_probability_pre_social` and
  `domain_acceptance_probability_post_social`.
- Revised flag.
- `domain_rewired` flag.

The summary JSON exposes common fields such as `run_dir`, `toy`, and
`final_fragmentation_components`; Toy-specific final metrics live under
`domain_metrics` with `domain_*` keys.

## Sweep

The sweep entry point is:

```bash
scripts/run_toy3_sweep.py
```

Default axes:

- `update_rule`: `hk`, `deffuant`, `neural_policy`
- `confidence_threshold`: `0.25`, `0.35`, `0.5`
- `rewiring_rate`: `0.0`, `0.25`
- `mixer`: `none`, `output_average`
- `seed`: `1`, `2`, `3`, `4`, `5`

Summary CSVs include polarization, opinion cluster count, rewiring rate, and
mean edge disagreement.

## First Success Gate

Toy 3 is ready for research ablations when:

- The no-social HK baseline forms interpretable opinion clusters under bounded
  confidence.
- Rewiring increases homophily or reduces mean edge disagreement.
- The neural/social variant changes polarization or fragmentation relative to
  the classical baselines.
- The runner writes `aggregate_metrics.csv`, `micro_state.csv`, `summary.json`,
  `metadata.json`, `config.yaml`, and `resolved_config.yaml`.
