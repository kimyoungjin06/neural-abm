# Model Family Roadmap

The model families are research-validation axes, not just examples. Each family
keeps a classical baseline next to the neural/social variant so the outcome is
interpretable.

Current scope uses feature names for conceptual organization and stable IDs for
reproducibility:

- Neural HK Classification (`toy1`).
- Spatial Prisoner's Dilemma (`toy2`).
- Opinion Rewiring (`toy3`).
- Public Goods Commons (`toy4`).
- Contagion Adoption (`toy5`).
- Categorical Spatial Game (`toy6`).
- Resource Intensity (`toy7`).
- Async Event ABM (`toy8`).
- Heterogeneous Agent Rules (`toy9`).
- Market Ecology Network (`toy10`).

Capability matrix: [capability-matrix.md](capability-matrix.md)

## Public API Contract

All checked-in model-family YAML configs use the same top-level shape:

```text
run
simulation
model
domain
logging
```

`model` contains policy/agent/coordination state for the toy, while `domain`
contains domain-specific environment, game, graph, data, or rewiring settings
and always includes a stable model ID such as `toy1`.

Runner outputs also share a public shape. Summary JSON files include `run_dir`,
`toy`, common `final_*` fields, and `domain_metrics`. Toy-specific CSV and
summary metrics use `domain_*` names. The public contract is guarded by
`tests/test_toy_public_api_contract.py`, and deterministic tiny-run behavior is
guarded by `tests/test_toy_golden_regression.py`.

## Toy 1: Neural HK Classification

Detailed spec: [neural-hk-classification.md](neural-hk-classification.md)

Purpose:

- Provide a clean benchmark where task accuracy is measurable.
- Test whether social mixing helps agents with biased private data shards.
- Compare state-similarity and output-similarity peer selection.

Environment:

```text
x in R^2
y in {0, 1}
y = nonlinear boundary with optional label noise
```

Agent data shards:

- Region-biased shards.
- Boundary-heavy shards.
- Noisy-label shards.
- Small balanced shards.

Agent model:

```text
MLP: 2 -> 16 -> 2
output: class probability
latent: hidden activation summary or agent embedding
```

Social message:

```text
prediction on shared probe set
confidence or entropy
latent summary
optional parameter summary
```

Key comparisons:

- No social.
- HK-style output averaging.
- Latent/message mixing.
- Parameter averaging.
- Parameter averaging with alignment.
- Learned edge mixer.

Primary plots:

- Accuracy over epoch.
- Consensus versus accuracy.
- Peer-threshold phase diagram.
- Fragmentation under state-peer versus output-peer selection.
- Robustness as noisy-agent ratio increases.

## Toy 2: Neural Spatial Prisoner's Dilemma

Detailed spec: [neural-spatial-pd.md](neural-spatial-pd.md)

Purpose:

- Test whether the same social mixer concepts affect game dynamics.
- Study cooperation, defection, clustering, echo chambers, and exploitation.

Payoff table:

| | Opponent C | Opponent D |
| --- | --- | --- |
| Agent C | `R = 3` | `S = 0` |
| Agent D | `T = 5` | `P = 1` |

Agent observation:

```text
own previous action
neighbor action histogram
recent payoff mean
recent payoff trend
domain_neighbor_action_rate
trust state
social message summary
```

Agent model:

```text
policy MLP: observation -> p(action)
```

Social message:

```text
p(action)
recent payoff
trust summary
latent policy embedding
```

Key comparisons:

- No social.
- Strategy imitation baseline.
- Output policy mixing.
- Latent/message mixing.
- Learned edge mixer.

Primary plots:

- Action rate over time.
- Average payoff over time.
- Domain action cluster size.
- Defector invasion resistance.
- Trust segregation or edge rewiring rate.

## Toy 3: Neural Opinion Rewiring

Detailed spec: [neural-opinion-rewiring.md](neural-opinion-rewiring.md)

Purpose:

- Test bounded-confidence opinion clustering.
- Compare HK, Deffuant, and neural acceptance dynamics.
- Introduce dynamic graph rewiring without generalizing it prematurely.
- Track polarization, edge disagreement, and fragmentation.

State:

```text
opinion_i in [-1, 1]
```

Default graph:

```text
Watts-Strogatz, n = 100, k = 6, p = 0.1
```

Key comparisons:

- HK bounded-confidence baseline.
- Deffuant pairwise-update baseline.
- Neural acceptance policy with optional output averaging.
- Rewiring disabled versus homophilous rewiring.

Primary plots:

- Opinion variance and polarization over time.
- Opinion cluster count over time.
- Mean edge disagreement over time.
- Connected components and largest component fraction.
- Rewiring rate and cumulative rewired edges.

## Toy 4: Neural Public Goods and Commons

Detailed spec: [neural-public-goods-commons.md](neural-public-goods-commons.md)

Purpose:

- Test group-level externalities instead of pairwise payoff.
- Compare public-goods imitation baselines with neural contribution policies.
- Add an optional commons stock to study collapse and recovery.

State:

```text
action_i in {free_ride, contribute}
optional resource_t in [0, carrying_capacity]
```

Primary metrics:

- Action rate.
- Mean payoff and payoff inequality.
- Resource level and collapse time.
- Domain action clustering.
- Exploitation index.

Implementation status: binary-contribution vertical slice is implemented with
`neural_policy` and `imitation` update rules, optional output averaging, and an
optional commons resource stock.

## Toy 5: Neural Contagion and Adoption

Detailed spec: [neural-contagion-adoption.md](neural-contagion-adoption.md)

Purpose:

- Test simple contagion, complex contagion, and heterogeneous thresholds.
- Compare classical threshold diffusion with neural adoption policies.
- Track cascade success, partial cascades, and exposure-response behavior.

State:

```text
adopted_i in {0, 1}
optional threshold_i in [0, 1]
```

Primary metrics:

- Action rate over time.
- Cascade size.
- Time to 50 percent adoption.
- Failed cascade count.
- Domain action cluster count.
- Exposure-response curve by threshold group.

Implementation status: discrete-adoption vertical slice is implemented with
`simple_contagion`, `complex_threshold`, and `neural_policy` update rules,
heterogeneous thresholds, optional output averaging, and cascade diagnostics.

## Implementation Order

1. Implement common logging and config schema.
2. Implement Toy 1 with fixed graph and synchronous updates.
3. Add output and latent/message social mixing.
4. Add parameter averaging and alignment ablation.
5. Add learned edge mixer.
6. Port the same node/mixer interface to Toy 2.
7. Add Toy 3 opinion rewiring as the first dynamic-graph vertical slice.
8. Add Toy 3 sweep and diagnostics.
9. Specify Toy 4 and Toy 5 in docs before implementation.
10. Implement Toy 4 binary public-goods contribution.
11. Implement Toy 5 discrete adoption diffusion.
12. Add asynchronous and shared dynamic-graph experiments only after they are
    justified by Toy 3 through Toy 5.
