# Toy 4: Neural Public Goods and Commons

Toy 4 tests Neural ABM under group-level externalities. The point is to
move beyond pairwise payoff interactions and check whether social mixing affects
cooperation, exploitation, and resource collapse in a public-goods setting.

The first implementation is a binary-contribution vertical slice with a
classical imitation baseline, a neural policy variant, optional output mixing,
and an optional commons resource stock.

## Purpose

Toy 4 should answer:

- Does the framework handle group payoff coupling rather than pairwise payoff?
- Do social outputs change contribution rate or collapse time?
- Can the logs distinguish cooperation, payoff, inequality, and exploitation?
- Does the neural/social variant remain interpretable against classical
  public-goods baselines?

## State and Action Space

Agent action:

```text
action_i in {0, 1}
0 = free ride
1 = contribute
```

Optional commons state for the resource variant:

```text
resource_t in [0, carrying_capacity]
```

The first implementation should use binary contribution only. Continuous
contribution is deferred until the binary version is stable.

## Group Structure

The default group is each agent's local neighborhood on a graph or grid. For a
focal group:

```text
group_contribution = sum(action_j for j in group)
public_return = multiplier * group_contribution
share = public_return / group_size
payoff_i = share - contribution_cost * action_i
```

The same agent can participate in multiple local pools if the implementation
uses overlapping neighborhood games. The config must make this explicit.

## Config Schema

Toy 4 uses these top-level blocks:

```yaml
run:
simulation:
model:
  policy:
  agents:
  coordination:
  state:
domain:
  toy: toy4
  environment:
  game:
  graph:
logging:
```

Default values for v1:

```yaml
model:
  policy:
    rule: neural_policy
    learning_enabled: true
    revision_rate: 1.0
    selection_strength: 1.0
    temperature: 1.0
    decision:
      mode: sampled
      action_temperature: 1.0
      exploration_epsilon: 0.0
    domain: {}
  agents:
    init_mode: independent_init
    model:
      input_dim: 6
      hidden_dim: 16
      output_dim: 2
      activation: relu
  coordination:
    mixer: none
    peer_rule: none
    alpha: 0.0
    threshold: 0.0
  state:
    reputation:
      enabled: true
      decay: 0.9
      peer_rule: spatial
      temperature: 1.0
      noise: 0.0
      observation_mode: none
    mobility:
      enabled: false
      rate: 0.0
      candidate_pool_size: 8
      selection_rule: local_quality
      move_cost: 0.0

domain:
  toy: toy4
  environment:
    grid_width: 10
    grid_height: 10
    initial_action_probability: 0.5
    reward_ema_decay: 0.90
    entropy_beta: 0.01
    resource_enabled: false
    resource_initial: 100.0
    resource_carrying_capacity: 100.0
    resource_recovery_rate: 0.05
    resource_extraction_per_defector: 1.0
    resource_collapse_threshold: 0.0
  game:
    multiplier: 1.6
    contribution_cost: 1.0
    group_mode: local_neighborhood
  graph:
    type: grid
    neighborhood: von_neumann
    periodic: true
```

Toy-specific CSV and summary fields use `domain_*` names, for example
`domain_resource_level`, `domain_payoff_gini`, and `domain_collapse_time`.
For the imitation baseline, `selection_strength` is the inverse-temperature-like
scale in a Fermi copy probability over payoff differences. `0.0` disables
payoff-biased copying; larger values make copying the higher-payoff neighbor
more deterministic.

The baseline config is:

```text
experiments/configs/toy4_public_goods_baseline.yaml
```

Run it with:

```bash
scripts/run_toy4.py --config experiments/configs/toy4_public_goods_baseline.yaml
```

Toy 2 payoff-threshold calibration must remain Toy 2-only. Toy 4 should reuse
the sampled/argmax action-selection structure where useful, but should not
promote payoff-threshold calibration to a global default.

## Baselines

Implemented baselines:

- Static public-goods game with no resource stock.
- Imitation baseline where agents can copy higher-payoff local neighbors.
- Optional commons stock variant with depletion and recovery.

Deferred baselines:

- Continuous contribution.
- Punishment or sanctioning.
- Reputation and trust.

## Neural Variant

Observation vector for v1:

```text
previous action
local contribution rate
own payoff EMA
group payoff mean
resource level
bias
```

If resource is disabled, `resource level` should be a constant 1.0 so the model
input dimension remains stable.

Model output:

```text
softmax over [free ride, contribute]
```

Coordination output mixer:

```text
p_contribute_i <- (1 - alpha) * p_contribute_i + alpha * mean(peer p_contribute)
```

For the neural path, output mixing is implemented as policy distillation toward
the mixed target, so its behavioral effect appears through subsequent decisions.
For the imitation path, output mixing applies to candidate contribution
probabilities before revised actions are sampled.

## Metrics

Aggregate metrics:

- Contribution rate.
- Mean payoff.
- Payoff variance.
- Payoff Gini coefficient.
- Resource level.
- Resource fraction.
- Collapse time, defined as first epoch where resource reaches zero or a
  configured collapse threshold.
- Contributor cluster count.
- Largest contributor cluster fraction.
- Exploitation index: mean payoff among low contributors minus population mean
  payoff, or another explicit high-payoff/low-contribution measure.
- Peer graph fragmentation.
- Mean peer count.

Micro-state fields:

- Action.
- Contribution probability.
- Payoff.
- Payoff EMA.
- Local contribution rate.
- Group payoff mean.
- Resource level.
- Peer IDs and peer count.
- Component ID.
- Revised flag.
- Local and social losses for neural runs.

## First Success Gate

Toy 4 is ready for implementation validation when:

- The static public-goods baseline shows free-riding pressure under default
  multiplier and cost.
- The resource variant can collapse under low contribution.
- Social output mixing measurably shifts contribution rate or collapse time.
- Logs make cooperation, payoff, inequality, and exploitation visible without
  relying on external post-processing.
