# Toy 5: Neural Contagion and Adoption

Toy 5 tests threshold diffusion, complex contagion, and adoption cascades.
The model should make it clear when cascades fail because exposure is too weak,
thresholds are too high, or the graph fragments.

The first implementation is a discrete-adoption vertical slice with simple
contagion, complex threshold diffusion, heterogeneous thresholds, optional
output mixing, and a neural adoption policy.

## Purpose

Toy 5 should answer:

- Can Neural ABM reproduce simple and complex contagion baselines?
- Do heterogeneous thresholds produce partial cascades?
- Does social output mixing change cascade size without breaking classical
  threshold behavior?
- Can logs capture exposure-response behavior by threshold group?

## State and Action Space

Agent adoption state:

```text
adopted_i in {0, 1}
0 = not adopted
1 = adopted
```

Optional agent attributes:

```text
threshold_i in [0, 1]
susceptibility_i in [0, 1]
exposure_count_i >= 0
```

The first implementation should keep adoption state discrete. Belief and
susceptibility learning can be added after the baseline cascade behavior is
validated.

## Config Schema

Toy 5 uses these top-level blocks:

```yaml
run:
simulation:
model:
  policy:
  agents:
  coordination:
  state:
domain:
  toy: toy5
  environment:
  graph:
logging:
```

Default values for v1:

```yaml
model:
  policy:
    rule: complex_threshold
    learning_enabled: true
    revision_rate: 1.0
    temperature: 1.0
    decision:
      mode: sampled
      action_temperature: 1.0
      exploration_epsilon: 0.0
    domain:
      repeated_exposure_decay: 0.0
      adoption_is_absorbing: true
  agents:
    count: 100
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
  toy: toy5
  environment:
    initial_action_fraction: 0.05
    seed_selection: random
    threshold_mode: homogeneous
    homogeneous_threshold: 0.25
    heterogeneous_threshold_low: 0.15
    heterogeneous_threshold_high: 0.55
    simple_contagion_probability: 0.08
  graph:
    type: watts_strogatz
    k: 6
    rewire_probability: 0.1
```

If `initial_action_fraction` is positive but rounds to zero for a small
network, the implementation seeds one initial adopter so nonzero seed
configurations cannot silently produce seedless cascades.

Toy-specific CSV and summary fields use `domain_*` names, for example
`domain_cascade_size`, `domain_time_to_50_action`,
`domain_failed_cascade`, and `domain_threshold`.

The baseline config is:

```text
experiments/configs/toy5_contagion_adoption_baseline.yaml
```

Run it with:

```bash
scripts/run_toy5.py --config experiments/configs/toy5_contagion_adoption_baseline.yaml
```

## Baselines

Implemented baselines:

- Simple contagion: each adopted neighbor gives an independent adoption chance.
- Complex contagion: adoption occurs when adopted-neighbor fraction exceeds an
  agent threshold.
- Heterogeneous thresholds: thresholds are sampled from configured groups or a
  configured distribution.

Deferred baselines:

- Recovery or disadoption.
- Competing contagions.
- Dynamic network rewiring.

## Neural Variant

Observation vector for v1:

```text
current adopted state
neighbor adoption rate
repeated exposure count
degree
utility proxy
bias
```

Model output:

```text
softmax over [not adopted, adopted]
```

The first neural variant should treat adoption as absorbing by default. If an
agent has already adopted, the realized state stays adopted even if the neural
output later favors non-adoption.

Coordination output mixer:

```text
p_adopt_i <- (1 - alpha) * p_adopt_i + alpha * mean(peer p_adopt)
```

For classical baselines, output mixing is applied to candidate adoption
probabilities before revised states are sampled. For the neural path, output
mixing is implemented as policy distillation toward the mixed adoption target.

An alternative threshold-aligned mixer may be added later:

```text
adoption_pressure_i <- local_exposure + alpha * peer_output_pressure
```

## Metrics

Aggregate metrics:

- Adoption rate over time.
- Cascade size, equal to final adoption count or fraction.
- Time to 50 percent adoption, blank if never reached.
- Failed cascade indicator, defined as current or final adoption rate below 50
  percent.
- Adoption cluster count.
- Largest adoption cluster fraction.
- Threshold-group adoption rates for homogeneous, low-threshold, and
  high-threshold groups.
- Mean neighbor adoption rate.
- Mean repeated exposure count.
- Peer graph fragmentation and mean peer count.

Micro-state fields:

- Adoption state.
- Adoption probability.
- Threshold.
- Neighbor adoption rate.
- Repeated exposure count.
- Degree.
- Peer IDs and peer count.
- Component ID.
- Revised flag.
- Newly adopted flag.
- Threshold group.
- Utility proxy.
- Local and social losses.

## First Success Gate

Toy 5 is ready for implementation validation when:

- Low thresholds produce cascades.
- High thresholds block cascades.
- Heterogeneous thresholds produce partial cascades.
- The neural/social variant changes cascade size without breaking the baseline
  simple and complex contagion dynamics.
- Summary outputs expose cascade size, time to 50 percent adoption, failed
  cascade status, and exposure-response by threshold group.
