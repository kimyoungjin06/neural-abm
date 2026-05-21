# Paper Outline

Working title:

> Neural Social Nodes for Agent-Based Modeling

## 1. Introduction

- Motivation: rule-based ABM agents are interpretable but limited in adaptive
  learning behavior.
- Goal: define a reusable Neural ABM Node with explicit social mixing.
- Claim: the right positioning is a temporal heterogeneous graph simulation
  framework, not a direct Transformer replacement.

## 2. Background

- Agent-based models and bounded confidence.
- Neural agent-based models.
- Social learning via state-based and output-based updates.
- Dynamic graph learning, federated learning, and distillation lineages.

## 3. Neural ABM Node

Draft: [sections/03-neural-abm-node.md](sections/03-neural-abm-node.md)

- Fast, medium, and slow state.
- Common node interface.
- NABM Unit v1 contract: what the reusable unit owns, and which domain
  semantics stay outside it.
- Social mixing taxonomy.
- Communication budget and synchronization modes.
- Logging requirements.

## 4. Toy 1: Neural HK Classification

- Biased private data shards.
- State-similarity versus output-similarity peer selection.
- Mixing ablations.
- Accuracy, consensus, polarization, and fragmentation.

## 5. Toy 2: Neural Spatial Prisoner's Dilemma

- Game dynamics and social mixing.
- Cooperation, payoff, exploitation, and cluster formation.
- Transfer of the same mixer taxonomy from Toy 1.

## 6. Calibration and Analysis

Draft:
[sections/06-calibration-and-analysis.md](sections/06-calibration-and-analysis.md)

- Manuscript claim matrix: every claim must map to code path, manifest, result
  artifact, figure/table candidate, and limitation.
- Manuscript table candidates: unit-boundary, Toy5 safety/spread, Gate 3
  failure modes, Toy2/Toy4 reputation fragility, and Toy4 local resource stress.
- Toy5 threshold-aware readiness grid: no-seed safety and sparse-seed spread.
- Toy2/Toy4 failure-mode taxonomy: stochastic final-epoch hazard, slow TtC,
  baseline-favored environment, and mechanism failure candidate.
- Targeted Toy2/Toy4 robustness: sparse seeds, open boundaries, noisy ranking,
  and Toy4 heterogeneous resource extraction.
- Micro-state logs.
- Sensitivity analysis.
- Time-to-ceiling as a secondary diagnostic: distinguish collapse recovery,
  slow monotone climb, ambiguous probability dwell, threshold oscillation,
  polarization, and policy-action lag before treating speed as an objective.
- Possible simulation-based inference or temporal-GNN posterior estimation.

## 7. Discussion

- What the toy models show.
- Limits of the Transformer analogy.
- Limits of parameter averaging.
- Path to richer ABM environments.

## 8. Conclusion

- Neural Social Nodes are a practical unit for NABM prototyping.
- The primitive claim should be earned through ablations, not assumed.
