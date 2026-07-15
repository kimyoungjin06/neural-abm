# Paper Outline

Working title:

> Neural Social Nodes for Agent-Based Modeling

## 1. Introduction

- Motivation: rule-based ABM agents are interpretable but limited in adaptive
  learning behavior; fully neural and LLM-agent simulations adapt but resist
  attribution, audit, and seeded replication.
- Positioning: the auditable middle — the maximum agent adaptivity that
  still supports paired counterfactual control, named-parameter audit, and
  deterministic replication. Not a cheaper substitute for LLM-agent
  simulation and not a replacement for classical ABM; a different point with
  different guarantees.
- Goal: define a reusable Neural ABM Node with explicit social mixing.
- Claim: the right positioning is a temporal heterogeneous graph simulation
  framework, not a direct Transformer replacement.
- Contribution preview: unit contract (Sec. 3), toy evidence (Secs. 4-5),
  calibration discipline (Sec. 6), and an end-to-end research application
  with a fixed-rule-versus-learning controlled comparison (Sec. 7).

## 2. Background

- Agent-based models and bounded confidence.
- Verified classical instances and labeled near variants: exact DeGroot and
  Granovetter special cases, FJ-like pre-mix anchoring, and self-excluding HK
  bounded confidence as settings of the unit lifecycle:
  `docs/classical-reductions.md`, `examples/classical_reductions.py`.
- Neural agent-based models.
- LLM-agent social simulation and its audit/reproducibility critiques, as
  the opposite pole motivating the auditable-middle positioning.
- Social learning via state-based and output-based updates; imitative versus
  postulated failure-only outcome updates as the Sec. 7 learning-rule axis,
  with the canonical informational-cascade definition kept separate.
- Dynamic graph learning, federated learning, and distillation lineages.
- Model capacity as a modeling assumption: cue-weighting/improper-linear
  views of judgment and the interaction-first ABM tradition (grounds for
  Sec. 7.5).

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

## 7. Research Application: Replicated Scenario Studies

Draft:
[sections/07-research-application.md](sections/07-research-application.md)

- A researcher-authored science-of-science question (field-pivot
  productivity) carried end to end on the reusable surfaces.
- Study 1: baseline/counterfactual environments, seed-paired replication,
  placebo control, sensitivity sweeps (torch-free surface).
- Study 2: frozen versus imitative and failure-only learning agents on the
  NABM lifecycle; environment-dependent feedback loops, trade-offs, and full
  named-parameter traces as bounded endogenous rule-change results.
- Expressiveness and discipline claim only; the fixed rule is a control, not
  a defeated baseline.

## 8. Discussion

- What the toy models show.
- What the research application shows: the contract carries a full study,
  and learning direction is a first-class experimental axis.
- Limits of the Transformer analogy.
- Limits of parameter averaging.
- Path to richer ABM environments.

## 9. Conclusion

- Neural Social Nodes are a practical unit for NABM prototyping.
- The primitive claim should be earned through ablations, not assumed.
