# Decision 0001: Project Positioning

Status: Accepted

Date: 2026-04-29

## Context

The initial reference report argues that NABM can be interpreted as a minimal
unit intelligence for ABM, with MLP agents and social updates based on state or
output similarity. Later discussion clarified that the Transformer/MLP-Mixer
analogy is useful but technically limited.

The project needs a defensible position before implementation.

## Decision

Position the project as:

```text
Neural ABM
= temporal heterogeneous GNN-style simulation framework
+ neural agents
+ social mixing functions
+ local learning
+ explicit ABM scheduling and logging
```

Do not claim that the Neural ABM Node is already a Transformer-grade universal
primitive.

Use two direct technical lineages:

- Parameter-level social update should be compared with federated learning and
  parameter averaging methods.
- Output-level social update should be compared with ensemble or federated
  distillation methods.

Use graph learning language for dynamic social mixing:

- Peer selection as graph construction.
- Social influence as edge weight.
- Learned mixer as a dynamic edge function.

## Consequences

- The first implementation should prioritize small toy models, ablations, and
  logging over complex agents.
- Parameter averaging must include an alignment/no-alignment ablation.
- The project should record micro-state logs from the first experiment.
- Toy 1 should use a supervised task to give a clean correctness signal.
- Toy 2 should use a game-dynamics task to test social behavior transfer.
