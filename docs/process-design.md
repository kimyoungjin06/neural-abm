# Process Design

This document captures the current research process before implementation.

## Research Claim

The project studies whether a reusable Neural ABM Node can serve as a minimal
unit for agent-based simulations where agents learn from both private
environmental data and social interaction.

The defensible claim is:

> A Neural ABM Node can be modeled as a neural agent with explicit social
> mixing, evaluated as a temporal heterogeneous graph system.

The current non-claim is:

> A Neural ABM Node is not yet a general Transformer-grade primitive.

## Modeling Position

The strongest positioning is:

```text
Neural ABM system
= temporal heterogeneous GNN-style simulator
+ neural internal agent model
+ explicit social compatibility and peer selection
+ optional alignment or translation between agent states
+ typed social mixer over output, latent, parameter, action, or memory channels
+ local learning update
```

This framing preserves the useful intuition from Transformer/MLP-Mixer-style
mixing while avoiding an overclaim. The closer technical relatives are:

- Dynamic graph neural networks and temporal graph networks.
- Graph attention and learned edge functions.
- Federated learning for parameter-level update paths.
- Ensemble or federated distillation for output-level update paths.
- HK-style bounded confidence models for peer selection and phase behavior.
- Model alignment and permutation matching for parameter-level social paths.

## Non-Goals

- Proving a universal Transformer replacement.
- Running large LLM social simulations.
- Predicting real-world social systems before calibration.
- Treating parameter averaging as safe without an alignment ablation.

## Process Stages

### Stage 0: Framing and Structure

- Fix project structure.
- Document the core claim and non-goals.
- Define the Neural ABM Node interface.
- Define logging and metric requirements before experiments begin.

### Stage 1: Toy 1, Neural HK Classification

Purpose:

- Test whether social mixing improves learning under private biased data.
- Compare no-social, output, latent/message, parameter, and learned mixing.
- Measure accuracy, consensus, polarization, and fragmentation.

Expected result:

- The model should show when social mixing helps or hurts learning.
- State-similarity and output-similarity peer selection should produce different
  phase behavior.
- Raw and aligned parameter similarity should produce different peer graphs when
  agents use independent initialization.

### Stage 2: Toy 2, Neural Spatial Prisoner's Dilemma

Purpose:

- Move from supervised correctness to social/game dynamics.
- Reuse the same node and mixer concepts.
- Measure cooperation, payoff, cluster formation, and exploitation resistance.

Expected result:

- The same social mixer taxonomy should change cooperation dynamics and network
  structure.

### Stage 3: Calibration Path

Purpose:

- Convert toy simulations into data-generating systems with reproducible logs.
- Prepare for simulation-based inference or temporal-GNN posterior estimation.

Minimum requirement:

- Micro-state logs must be rich enough to reconstruct agent states, social
  messages, edge weights, outputs, and aggregate metrics over time.

## Decision Gates

Proceed from Stage 1 to Stage 2 only if:

- Social update variants produce measurable differences from no-social.
- Results are stable across multiple seeds.
- Micro-state logs can reproduce aggregate metrics.
- Parameter averaging behaves differently under aligned and unaligned settings.
- Peer selection and mixing effects can be separated in at least one diagnostic.

Proceed from Stage 2 to larger experiments only if:

- Cooperation and payoff dynamics differ meaningfully across mixer types.
- Dynamic peer selection changes cluster structure or exploitation behavior.
- Communication budget and synchronization mode have measurable effects.

## Failure Conditions

The approach should be reconsidered if:

- All social mixers behave indistinguishably from no-social.
- Results are dominated by random seed selection.
- Learned mixer behavior cannot be inspected from logs.
- The model needs task-specific hacks that break the common node interface.
