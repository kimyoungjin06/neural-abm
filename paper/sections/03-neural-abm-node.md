# Section 3 Draft: Neural ABM Node

Status: draft prose candidate.

Source table: `paper/tables/nabm-unit-v1-manuscript-tables.md` Table 1.

## 3.1 Motivation

The purpose of the Neural ABM Node is not to replace classical agent-based
models with a black-box neural simulator. The useful unit is narrower: a
reusable neural lifecycle that can be attached to domain-specific ABM
semantics without absorbing those semantics into a generic learner. This
distinction matters because many ABM rules are valuable precisely because their
state variables, transition rules, and intervention points are explicit.

The project therefore separates reusable neural infrastructure from domain
meaning. The reusable layer owns update order, typed exchange, backend dispatch,
and diagnostics. Domain adapters retain the meaning of observations, rewards,
thresholds, environmental transitions, teacher signals, basin signals, and
evidence gates. In other words, the node is a disciplined execution contract
rather than a universal behavioral model.

![NABM Unit recurrent block with explicit ownership boundaries](../../docs/figures/nabm_unit_recurrent_block.svg)

**Figure 1. The NABM Unit as an auditable recurrent block.** Panel A shows how
one synchronous call orders optional local adaptation, validated typed-message
construction, peer selection, social mixing, optional adapter commit, and
diagnostics across agents. Domain adapters retain state meaning and
environmental transitions, after which the caller repeats the block over time;
`×N` marks protocol repetition and does not imply shared parameters. Panel B
maps the deterministic classical examples to their actual torch-free
bounded-scalar path rather than implying direct execution of the full class
path. The four exposed controls are the local rule, peer rule, exchange channel
and mixing strength, and commit rule.

## 3.2 Unit Contract

The current NABM Unit v1 contract has five reusable surfaces. First, the
generic lifecycle fixes the order of local and social updates and preserves
typed state exchange. Second, the binary policy lifecycle wraps neural readout,
probability construction, decision sampling, local commit, refresh, and
post-readout ordering behind `BinaryPolicyLearningUnit`. Third, the optional
binary revision lifecycle provides stay/switch sequencing through
`BinaryRevisionLearningUnit` without deciding what a stay or switch means in a
domain. Fourth, readiness propagation aggregates peer-readiness values only
after a domain has defined threshold, direction, confidence, and commitment
semantics. Fifth, backend commit adapters hide loop, batched, and tensor-runtime
dispatch differences without owning accelerator caches or domain state
transitions.

This decomposition is intentionally conservative. It makes the reusable unit
responsible for coordination, not reward design. Toy2 still owns game payoffs,
counterfactual advantages, and basin handoff. Toy4 still owns public-goods
groups, resource dynamics, local-sustain observation, welfare/resource
advantages, and resource-threshold meaning. Toy5 still owns adoption and
readiness meaning. The shared unit can execute the lifecycle across these toys,
but it does not convert their objectives into one hidden generic objective.

## 3.3 What The Unit Claim Means

The architecture claim supported by the current evidence is that Toy2, Toy4,
and Toy5 can route key binary policy, readiness, and diagnostic lifecycle
surfaces through shared unit infrastructure while preserving domain-specific
semantics in adapters. This is a reusable infrastructure claim. It is not a
claim that a neural policy generally outperforms Fermi, RD, reputation
imitation, threshold rules, or other classical ABM baselines.

This boundary is useful for two reasons. First, it makes cross-toy reuse
auditable: a new toy can be checked against observation builders, policy
readout callbacks, decision samplers, local commit hooks, refresh hooks, social
message builders, peer selectors, social commit adapters, diagnostics mappers,
and evidence manifests. Second, it prevents implementation convenience from
weakening the scientific claim. Moving Toy-specific rewards, thresholds, or
evidence criteria into the generic unit would make the code shorter, but it
would make the mechanism less inspectable.

## 3.4 Manuscript Insertion Notes

Use Table 1 after the first paragraph that introduces the NABM Unit v1
contract. The caption should emphasize that the unit owns lifecycle order,
typed exchange, backend dispatch, and diagnostics, while adapters retain
reward, threshold, teacher, basin, readiness-meaning, and evidence-gate
semantics.

Do not use this section to claim baseline dominance. The strongest wording
available here is:

> The current implementation supports a reusable NABM Unit v1 contract for
> binary policy, revision, readiness, backend commit, and diagnostic lifecycles
> across Toy2, Toy4, and Toy5 while keeping domain semantics outside the
> reusable layer.
