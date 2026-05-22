# Decision 0010: NABM Unit v1 Contract and Migration Boundary

## Status

Accepted.

## Date

2026-05-21

## Context

Decision 0009 established the general NABM unit as the project priority and
then recorded several migration slices:

- generic `NABMUnit`, `NABMStep`, and `NABMLocalStep` primitives;
- binary output-distribution distillation through loop, batched, and
  tensor-runtime commit adapters;
- binary local policy-gradient commit adapters;
- `BinaryPolicyLearningUnit` migration for Toy5, Toy4, and Toy2;
- typed callback extraction for the binary policy-learning lifecycle.

The Toy4 resource-threshold evidence ladder also clarified an important
research boundary. Additional stress manifests can keep producing useful
diagnostics, but they will not by themselves make the project a general NABM
architecture. The next useful step is to freeze the v1 unit contract and use it
as the migration gate for any future toy or holdout domain.

## Decision

The project now treats the following as the **NABM Unit v1 contract**.

### Unit-Owned Responsibilities

The reusable unit owns lifecycle sequencing, typed state exchange, backend
commit dispatch, and stable diagnostics.

| Surface | Unit responsibility |
| --- | --- |
| Local lifecycle | Run policy readout, action or revision selection, local-update commit, backend cache/runtime refresh, and post-update readout in a fixed order. |
| Social lifecycle | Validate bounded social messages, select or consume peers, build typed social values, mix through a declared channel, commit through an adapter, and emit aggregate/micro rows. |
| Typed channels | Preserve channel names and output kinds such as probability distribution, scalar probability, bounded scalar, state dict, tensor, action, or revision choice. |
| Backend boundary | Hide loop, batched, and tensor-runtime commit differences behind adapters without making the unit own accelerator caches or runtime internals. |
| Binary policy lifecycle | Provide `BinaryPolicyLearningUnit` for pre-policy readout, decision probability construction, action sampling, local commit, refresh, and post-readout. |
| Binary revision lifecycle | Provide `BinaryRevisionLearningUnit` for signal collection, stay/switch probability readout, revision choice sampling, local commit, refresh, and post-readout. |
| Readiness propagation | Provide generic binary peer-readiness aggregation and diagnostics where readiness values already have domain meaning. |
| Diagnostics | Emit stable aggregate and micro fields for local loss, social channel, commit mode, update norms, revision probabilities, readiness state, and peer evidence. |

### Domain-Owned Responsibilities

Domain adapters own the meaning of observations, rewards, pressures, objective
terms, and environmental transitions. These must not be hidden inside the
generic unit.

| Surface | Domain responsibility |
| --- | --- |
| Observation semantics | Build Toy-specific observations such as PD neighbor payoffs, public-goods group summaries, resource features, adoption signals, or reputation state. |
| Environment transition | Update actions, resources, graph state, payoff state, adoption state, or other world variables. |
| Objective construction | Compute material, social, welfare, risk, environment, threshold, counterfactual, basin, or bootstrap advantages. |
| Revision pressure meaning | Decide what makes stay, switch-to-1, or switch-to-0 desirable in the domain. |
| Readiness direction meaning | Decide what direction, threshold, confidence, or basin signal counts as evidence for commitment. |
| Reference policies | Keep hand-coded imitation, reputation imitation, Fermi, RD, or other classical ABM rules explicit as baselines or diagnostics. |
| Evidence gates | Define case-specific ceiling metrics, tolerances, stress variants, and claim groups. |

## Current Toy2/Toy4 Boundary

The current full-NABM binary spatial family should be read as:

- `BinaryPolicyLearningUnit` owns the common neural policy-learning sequence.
- `BinaryRevisionLearningUnit` owns the optional stay/switch lifecycle when
  `revision_operator_enabled=true`.
- `BinaryReadinessPropagationUnit` owns peer-readiness aggregation, not the
  meaning of readiness.
- Toy2 still owns game family, payoff recomputation, counterfactual advantage,
  basin handoff, bootstrap, and teacher-alignment diagnostics.
- Toy4 still owns public-goods group construction, resource transition,
  resource-threshold signals, local-sustain observation, basin handoff,
  bootstrap, and teacher-alignment diagnostics.

This boundary is intentional. Moving Toy2 or Toy4 objective terms into the
generic unit would make the code shorter but would weaken the NABM claim by
turning reusable infrastructure into hidden Toy-specific semantics.

## V1 Completion Bar

The project can call the structure a meaningful NABM Unit v1 when all of the
following are true:

1. Toy2 and Toy4 execute neural policy learning through the same binary unit
   lifecycle without changing their domain objective semantics.
2. At least one holdout toy, preferably Toy5 or a new small binary domain,
   uses the same unit lifecycle with adapter-only domain additions.
3. The holdout migration does not require new generic unit behavior beyond
   typed callbacks or diagnostics.
4. Evidence manifests compare the migrated path against a domain-appropriate
   hand-coded baseline and a negative-control variant.
5. Documentation states whether the result is speed, stability, robustness, or
   interpretability evidence. It must not imply universal baseline dominance.

## Migration Gate for New Toys

Any new toy that claims NABM-unit compatibility must provide this adapter
checklist before running large evidence sweeps:

| Adapter item | Required artifact |
| --- | --- |
| Observation builder | Function or method that returns typed per-agent observations. |
| Policy readout | Callback returning per-agent action probabilities or revision probabilities. |
| Decision sampler | Callback or `DecisionKernel` use with explicit sampled/argmax semantics. |
| Local commit | Callback that receives actions or revision-selected actions and commits backend updates. |
| Refresh hook | Callback for agent/cache/runtime synchronization after local commit. |
| Social message builder | Bounded message schema and typed social value builder. |
| Peer selector | Graph or message-based peer rule that is independent of the mixer. |
| Social commit adapter | Loop, batched, or tensor-runtime adapter for the chosen channel. |
| Diagnostics mapper | Aggregate and micro fields that preserve unit-owned fields plus domain-specific fields. |
| Evidence manifest | Small quick manifest with baseline, main, and negative-control groups. |

If a new toy needs to modify the unit itself before it can run, that is a
structural finding. The implementation should stop and document why the v1
contract was insufficient before adding another toy-specific workaround.

## Non-Goals

This decision does not claim that:

- the project is already a general-purpose ABM framework;
- neural policies should beat clean hand-coded rules in every toy;
- basin credit is a finalized learned critic;
- local resource thresholding is a generic social mechanism;
- revision probabilities are already learned from first principles.

The current claim is narrower: the project now has enough reusable lifecycle
structure to test whether domain adapters can plug into one neural ABM unit
without moving domain semantics into the generic layer.

## Next Work

1. Add a compact unit-boundary audit for Toy2 and Toy4 call sites.
2. Pick one holdout migration target.
3. Implement only adapter glue first.
4. Run a small manifest with baseline, migrated unit path, and negative
   control.
5. Update this decision only if the holdout requires changing the v1 contract.

## Later Consolidation Notes

Decision 0011 added bounded scalar exchange after Toy7 showed that continuous
extraction intensity should not be represented as scalar probability. Decision
0012 then consolidated the Toy6-10 migration parity slices. Those decisions
extend the typed exchange surface while preserving this decision's main
boundary: domain equations and evidence criteria remain outside the generic
unit.
