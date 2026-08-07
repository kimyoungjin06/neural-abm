# Pivot to NABM flagship case-study bridge v1

Date: 2026-08-07

Status: `DESIGN_BRIDGE_ONLY_NO_EXECUTION_NO_MODEL_SELECTION`

Machine-readable companion:
[`pivot_nabm_case_study_bridge_v1.yaml`](../../experiments/configs/pivot_nabm_case_study_bridge_v1.yaml)

## Decision

The project goal is to complete and improve NABM as a reusable, inspectable
framework for adaptive agent-based models. Researcher Pivot is the flagship
case used to expose missing framework capabilities, test them in a nontrivial
domain, and demonstrate when neural social learning adds value over fair
simpler models.

Lean H3 is subordinate to that goal. Its entry-versus-persistence timing object
is one empirical constraint on simulated observables. It is not the project
destination, and estimator development should not continue unless it supplies
a named case-study target, applicability boundary, or baseline input.

The resulting hierarchy is:

```text
complete and improve NABM
    -> build a compelling Pivot flagship case
        -> constrain the case with bounded empirical observables
            -> develop H3 timing measurement only as one comparison interface
```

This bridge is not another empirical gate. It defines the next useful model
and framework work before any new execution.

## What is already worth retaining

The tracked stylized predecessor is already more than a smoke example.

- Study 1 shows that seed grants and hype can have similar mean productive
  gains with very different pivot and failure costs.
- Social mixing is regime-dependent: it can suppress supported exploration
  while consolidating hype-driven mass movement.
- Study 2 shows that imitative outcome learning can create a self-reinforcing
  loop.
- Failure-only cautionary learning can reduce failed pivots while also reducing
  productive exploration.

These findings make Pivot a promising NABM case because the important result is
not a monotone treatment effect. The same social-learning machinery can be
helpful, harmful, or self-limiting depending on support, visibility, and the
learning rule.

The predecessor nevertheless remains hand-set and stylized. Its pivot is a
one-time absorbing event; productive or failed status is realized and made
visible immediately; the network is fixed within a run; the social exchange
centers on one readiness channel; and no common adapter maps simulated curves
to the empirical set-valued timing object.

## Flagship question

The case study should ask:

> When attention and early success become visible before failure, when does
> social learning support productive exploration, and when does it create
> overshoot, lock-in, or delayed correction?

This is broader and more useful for NABM than asking only whether an empirical
timing gap is positive. It requires the model to represent private state,
public availability, heterogeneous learning, delayed evidence, and domain
transitions separately.

The central feedback sequence is:

```text
attention shock
    -> prepared and unprepared entry
    -> selectively visible early success
    -> social learning and additional entry
    -> later-cohort persistence weakens
    -> failure becomes publicly available after a longer delay
    -> imitation persists, caution grows, or policy weights adapt
    -> entry corrects, locks in, or shifts to another domain
```

The case remains informative if that sequence is absent. A simpler model
winning, observationally equivalent mechanisms, no empirical timing signature,
or unstable intervention rankings are all registered reportable outcomes.

## Empirical-to-simulation interface

The empirical layer constrains observable summaries. It does not assign an
agent's belief or identify a feedback mechanism.

| Empirical object | Simulated observable | Owner | Interpretation |
|---|---|---|---|
| Attention-surge clock | Attention-shock event time | Pivot environment | Align event time only. |
| Entry curve and practical peak set | First target-entry incidence by event time | Domain commit plus observation adapter | Compare set-valued timing, not one argmax. |
| Later-cohort persistence curve | Productive persistence or exit by entry cohort | Delayed outcome state machine | Supply decline timing and uncertainty. |
| Public-availability proxy | Time-eligible release of an outcome message | Event queue plus typed channel | Availability is not receipt or belief. |
| Capacity and bridge proxies | Resource, skill, network, and bridge agent state | Agent observation and local policy | Represent heterogeneity and baseline controls. |
| Four-field timing-set vector | Domain-stratified simulated timing-set vector | Case evidence adapter | Preserve the vector; do not replace it with a scalar mean. |
| Missing or zero-denominator cells | Explicit unidentified or not-applicable output | Case evidence adapter | Do not rescue with implicit imputation. |

The v3 stochastic set diagnostic matters here because it showed that valid
labels can coexist with undercovered intervals. Its `58.1%` field-vector
containment and zero containment for registered near-tie/set geometries mean
that the empirical adapter is not ready to score simulations against a single
precise H3 target. A successor must pass set-valued uncertainty and
applicability to the model-comparison layer separately.

## What NABM already supplies

The current framework already provides the right separation of concerns for a
first prototype:

- `NABMAgent` observation, prediction, local-update, social-message, and state
  logging lifecycle;
- typed bounded-scalar and distribution-valued social exchange;
- domain-owned commit adapters;
- per-agent torch policies and weight diagnostics;
- paired replicated scenario orchestration; and
- aggregate and micro audit artifacts.

The generic unit should continue to own lifecycle, typed exchange, backend
dispatch, and diagnostics. Pivot must own scientific meanings such as entry,
success, failure, delay, return, and evidence eligibility.

## Capability gaps exposed by Pivot

| Needed capability | Current gap | First implementation owner | Generalization rule |
|---|---|---|---|
| Delayed selective outcome queue | Pivot outcomes are visible immediately. | Pivot domain adapter | Extract only after a second model family needs the same lifecycle. |
| Private state versus public message | Realized and broadcast outcome are not separate. | Pivot observation adapter | Consider a reusable observation contract after case validation. |
| Attention, success, and failure channels | The current study mixes one readiness channel. | Pivot orchestrator using typed channels | Generalize with a second multi-channel consumer such as Toy10. |
| Persistence, exit, and return states | A pivot is one-time and absorbing. | Pivot domain state machine | Keep these scientific semantics outside `NABMUnit`. |
| Simulation-to-empirical set adapter | No shared timing-set result contract exists. | Case evidence layer | Reuse numerical envelopes, never framework-owned claim judgment. |
| Fair model ladder | Existing studies compare arms, not complete model classes. | Case runner | Hold inputs, randomness, scoring rows, and budget rules fixed. |

Generic dynamic topology, learned edges, and nonbinary tensor acceleration are
deferred until a diagnostic shows they are necessary. The case must not become
an excuse to add every possible framework feature.

## Fair model ladder

The flagship comparison uses the same population or risk set, attention and
outcome-availability process, topology, component-keyed randomness, data
partition, scoring rows, and compute-budget rule for every model.

| ID | Model | Question |
|---|---|---|
| B1 | Fixed statistical timing or hazard baseline | Are agent dynamics needed at all? |
| B2 | Fixed-rule ABM | Are heterogeneous agents sufficient without adaptation? |
| B3 | Explicit adaptive non-neural baseline | Can an interpretable Bayesian, bandit, or delta rule explain adaptation? |
| B4 | Local-only NABM | Does neural local learning help without social outcome exchange? |
| B5 | Imitative NABM | What follows from learning from visible success and failure? |
| B6 | Cautionary or mixed NABM | Does asymmetric or heterogeneous learning correct overshoot, and at what exploration cost? |

If B1-B3 match B4-B6 on disjoint predictive and generative targets, the simpler
model is the scientific result. Any independently useful lifecycle or
diagnostic improvement may remain in NABM, but neural necessity must be
rejected.

## Registered result space

The study is designed to remain interesting without guaranteeing a favorable
NABM result. It reports all of the following if observed:

- productive support without mass overshoot;
- hype-driven entry overshoot and delayed failure;
- imitative lock-in or self-reinforcement;
- cautionary correction with a productive-exploration cost;
- reversal of the social-mixing effect across environments;
- observational equivalence across distinct mechanisms;
- predictive NABM gain with unstable intervention ranking;
- no NABM gain and preference for a simpler model; or
- no empirical timing signature, leaving the case stylized.

This avoids making “NABM wins” a completion condition.

## What counts as project progress

Framework progress requires one of two explicit outcomes:

1. a reusable lifecycle, typed-exchange, observation, or diagnostic capability
   that is supported by more than one model family; or
2. an explicit decision that a useful Pivot mechanism remains domain-owned and
   should not expand the generic API.

Case-study progress requires reproducible trajectories, an empirical-alignment
adapter with honest denominators, a fair model ladder, capability ablations,
and negative-result reporting. A new empirical audit that supplies none of
those objects is not progress merely because another audit can be performed.

## Flagship outputs

The target narrative should be supported by six integrated visual objects:

1. observed and simulated entry/persistence curves with practical peak sets;
2. a visibility-delay by social-learning phase map;
3. a support by attention map of productive and failed pivots;
4. imitative, cautionary, and mixed-policy trajectories;
5. a fair baseline-ladder disjoint scorecard; and
6. a capability-ablation and framework-change ledger.

The figures must distinguish simulated mechanism behavior, empirical
measurement, predictive comparison, and intervention interpretation.

## Workstreams and convergence

Three workstreams now proceed with different roles.

```text
NABM framework -- primary goal
    -> reusable capability or explicit non-extraction decision

Pivot model -- primary stress test
    -> reproducible delayed-feedback dynamics and counterfactuals

Empirical measurement -- supporting constraint
    -> set-valued targets, applicability, and uncertainty

convergence
    -> fair baseline comparison and flagship case-study narrative
```

Empirical H3 work may proceed only when it supplies a named comparison target
or applicability boundary. Framework work may generalize a Pivot feature only
when a second model family needs the same semantics. These two rules replace a
long chain of micro-gates with direct relevance tests.

## Next build

The next executable object is
`PIVOT_DELAYED_FEEDBACK_LEARNING_PROTOTYPE_V1`. It should:

- keep success and failure visibility delays domain-owned;
- separate private realized outcome from publicly available message state;
- retain frozen, imitative, and cautionary policy arms;
- use immediate visibility as a nested predecessor baseline;
- produce a delay-by-learning phase map; and
- fit no empirical data.

The prototype should be precommitted only to prevent result-contingent rule
changes. It requires no additional governance review. The generic NABM API
must not change in this tranche; first determine whether the domain-owned
prototype creates a reusable contract gap.

## Boundary

This bridge authorizes documentation only. A separately frozen synthetic
prototype may follow. It does not authorize empirical-row access, opened
challenge reuse, model selection, fresh blind generation, holdout opening,
neural-necessity language, or a scientific-effect claim.
