# Pivot persistence and exit prototype v1

Date: 2026-08-07

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_persistence_exit_prototype_v1.yaml`](../../experiments/configs/pivot_persistence_exit_prototype_v1.yaml)

## Decision

The delayed-feedback prototype produced an informative phase map, but also
showed that failure evidence often arrived after almost every agent had made
an absorbing entry. Learned policy states then diverged while final entry
outcomes remained nearly identical. The next Pivot extension therefore changes
the observation endpoint rather than the generic NABM lifecycle.

This prototype adds domain-owned persistence and failed exit. It keeps the
entire entry and learning process nested inside the delayed-feedback
predecessor. Its question is whether a late policy correction can change the
stock of active failed pivots even when cumulative entry can no longer change.

## State model

The registered states are:

```text
NOT_YET_ENTERED
    -> ACTIVE_PRODUCTIVE -> remains active in v1
    -> ACTIVE_FAILED -> EXITED_OR_REDIRECTED
```

Entry and the private productive-fit outcome are exactly the predecessor
rules. A failed entrant remains active for at least two simulation steps. It
then receives a continuation score:

```text
0.5 * structural productive fit + 0.5 * current learned policy output
```

The agent exits when this score is below `0.40`. Productive entrants remain
active in v1. Exited agents do not re-enter. These choices are fixed before
execution and are stylized domain assumptions, not fitted behavioral rules.

The agent's current learned policy output is already updated through the NABM
local-learning lifecycle even after entry. Readiness broadcasts, social
mixing, topology, private outcome assignment, and public event availability
remain unchanged. Exit does not remove an agent from the network in v1. This
holds the learning and entry process constant and isolates the persistence
endpoint; dynamic topology remains deferred.

## Exact nested baseline

Every run contains both `absorbing_entry` and `persistence_exit_v1`. The
absorbing row calls the frozen delayed-feedback predecessor. The persistence
row uses the same population, network, local shocks, policy updates, event
queue, and entry commit order.

Consequently, final cumulative entry and its complete trajectory must be
exactly identical between state models within every replicate, visibility
regime, and learning arm. Any violation is an implementation failure, not an
empirical result.

## Comparison surface

The focused design retains immediate visibility, two- and four-step failure
delay, and four-step delay with half of failure outcomes publicly eligible.
Success remains immediately visible. Frozen, imitative, and cautionary arms
are retained. The development profile uses 80 agents, 18 steps, and 16 paired
replicates.

Primary outputs are cumulative entry, active productive and failed stocks,
failed exit, productive share among active agents, public-release trajectories,
and learned attention-weight trajectories. Comparisons are paired across the
state model, visibility regime, and learning policy.

There is no success threshold or post-result phase classification. Registered
outcomes include delayed exit, selective-visibility persistence, newly visible
policy divergence, dominance by the continuation rule, and no added
information from the nonabsorbing endpoint.

## Framework and claim boundary

Persistence, productive fit, exit, and redirection are Pivot meanings and
remain outside `NABMUnit`. The generic API is unchanged. The result may support
a later reusable observation or transition contract only after another model
family demonstrates the same need.

The prototype reads no empirical rows, fits no empirical target, selects no
model, establishes no neural necessity or scientific effect, and authorizes no
package promotion.
