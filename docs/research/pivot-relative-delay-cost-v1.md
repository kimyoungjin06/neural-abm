# Pivot relative delay and cost v1

Date: 2026-09-29

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_relative_delay_cost_v1.yaml`](../../experiments/configs/pivot_relative_delay_cost_v1.yaml)

## Why this study exists

This is step N01 of the
[policy-ladder consolidation](pivot-policy-ladder-consolidation-v1.md), extended
with a cost ledger. It answers open question O04 and adds the first
measurement behind NABM's intended position. That position is heavier than a
rule-based ABM, because agents learn policies, and much lighter than an
LLM-agent simulation.

**O04.** Nonlinear policy ABM v2 found that learning changes who enters only
when failure is visible at once, and that entry saturates from a two-step
delay. But entry in that ABM is compressed. In all 1,440 v2 runs, entry
incidence peaked at the first open step, and at least 53% of final entry
happened there. "Saturation from a two-step delay" may therefore mean
"failure evidence arrived after the entry wave had ended". If so, the
comparison with data must use delay relative to entry-wave width, not
absolute delay.

**Cost.** The middle-ground claim needs numbers. This study measures the
compute cost of each policy family in the same ABM. It also counts the policy
queries an LLM-agent version of the same design would need.

## Design

**Entry hazard.** An agent whose committed readiness meets the threshold
enters only if a uniform draw falls below `q`. Each draw is keyed to
(seed, replicate, step, agent), so it perturbs no other randomness. At
`q = 1` the adapter is exactly the persistence predecessor. `q = 0.5` and
`q = 0.25` widen the entry wave, a domain-owned form of entry friction.

**Grid.** Three hazards (`q` = 1, 0.5, 0.25) are crossed with four failure
delays (0, 1, 2, and 4 steps) and the three v2 fit worlds. The arms are the
fixed rule and the field-wide cross-fitted linear, tree, and MLP learners from
v2, with settings unchanged. The development profile uses 256 agents, 24 steps
to leave room for the widened wave, and 16 replicates. That is 2,304 runs.
Seeds are fresh. The expensive neighbor-evidence learners do not bear on O04,
so they appear only in the cost benchmark.

**Nesting.** At `q = 1` in the linear world, the fixed arm must equal the
persistence predecessor. Under immediate visibility, the MLP field arm must
also equal the v2 arm. Either mismatch stops the run.

## Registered decisions

**Entry channel per cell.** For each hazard and delay, the tree or MLP must
raise the productive share of entrants over the linear policy by at least 0.02
beyond the same gap in the linear world, in at least 12 of 16 replicates. This
is the v2 rule. A cell is:

- `ENTRY_CHANNEL_ACTIVE` if any nonlinear family-geometry pair passes;
- otherwise `ENTRY_SATURATED` if the linear, tree, and MLP arms reach at least
  98% final entry in 12 replicates of every world; and
- otherwise `ENTRY_CHANNEL_ABSENT`.

**Relative delay.** Entry-wave width is the number of steps the fixed rule
needs, from the first open step, to reach 90% of its final entry. It is
averaged per hazard. Relative delay is the first public failure lag (delay + 1)
divided by that width.

**Relative-delay test.** Two facts decide the test. Is the set of active
delays the same at every hazard? Does a single relative-delay threshold
separate active cells from inactive ones?

| Same active delays at every q | One threshold separates | Disposition |
|---|---|---|
| No | Yes | `RELATIVE_DELAY_SUPPORTED` |
| Yes | No | `ABSOLUTE_DELAY_SUPPORTED` |
| Yes | Yes, or all cells share one status | `NOT_DISCRIMINATED` |
| No | No | `NEITHER_SIMPLE_RULE` |

## Simulation-side timing observables

The runner records the observables the empirical H3 comparison will need:

- entry incidence per step;
- the entry peak set;
- cohort persistence four steps after entry;
- the first later cohort whose persistence falls 0.05 below the first
  cohort's; and
- the model analogue of the H3 gap.

These are descriptive. A full H3 comparison needs an interior entry peak. The
current entry process, even with a hazard, may keep the peak at the first open
step, and producing an interior peak would need an attention-surge clock this
model does not have. The study reports how often the peak stays pinned.

## Cost ledger

A separate `benchmark` command times each of the ten v2 arms on one
development-size task: replicate 0, immediate visibility, axis world, `q = 1`.
It uses single-thread process CPU time, three repeats, and the median. It
reports cost per agent-step and relative to the fixed rule, and records machine
load before and after.

Timing is not byte-reproducible, so the ledger is kept apart from the
scorecard. For the LLM end, the runner counts policy queries (agents × steps)
and pre-entry decisions per run. These are the calls an LLM-agent version would
make. No price or latency is assumed.

## Boundary

The executable must be committed before the development profile runs, and a
deterministic rerun must reproduce every scorecard artifact byte for byte. No
post-result change to hazards, delays, learners, or thresholds is authorized.

This is a synthetic generative comparison. It reads no empirical rows, selects
no model, and changes no generic NABM API. The entry hazard is a stylized
friction, not a claim about how real researchers time their entry.
