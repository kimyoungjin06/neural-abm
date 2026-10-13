# Information-spread policy ladder: stage D1 feasibility

Date: 2026-10-13

Status: `D1_FEASIBILITY_ONLY_NO_BEHAVIORAL_OUTCOME_REPORTED`

Design: [info-spread-policy-ladder-design-v1.md](../../../docs/research/info-spread-policy-ladder-design-v1.md).
Code: `dfc5372` on `agent/info-spread-policy-ladder`. Nothing is frozen or
registered. Stage D1 measures whether the design can run. It reports no
behavioral outcome of any arm.

## What was built

- **Domain-free core** (`scripts/policy_ladder_core_v1.py`).
  - Prior-anchored learners: residual linear, residual MLP, and prior-boosted
    trees.
  - Per-agent learners, trained as one batch.
  - Keyed randomness, an exploration floor, an outcome record with a
    visibility delay, and policy snapshots for table replay.
  - With the Pivot prior, the three learners reproduce the Pivot
    implementations byte for byte (10 tests). This is the first parity check
    for N14.
- **World and non-LLM arms** (`scripts/prototype_info_spread_policy_ladder_v1.py`).
  - The world: delayed-veracity spread.
  - Rule arms: threshold (F0) and the hand rule (F1).
  - Learning arms: pooled learners (Regime A) and per-agent learners
    (Regime B).
  - Precomputed tables, replaying four donor seeds as final policies or as
    trajectories.
  - The archetype machinery, with a stub in place of the LLM.
- **Nesting checks.** All 14 pass, each by exact decision digest:
  - the threshold arm matches an independent implementation;
  - the Granovetter knife edge holds: 20 sharers, or 1 when one threshold is
    raised;
  - learning switched off reproduces F1;
  - a constant table reproduces F1;
  - replaying a run's own trajectory reproduces the run;
  - identity archetypes answered by F1 reproduce F1;
  - an exploration floor of zero changes nothing.

## Feasibility at 1,000 agents

The run used 60 steps, 20 items per step, 2 replicates, and 4 donors. It
covered three geometries (linear, axis, null) in each regime. On the GPU
server's CPUs, with 6 workers, the whole profile took 100 s at 0.8 GB peak
memory.

| Regime | Arm | µs per agent-step | Revealed outcomes per agent, median (range over geometries) | Archetype queries per run |
|---|---|---:|---|---:|
| A | F1 hand rule | 6.7 | 76 (76–77) | |
| A | Pooled linear / MLP / tree | 13 / 70 / 19 | 56 / 62 / 48 | |
| A | Tables (final, trajectory) | 3–7 | 39–48 | |
| A | Archetypes, K = 2 or 4 threshold bins × 4 or 16 item classes | 9–10 | 76–81 | 479–3,517 |
| B | F1 hand rule | 9.1 | 66 (63–68) | |
| B | Per-agent linear / MLP | 26 / 159 | 34 (11–51) / 34 (19–53) | |
| B | Tables (final, trajectory) | 8–12 | 8–21 | |
| B | Archetypes, × 3 cost levels | 10–15 | 75–78 | 1,440–10,568 |

Revealed outcomes per agent count the agent's own fact-checked shares. They
measure the evidence a learner can use, not how well it uses it.

## Reading for the next stages

- **Evidence is not the binding constraint in Regime B.** A per-agent learner
  sees a median of 34 of its own revealed outcomes, and at least 11 in every
  geometry. This is unlike Pivot, where per-agent learners starved on
  neighbor-scale evidence (C11). D2 still has to show the power of the
  closed-loop test.
- **No GPU is needed at 1,000 agents.** The heaviest arm, per-agent MLP, takes
  about 10 s per run. The GPU matters for the proposed scale cell (10⁴ to 10⁵
  agents) and for the LLM arms.
- **The archetype budget scales with K × item classes × regime.**
  - Planning assumption: about 0.2 s per query at batch 1, from the earlier
    elicitation (3,104 queries in under 10 GPU minutes).
  - At that rate, the smallest design (K = 2, 4 classes) needs about 2 min
    per run in Regime A and 5 min in Regime B.
  - The largest (K = 4, 16 classes) needs about 35 min per run in Regime B.
  - Reusing identical prompts would lower this. The counts come from the stub,
    whose answers differ from the LLM's, so they are planning figures, not
    bounds.

## Changes to the design brief

- **Reconsidered declines.** A decline is reconsidered at each new exposure,
  and a share is absorbing, as in Watts–Granovetter dynamics. Deciding once
  per item cannot reproduce the threshold reduction.
- **Archetype attributes.** Archetypes group agents by threshold quantile,
  plus cost level in Regime B. Degree is nearly constant on a Watts–Strogatz
  graph, so it cannot form groups.

## Boundary

No behavioral outcome was reported, no LLM was called, and no empirical data
was read. This record fixes nothing for D3.
