# N14 parity: the Pivot ladder re-expressed through the shared core

Date: 2026-10-19

Status: fixed before the parity run.

## Why

The information-spread study used a domain-free core,
`scripts/policy_ladder_core_v1.py`, lifted from the Pivot policy ladder.
The design brief's reusability test (N14) asks for more than a second
consumer. One recorded Pivot configuration must run through the same core
and reproduce its recorded result
([brief](info-spread-policy-ladder-design-v1.md), "Reusability test").

The user set the parity host on 2026-10-19: byte for byte, on the
workstation where the Pivot result was recorded.

## Target

The [Pivot LLM-prior pilot v3](pivot-llm-prior-pilot-v3.md), frozen at
`87c90eb`. It is the only recorded Pivot run that uses all three of these
together:

- the prior slot;
- the exploration floor;
- the three core learner families.

The run itself:

- **Host:** CPU only, with no LLM call (it reuses the hash-bound v2
  calibrated prior);
- **Size:** 16 replicates × 3 fit worlds × 6 conditions, 4 arms each, giving
  1,152 rows;
- **Recorded rows:** `runs.jsonl`, sha256 `1f532d98…`. They are on this
  workstation in the development and reproduction directories, both with
  that hash.

## What is re-expressed

The runner is `scripts/reexpress_pivot_llm_prior_v3_through_core_v1.py`.
For the three field-public learner arms, the policy-ladder semantics come
from the core:

| Semantics | Pivot | Core |
|---|---|---|
| The learners' prior | The prior slot replaces `_initial_policy` | `core.LinearPrior` from the condition |
| Learners | `fit_batch` (matched linear, residual MLP, prior-boosted trees) | `core.fit_residual` (linear, mlp) and `core.fit_prior_boosted_trees` |
| Refit seed | `SeedSequence((*components, refits))` | `core.keyed_seed(*components, refits)` |
| Outcome record | A feature and label list per fold learner | `core.OutcomeRecord` per fold learner |
| Exploration floor | `rng.random() < ε` on a keyed stream | `core.ExplorationFloor` on `core.keyed_uniform` |

The core learner is placed in Pivot's learner slot, with the same interface
and fold isolation.

**What stays in Pivot's code.**

- **The world:** the population, network, social step, persistence
  transition, and probe cohort.
- **The fixed-rule arm:** Pivot computes its rule in torch, and the core's
  numpy prior differs from it in the last bits.

## What the smoke checks showed before the freeze

**On the smoke profile.** All 18 tasks (replicate 1, every world and
condition) made the same rows through the core as through Pivot, byte for
byte.

**A second place for the prior.**
- **Where it lives.** Pivot holds the prior in two places:
  - as the learners' anchor;
  - as each agent's own policy parameters. The fixed rule uses these, and
    every arm reports them as `mean_attention_weight`.
- **Without the slot.** When the field arms ran without Pivot's prior slot,
  every decision, trajectory, and probe score was still identical. Only
  `mean_attention_weight` differed.
- **The core holds only the first.** So the runner keeps Pivot's prior slot
  entered for every arm to build the agents' own parameters. The field
  learners never read it.

## Runs and criterion

1. **Baseline.**
   - Pivot's own v3 code reruns replicate 0 (18 tasks, 72 rows).
   - Every row must equal the recorded row. This shows the workstation still
     makes the recorded rows.
2. **Parity.**
   - The re-expression runs the full study, with 16 workers, spawned
     processes, and one thread each, as recorded.
   - **Parity holds** if the whole `runs.jsonl` equals the recorded file
     byte for byte (sha256 `1f532d98…`).
   - Otherwise the mismatched rows and fields are reported, and parity does
     not hold.
3. **Record.** A summary and a result test.

The extraction decision (Decision 0017) follows the record, and uses its
result.

## Boundary

The parity run calls no LLM and reads no empirical data. It does not change
the package API or reinterpret any Pivot result. It tests only whether the
core expresses the Pivot ladder's semantics exactly.
