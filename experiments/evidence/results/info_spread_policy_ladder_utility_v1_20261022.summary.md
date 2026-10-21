# Information-spread policy ladder: utility study v1 (O25)

Date: 2026-10-22

Status: `O25_SOME_CONCLUSIONS_CHANGE_UNDER_UTILITY`

Design: [utility study v1](../../../docs/research/info-spread-policy-ladder-utility-v1.md),
frozen at `ebd2c8d` before the run.

## What ran

Every registered arm was scored by the agents' utility on unverified items,
in the same cells and runs:

- a share of a true item gains 1;
- a share of a false item costs the sharer c;
- a decline is worth 0.

**Regime A.** Every cost is 1, so utility came from the recorded counts of
study v2, D5, and O24, with no run.

**Regime B.** Every arm was rerun: F1, T0-1, T0-4, L0, P0, X0, L1, P0u, and
L1u.
- **Parity.** Every rerun row reproduced its recorded counts, precisions,
  reach, and decisions exactly. There were 0 parity failures.
- **Times.** On GPU-Server (GPU 2, 24 workers), 156 minutes for the
  development run and 155 for the reproduction, against a quoted estimate of
  2–3.5 hours.
- `cells.jsonl` and `summary.json` match byte for byte between the two runs.

**The minimum effect** was fixed before any utility was computed, as 1% of
F1's mean absolute utility. That gives 0.59 per agent per run in Regime A and
0.15 in Regime B.

## Registered result

**Overall: `SOME_CONCLUSIONS_CHANGE_UNDER_UTILITY`.** Of 20 comparison and
regime summaries, 15 agree with precision and 5 change.

| Comparison | Regime | Under precision | Under utility |
|---|---|---|---|
| L0 − T0-4, the best table | A, B | `MOSTLY_NEGATIVE` | `MOSTLY_NEGATIVE` (6 of 6 each) |
| L0 − T0-1, the closed loop | A, B | `MIXED_OR_NONE` | `MIXED_OR_NONE` |
| L0 − F1, the hand rule | A, B | `MOSTLY_POSITIVE` | `MOSTLY_POSITIVE` |
| L0 − X0, NABM against archetypes | A, B | `MOSTLY_POSITIVE` | `MOSTLY_POSITIVE` (6 of 6 each) |
| P0 − F1 | A / B | `MOSTLY_NEGATIVE` / `MOSTLY_POSITIVE` | same |
| P0u − F1 | A, B | `MOSTLY_POSITIVE` | same |
| L1u − P0u | A | `MIXED_OR_NONE` | same |
| L1u − L0 | A | `MIXED_OR_NONE` | same |
| **L1 − L0, the LLM prior in the learner** | **A** | `MIXED_OR_NONE` | **`MOSTLY_NEGATIVE` (6 of 6)** |
| **L1 − L0** | **B** | `MIXED_OR_NONE` | **`MOSTLY_NEGATIVE` (4 of 6)** |
| **L1u − L0** | **B** | `MIXED_OR_NONE` | **`MOSTLY_NEGATIVE` (4 of 6)** |
| **L1 − P0, the closed loop on the LLM prior** | **B** | `MIXED_OR_NONE` | **`MOSTLY_POSITIVE` (4 of 6)** |
| **L1u − P0u** | **B** | `MIXED_OR_NONE` | **`MOSTLY_POSITIVE` (5 of 6)** |

## Reading

Within this design:

- **The framework's main comparisons hold under utility.**
  - Tables written before the run beat the adaptive learner in stationary
    worlds.
  - The closed loop shows no gain at matched evidence.
  - Learners beat the hand rule.
  - NABM beats the LLM archetypes.
- **What precision hid was the cost of selectivity in the LLM priors.**
  - **Inside the learner, an LLM prior lowers utility, rather than "adding
    nothing".** In Regime A, D5's clipped prior is close to "never share".
    A learner started from it shares less: its precision stays about the
    same, but it forgoes shares worth making.
  - **Against the prior alone, the closed loop now wins in Regime B.** The
    priors there are selective. They looked good on precision but share too
    little to match the learner's utility.

**Boundary.**
- **One utility,** the one the agents' decision rule assumes. Effects on
  others, such as the reach of false items, are not scored.
- **Null groups** are reported, not judged.
- **The worlds are synthetic.**
