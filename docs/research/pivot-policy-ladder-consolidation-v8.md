# Pivot policy-layer ladder consolidation v8

Date: 2026-10-23

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v8.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v8.yaml)

Extends: [consolidation v7](pivot-policy-ladder-consolidation-v7.md), and
through it v1 to v6, which remain the records for everything before.

## Purpose

v7 found that, scored by the agents' own utility, starting the learner from
an LLM prior lowers utility (C61). It ranked first the question of why
(O28). The user chose O28 on 2026-10-22.

This document records what O28 established. Every claim cites a hash-bound
result and a field that must hold, and a test checks all of them. This
document executes nothing.

| Study | Freeze / record | Outcome | Runs (development / reproduction) |
|---|---|---|---|
| O28 prior calibration | `555fd49` / `629a49b` | `O28_ATTRIBUTION_DIFFERS_ACROSS_COSTED_CASES` | 271 / 272 min, byte identical |

## The design in one paragraph

Each LLM prior has two parts:
- **its direction,** the model's answers;
- **its calibration,** its location and steepness over D3b's design cohort.

D3b set the location of every LLM prior to the base rate 0.4. The hand rule
F1 sits near 0.68.

O28 crossed the two parts against F1 for each LLM prior (P0, and the
unclipped P0u) and each regime. Two new learner arms filled the corners the
records lacked:
- the LLM's answers at F1's calibration;
- F1's answers at the LLM's calibration.

Each factor was then compared along both routes from L0 to L1. Three
"costed cases" had shown a utility loss in O25: P0/A, P0/B, and P0u/B.

## What is established

All results are synthetic. Ledger ids are in parentheses.

- **For D5's prior, the cost comes from its calibration, not from the
  model's answers (C63).**
  - Giving the learner P0's calibration lowers utility along both routes:
    6 and 6 of 6 Regime A groups, 4 and 4 of 6 Regime B groups.
  - P0's answers lower it along one route only.
  - At P0's own calibration its answers make no material difference in
    Regime A, where the prior's logit sd over the cohort is 0.06.
- **For the unclipped prior, answers and calibration interact (C64).**
  - Its answers lower utility at F1's calibration (6 of 6 Regime B groups).
  - They raise it at their own calibration: 4 of 6 groups in each regime.
  - Its calibration lowers utility given F1's answers (4 of 6) and is mixed
    given its own.
- **No LLM-informed start beats the hand rule's start under utility (C65).**
  - Of the 8 prior and regime cases against L0, 7 are `MOSTLY_NEGATIVE` and
    one is `MIXED_OR_NONE`.
  - None is `MOSTLY_POSITIVE`.
- **A prior recalibrated on the cohort does not share like its source in the
  world (C66).**
  - Every recalibrated prior moved toward its calibration source, as
    registered, but overshot it.
  - LLM directions at F1's calibration share 1.35 to 1.48 times as much as
    F1.
  - F1's direction at an LLM calibration shares 20 to 46% as much as the
    LLM prior, or, with P0's Regime A calibration, as little.
- **O28 reproduced byte for byte (C67).** L0 parity held in all 240 learner
  tasks, and every group's L1 − L0 and L1u − L0 equalled O25's record.

## What is rejected

| Claim | Why it does not hold |
|---|---|
| Calibration carries the cost in every costed case, O28's stated expectation (R20) | It holds for D5's prior in both regimes. The unclipped prior in Regime B is `NEITHER_CONSISTENTLY` |

**C61 is now explained.** For D5's prior, the cost comes from its
calibration (C63). For the unclipped prior in Regime B, the answers and the
calibration interact (C64).

## What is open

| Rank | Question | Kind |
|---|---|---|
| 1 | Do C63 and C64 hold when a prior is recalibrated to match its source's share rate in the world, not its calibration on the cohort? (O33) | Synthetic |
| 2 | Is it the location (the base-rate shift) or the steepness that carries the cost of D5's prior? (O32) | Synthetic |
| 2 | Does the onset move with a larger table, or with information that shifts within a run? (O29, carried) | Synthetic |
| 3 | Does a table suffice across veracity relations or domains? (O30, carried) | Synthetic |
| 3 | Do the comparisons hold when an arm is charged for the reach of its false shares? (O31, carried) | Synthetic |

## Reporting rules

These carry v7's five rules and add two.

- **LLM prior calibration.** Name an LLM prior's calibration with any result
  about it: the rule that set its location, and its steepness. For D5's
  prior, the utility cost inside the learner came from that calibration, not
  from the model's answers.
- **The cohort is not the world.** A prior matched to another's calibration
  on a design cohort need not share like it in the world. Report the world
  share counts of any recalibrated prior beside the comparisons that use it.

## Paper-1

**The central claim is unchanged.** NABM is a reusable framework that
separates policy learning from social interaction, so fixed policies,
adaptive policies, and external priors can be compared and traced within one
simulation lifecycle.

**What this consolidation changes:** the LLM-prior case can now say where
its cost comes from.
- An LLM prior adds nothing detectable inside the learner on precision, and
  it lowers the agents' utility.
- For D5's prior, the loss traces to how the prior was calibrated (D3b's
  shift to the base rate), not to the model's answers.
- The lifecycle made that attribution possible. The prior slot let one
  factor of a prior be swapped at a time, with every other part of the run
  held fixed.

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N23 | Paper | Update Section 8, the claim matrix, and the outline with C63–C67 and these rules | After this consolidation |
| N24 | Repository | Merge the O28 branch into `main` through a pull request | Confirmed by the user on 2026-10-23 |
| N25 | Comparison design | Choose among O29–O33, each as a separate precommit | The user's choice |
| N26 | Release | Tag a next alpha that ships the policy-ladder package surface (carried from N22) | The user's confirmation |

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical rows, and changes no generic NABM API. It selects no
model and establishes no neural necessity or scientific effect.

O28's own limits carry over:
- **Location and steepness moved together.**
- **The recalibration was matched on the cohort,** and it overshot in the
  world.
- **P0's Regime A direction is mostly the clipped readout's noise.**
- **Utility is the sharer's own.**
