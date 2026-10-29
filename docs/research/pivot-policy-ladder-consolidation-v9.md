# Pivot policy-layer ladder consolidation v9

Date: 2026-10-29

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v9.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v9.yaml)

Extends: [consolidation v8](pivot-policy-ladder-consolidation-v8.md), and
through it v1 to v7, which remain the records for everything before.

## Purpose

v8 found where the utility cost of an LLM prior inside the learner comes
from (C63, C64). It also found that the recalibration behind that finding,
matched on the design cohort, overshot in the world (C66). It ranked first
the question of whether the attribution survives a match in the world
(O33). The user chose O33 on 2026-10-24.

This document records what O33 established and which v8 claims it narrows.
Every claim cites a hash-bound result and a field that must hold, and a
test checks all of them. This document executes nothing.

| Study | Freeze / record | Outcome | Runs (development / reproduction) |
|---|---|---|---|
| O33 world calibration | `a926c25` / `423a942` | `O33_NEITHER_HOLDS` | 270 / 270 min, byte identical |

## The design in one paragraph

O33 repeated O28's 2 × 2 of direction (the model's answers) and calibration
against the hand rule F1, for each LLM prior (P0, and the unclipped P0u) and
each regime. Each recalibrated prior kept O28's weights. Only its intercept
changed: it was set on 24 calibration runs per regime, at world seeds
disjoint from the study cells, so that the prior shares as many unverified
items as its source. The three costed cases were again P0/A, P0/B, and
P0u/B.

## What is established

All results are synthetic. Ledger ids are in parentheses.

- **The match in the world holds out of sample (C72).**
  - On the calibration runs, every target was met within 0.02%.
  - On the study cells, each world-calibrated fixed prior shares 0.984 to
    1.001 times as many unverified items as its source. Under O28's cohort
    match the same ratios ran from 0.20 to 1.48.
  - Only the location was matched this way. The steepness stayed O28's.
- **In Regime A, D5's prior still costs utility through its calibration
  (C68).**
  - Giving the learner P0's calibration lowers utility along both routes,
    in 6 and 6 of 6 groups.
  - At P0's own calibration, its answers make no material difference in 6
    of 6 groups.
- **In Regime B, once the share rates match, the model's answers lower
  utility too.**
  - **D5's prior (C69).** Its answers lower utility along both routes (6
    and 5 of 6 groups), and its calibration still does (4 and 4 of 6). The
    case is `BOTH`, where O28 found `CALIBRATION`.
  - **The unclipped prior (C70).** Its answers lower utility along both
    routes (6 and 5 of 6), including at its own calibration, where O28 found
    them raising it in 4 of 6. Its calibration is mixed given F1's answers.
    The case is `ANSWERS`, where O28 found `NEITHER_CONSISTENTLY`.
  - **How, descriptively (not judged).** At matched calibration, a learner
    started from an LLM direction shares between 6% fewer and 0.3% more
    unverified items than one started from F1's direction, at an unverified
    precision from −0.013 to +0.001 against it.
- **In Regime A, where it carries no utility cost, the unclipped prior's
  answers help at its own calibration** (5 of 6 groups, C70).
- **Still no LLM-informed start beats the hand rule's start under utility
  (C71).**
  - Of the 8 prior and regime cases against L0, 5 are `MOSTLY_NEGATIVE` and
    3 are `MIXED_OR_NONE`. None is `MOSTLY_POSITIVE`.
  - The closest is F1's answers at the unclipped prior's Regime B
    calibration, positive in 3 of 6 groups.
- **O33 reproduced byte for byte (C73),** its world calibration included.
  L0 parity held in all 240 learner tasks, and every group's L1 − L0 and
  L1u − L0 equalled O25's record.

## What is narrowed or rejected

| v8 claim | Scope now |
|---|---|
| C63, D5's prior costs utility through its calibration | Holds in Regime A with share rates matched in the world (C68). In Regime B the model's answers lower utility too, and the case is `BOTH` (C69) |
| C64, the unclipped prior's answers and calibration interact | Holds only under O28's cohort match, which overshot in the world. With share rates matched in the world, its cost in Regime B is the model's answers (C70) |

| Claim | Why it does not hold |
|---|---|
| C63 holds with share rates matched in the world, O33's stated expectation (R21) | It holds in Regime A. In Regime B, D5's prior is `BOTH` |

**C61 is now explained with share rates matched in the world.** The cost of
D5's prior is its calibration in Regime A (C68), and both its calibration
and the model's answers in Regime B (C69). For the unclipped prior in
Regime B, it is the model's answers (C70).

## What is open

| Rank | Question | Kind |
|---|---|---|
| 1 | Do the Regime B attributions hold when the steepness, and not only the location, is matched in the world? (O34) | Synthetic |
| 2 | Is it the location (the base-rate shift) or the steepness that carries the cost of D5's prior? (O32, carried) | Synthetic |
| 2 | Does the onset move with a larger table, or with information that shifts within a run? (O29, carried) | Synthetic |
| 3 | Does a table suffice across veracity relations or domains? (O30, carried) | Synthetic |
| 3 | Do the comparisons hold when an arm is charged for the reach of its false shares? (O31, carried) | Synthetic |

## Reporting rules

These carry five of v8's rules and replace two.

- **LLM prior attribution by regime** (replaces "LLM prior calibration").
  Name an LLM prior's calibration with any result about it: the rule that
  set its location, and its steepness. Attribute its utility cost per
  regime. With share rates matched in the world, D5's prior costs utility
  through its calibration in Regime A, and through its calibration and the
  model's answers in Regime B. The unclipped prior in Regime B costs it
  through the model's answers.
- **World-matched attribution** (replaces "the cohort is not the world").
  Attribute a prior's cost to its parts only with each recalibrated prior
  matched to its source's share rate in the world, and report that match
  beside the comparisons. Under O28's cohort match, two of the three costed
  attributions differed. Only the location has been matched in the world;
  the steepness has not.

## Paper-1

**The central claim is unchanged.** NABM is a reusable framework that
separates policy learning from social interaction, so fixed policies,
adaptive policies, and external priors can be compared and traced within one
simulation lifecycle.

**What this consolidation changes:** the attribution of the LLM prior's cost
is narrower.
- An LLM prior adds nothing detectable inside the learner on precision, and
  it lowers the agents' utility.
- For D5's prior, the loss traces to its calibration in Regime A. In
  Regime B, once the share rates are matched in the world, the model's
  answers carry part of it too.
- The lifecycle made the correction possible as well as the attribution.
  The prior slot let the intercept be reset in the world with the
  direction, the steepness, and every other part of the run held fixed.

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N27 | Paper | Update Section 8, the claim matrix, and the outline with C68–C73 and these rules | After this consolidation |
| N28 | Repository | Merge the O33 branch into `main` through a pull request | Pull request confirmed by the user on 2026-10-29; merge on the user's request |
| N29 | Comparison design | Choose among O29–O32 and O34, each as a separate precommit | The user's choice |
| N30 | Release | Tag a next alpha that ships the policy-ladder package surface (carried from N26) | The user's confirmation |

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical rows, and changes no generic NABM API. It selects no
model and establishes no neural necessity or scientific effect.

O33's own limits carry over:
- **Only the location was matched in the world.** The steepness stayed
  O28's, matched on the cohort.
- **The match is on how many unverified items are shared,** not on which.
- **P0's Regime A direction is mostly the clipped readout's noise.**
- **Utility is the sharer's own.**
