# Pivot policy-layer ladder consolidation v10

Date: 2026-10-29

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v10.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v10.yaml)

Extends: [consolidation v9](pivot-policy-ladder-consolidation-v9.md), and
through it v1 to v8, which remain the records for everything before.

## Purpose

An audit of the evidence behind the ladder's claims, on 2026-10-29, found
three reasons to doubt C22:
- the calibrated LLM prior is far more selective than the hand rule;
- the v2 control matched its mean, but not its entry; and
- its design-cohort AUC equals the hand rule's in both interaction worlds.

The user chose to test this first. The entry-matched control asks whether the
prior selects better entrants than the hand rule at equal selectivity in the
world.

This document records what the control established and which claims it
narrows. Every claim cites a hash-bound result and a field that must hold, and
a test checks all of them. This document executes nothing.

| Study | Freeze / record | Outcome | Runs (development / reproduction) |
|---|---|---|---|
| Entry-matched control | `243dad7` / `bbdc3b0` | `C22_PRECISION_IS_SELECTIVITY` | 628 / 572 s, byte identical |

## The design in one paragraph

A prior has a direction, its feature weights, and a selectivity, the
fixed-rule entry it produces in the world. Each matched condition keeps one
direction's weights exactly and shifts only its intercept. The shift is
bisected on 16 calibration replicates until its entry equals the other
direction's native entry. The study then compares the directions at equal
selectivity. The comparison is made under the fixed rule, inside the pooled
learners, and with C22's 20% exploration floor. It uses v3's arms, worlds,
and decision rule on fresh seeds. Learning non-inferiority is judged by an
interval.

## What is established

All results are synthetic. Ledger ids are in parentheses.

- **One number matches selectivity in the world (C78).**
  - Fixed-rule entry does not depend on the world.
  - Shifting the hand rule's intercept by −0.28 matched the LLM prior's entry
    of 0.323.
  - On the study replicates, the matched hand rule admitted 0.959 times as
    many agents. That is inside the registered bounds and worth at most
    +0.003 of precision to it.
  - Every artifact reproduced byte for byte.
- **At equal selectivity, the LLM prior ranks better only in the linear world
  (C74).** Under the fixed rule it picks more productive entrants there, by
  +0.059 (+0.035 to +0.082). In the smooth and axis worlds there is no clear
  difference (+0.015 and −0.001). This is what the design-cohort AUC
  predicted.
- **C22's precision gain is selectivity (C75).**
  - The comparison is against a hand rule made equally selective, with the
    same 20% floor.
  - The hybrid selects better in none of the 9 cells and worse in one (linear
    world, MLP learner).
  - No cell is dominated.
- **Inside the pooled learners, the LLM direction adds nothing at either
  selectivity (C76).**
  - At the LLM's selectivity it is better in 1 of 9 cells and worse in 1.
  - At the hand rule's selectivity it is better in none of 12 cells and worse
    in 7, all of them learner cells.
- **C22's dominance count depends on the seed set and the rule (C77).**
  - On a third seed set, the hybrid still beats the native hand rule on
    precision in 9 of 9 cells.
  - It dominates in 4 of 9 by C22's mean-only rule, and in 2 of 9 by an
    interval rule.
  - Recomputed from v3's record, C22's own cells pass the interval rule in
    2 of 9.

## What is narrowed or rejected

| Earlier claim | Scope now |
|---|---|
| C20, the calibrated LLM prior selects better entrants | Against the native hand rule, which admits about 0.82 of agents against 0.32. At equal selectivity, only in the linear world under the fixed rule (C74). Inside the pooled learners, at neither selectivity (C76) |
| C22, an exploration floor lets the prior dominate the hand rule in 7 of 9 cells | Its precision gain is selectivity (C75). Its count is 4 of 9 by its own rule on a third seed set, and 2 of 9 by an interval rule (C77) |
| C29, the v3 hybrid replicates on fresh seeds | Only on the v4 seeds. On a third seed set, 4 of 9 by the mean-only rule and 2 of 9 by an interval rule (C77) |

| Claim | Why it does not hold |
|---|---|
| C22's precision survives in linear-world cells, the control's stated expectation E3 (R22) | It survives in no cell. The fixed-rule advantage in the linear world does not carry into the learners once the floor is added |

## What is open

| Rank | Question | Kind |
|---|---|---|
| 1 | Why does the pooled linear learner lose to the MLP in the linear world, where it is correctly specified? (O35, reopening O12) | Synthetic |
| 1 | Do the Regime B attributions hold with the steepness also matched in the world? (O34, carried) | Synthetic |
| 2 | Is it the location or the steepness that carries the cost of D5's prior? (O32, carried) | Synthetic |
| 2 | Does the onset move with a larger table, or with information that shifts within a run? (O29, carried) | Synthetic |
| 3 | Does a table suffice across veracity relations or domains? (O30, carried) | Synthetic |
| 3 | Do the comparisons hold when an arm is charged for the reach of its false shares? (O31, carried) | Synthetic |

O35 matters for every capacity comparison. Under the hand rule in the linear
world, the linear learner's probe Brier is 0.060, against the MLP's 0.013.

## Reporting rules

These carry v9's seven rules and add two.

- **Selectivity-matched precision.** Compare two priors on the precision of
  whom they admit only beside a comparison at matched selectivity in the
  world. A more selective prior raises precision without ranking better.
- **Interval non-inferiority.** Judge non-inferiority by the lower bound of an
  interval, not by the mean, and name the seed set.

## Paper-1

**The central claim is unchanged.** NABM is a reusable framework that
separates policy learning from social interaction, so fixed policies,
adaptive policies, and external priors can be compared and traced within one
simulation lifecycle.

**What this consolidation changes:** the LLM prior's benefit on entrant
precision is narrower.
- Most of what looked like knowledge was selectivity. One intercept shift of
  the hand rule reproduces it.
- The prior's ranking helps only in the linear world under the fixed rule.
- The lifecycle made the check cheap. The prior slot let the hand rule's
  intercept be reset in the world with its weights and every other part of the
  run held fixed.

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N31 | Paper | Rewrite the manuscript's LLM-prior passages with C74–C78, in the manuscript kept outside the repository | After this consolidation |
| N32 | Comparison design | Freeze a study of O35 as a separate precommit | Confirmed by the user on 2026-10-29 |
| N33 | Repository | Merge the entry-matched branch into `main` through a pull request | The user's confirmation |
| N34 | Comparison design | Choose among O29–O32 and O34, each as a separate precommit (carried from N29) | The user's choice |

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical rows, and changes no generic NABM API. It selects no
model and establishes no neural necessity or scientific effect.

The control's own limits carry over:
- **One prior, from one model family, elicited once.**
- **Selectivity is matched on how many agents enter,** not on which.
- **The worlds are synthetic.** The result says nothing about LLM knowledge
  of real pivots.
