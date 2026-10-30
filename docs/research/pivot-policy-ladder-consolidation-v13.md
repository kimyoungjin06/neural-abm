# Pivot policy-layer ladder consolidation v13

Date: 2026-10-30

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v13.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v13.yaml)

Extends: [consolidation v12](pivot-policy-ladder-consolidation-v12.md), and
through it v1 to v11, which remain the records for everything before.

## Purpose

The user directed recommended experiment 4 on 2026-10-30: Study 2 against
real data. Study 2 postulates that researchers learn from their neighbors'
visible pivot outcomes. Does an author's entry into a surge topic depend on
a coauthor's visible outcome there? This is the first study to read outcome
rows from OpenAlex, and it reads the discovery partition only.

This document records what the study established. Every claim cites a
hash-bound result and a field that must hold, and a test checks all of them.
This document executes nothing.

| Study | Freeze / record | Outcome | Runs (development / reproduction) |
|---|---|---|---|
| Peer-outcome contrast v1 | `586f33b` / `e037f26` | `E4_NOT_INTERPRETABLE_THE_ESTIMATOR_FLAGS_STUDY2_FROZEN_ARM` | 1,384 / 1,487 s, byte identical |

## The design in one paragraph

One discrete-time hazard estimator was fitted twice.
- **In Study 2:** the code ran unchanged through two slots. Outcomes became
  visible to the agents either at once or after three steps.
- **In the 24 discovery episodes:** neighbors are coauthors, and the outcome
  is N11a v3's analyst label, a follow-up topic work within four quarters.

Each neighbor entrant is in one state: visible success, visible failure, or
pending with its eventual value. The visibility DiD compares the outcome gap
after visibility with the gap before it.

The decision rule had a precondition. The estimator must leave Study 2's
frozen arm flat, because that arm never learns. If it did not, the data DiD
would not be read.

## What is established

- **The estimator recovers Study 2's imitative arm (C89).**
  - Entry rises after a visible success, by +5.2 to +6.7 log-odds, and falls
    after a visible failure, by −4.3 to −4.7.
  - This holds in every scenario.
  - The cautionary arm never pivots after a visible failure.
- **The visibility placebo is biased without learning (C90).**
  - In the seed-grant world, Study 2's frozen arm shows a visibility DiD of
    −0.785 [−1.156, −0.371].
  - Its outcome gap is positive while outcomes are pending and negative once
    they are visible, so this association changes over time.
  - By the registered rule, the data DiD (−0.062 [−0.492, +0.538]) is not
    interpreted.
- **A visible coauthor entry goes with more entry (C91).**
  - Without recruitment: +0.520 [+0.367, +0.739] log-odds.
  - With recruited entries counted: +1.078.
  - These are associations, not effects.
- **Discovery entrants rarely follow up (C92).**
  - Of 6,291 precise entrants, 14.6% publish again in the topic within a
    year.
  - The follow-up rate does not change with entry quarter.
- **Post hoc: the data bound outcome dependence far below Study 2's
  imitative arm (C93).** Under the same estimator:
  - the imitative arm's visible outcome gap is at least 6.95 log-odds;
  - the data's upper bound is 0.79, and the frozen arm's largest bias is
    0.79;
  - Study 2's imitative arm also shows a falling entry-order gradient, and
    the data show none.
- **The study reproduced byte for byte with full parity (C94).**
  - Study 2's 270 tracked runs were reproduced.
  - The rows of all 177,254 exposed authors equal N11b's.
  - No confirmatory or G4 outcome row was read.

## What is rejected

Three of the four expectations stated before the run failed.

- **ES2 (R24).** It held in the seed-grant world, at +7.03. In the baseline
  world, the frozen arm had 9 entries after a visible failure, one below the
  registered minimum, so the check is undefined there.
- **ES3 (R25).** The estimator is not flat in Study 2's no-learning arm; see
  C90.
- **EE2 (R26).** Homophily before visibility was not detected: +0.290
  [−0.553, +0.583], a wide interval.

No earlier claim is narrowed. Study 2's learning rules were always stated as
assumptions.

## What is open

| Rank | Question | Kind |
|---|---|---|
| 1 | Do the Regime B attributions hold with the steepness also matched in the world? (O34, carried) | Synthetic |
| 2 | Is there a coauthor-outcome design that leaves Study 2's frozen arm flat and has power at the empirical shape? (O38) | Synthetic, then empirical |
| 2 | Does Study 2's imitative loop survive at a learning strength the data allow, with delayed visibility? (O39) | Synthetic |
| 2 | Does the LLM direction still add nothing inside tuned learners? (O37, carried) | Synthetic |
| 2 | Is it the location or the steepness that carries the cost of D5's prior? (O32, carried) | Synthetic |
| 2 | Does the onset move with a larger table, or with information that shifts within a run? (O29, carried) | Synthetic |
| 3 | Does a table suffice across veracity relations or domains? (O30, carried) | Synthetic |
| 3 | Do the comparisons hold when an arm is charged for the reach of its false shares? (O31, carried) | Synthetic |

**Data state.**
- The discovery partition's follow-up labels have now been read.
- The confirmatory and G4 outcomes stay sealed.
- O38 would need a design checked on Study 2's frozen arm before any
  confirmatory reading.

## Reporting rules

These carry v12's rules and add one.

- **Generator-first placebo.** Read an observational learning-direction
  estimate from data only after the same estimator, at the same delay, leaves
  a generator's no-learning arm flat. The before-and-after placebo showed
  −0.79 log-odds in Study 2's frozen seed-grant world, where no agent learns
  (C90).

## Paper-1

**The central claim is unchanged.** NABM is a reusable framework that
separates policy learning from social interaction, so fixed policies,
adaptive policies, and external priors can be compared and traced within one
simulation lifecycle.

**What this consolidation changes:**
- Study 2's learning rules remain postulated. The first check against
  coauthor data could not read learning direction.
- The data bound Study 2's imitative strength far below what it assumes
  (post hoc).
- The simulation lifecycle tested an empirical estimator on known truth, and
  the registered rule withheld the data reading when that test failed.

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N41 | Paper | Add C89–C94 to the manuscript kept outside the repository | After this consolidation |
| N42 | Repository | Merge the peer-outcome contrast branch into `main` through a pull request | The user's confirmation |
| N43 | Comparison design | Choose among O29–O32, O34, O37, O38, and O39, each as a separate precommit | The user's choice |

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical row, and changes no generic NABM API. It selects no
model and establishes no neural necessity, scientific effect, or learning in
real researchers.

The study's own limits carry over:
- **One partition.** Its data are the discovery partition only.
- **Associations.** Its data estimates are associations.
- **One scale, two designs.** Its comparisons with Study 2 put agent steps
  and topic quarters on one log-odds scale.
