# Pivot policy-layer ladder consolidation v14

Date: 2026-10-30

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v14.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v14.yaml)

Extends: [consolidation v13](pivot-policy-ladder-consolidation-v13.md), and
through it v1 to v12, which remain the records for everything before.

## Purpose

The user directed O39 on 2026-10-30. Study 2's imitative arm raises failed
pivots against the frozen arm by +0.281 at baseline and +0.289 under seed
grants, which the case study calls a self-reinforcing loop (R5). Under
experiment 4's estimator, that arm answers a neighbor's outcome far more
strongly than the coauthor data allow (C93). Does the loop survive at a
learning strength the data allow, and with delayed visibility?

This document records what the study established. Every claim cites a
hash-bound result and a field that must hold, and a test checks all of them.
This document executes nothing.

| Study | Freeze / record | Outcome | Runs (development / reproduction) |
|---|---|---|---|
| Learning strength v1 | `42b682e` / `da5f772` | `O39_THE_LOOP_FALLS_BELOW_MATERIAL_AT_THE_DATA_ALLOWED_STRENGTH` | 172 / 224 s, byte identical |

## The design in one paragraph

Study 2 ran unchanged through experiment 4's slots at eight learning rates:
the tracked rate 1.5 halved seven times. It ran at outcome delays 0 and 3.
Experiment 4's estimator, imported by hash, measured the imitative arm at
every rate.

- **The data-allowed rate.** The largest grid rate at or below which the
  imitative arm's visible outcome gap stays within the data's upper bound,
  0.789, in every scenario where it is defined.
- **The outcome.** R5's own comparison, read at that rate: failed pivots
  against the frozen arm, paired by replicate. A material size of 0.05 was
  registered.

## What is established

- **At the data-allowed rate, the loop falls below material (C95).**
  - With the delay: +0.0003 [+0.0000, +0.0008] at baseline and +0.0161
    [+0.0117, +0.0208] under grants.
  - At once: +0.0003 and +0.0222.
  - The grant difference is still detected.
- **That rate is one sixty-fourth of Study 2's (C96).**
  - The data-allowed rate is 0.0234375 at both delays.
  - The imitative gap falls with the rate from +7.0 to +11.2 log-odds at the
    tracked rate.
  - With the data's point estimate as the bound, the delayed calibration
    allows no grid rate.
- **Delay alone changes the loop unevenly (C97).** At the tracked rate:
  - the delay halves the baseline loop, to +0.138;
  - it leaves the grant loop at +0.301;
  - it turns the hype difference positive, +0.0506.
- **Failure-only learning shrinks too (C98).**
  - Its cut in hype failures (R6) is −0.1436 at the tracked rate and
    −0.0583 with the delay.
  - At the data-allowed rate it is −0.0200 at once and −0.0119 with the
    delay.
  - The cautionary arm cannot be calibrated itself: it never pivots after a
    visible failure.
- **Post hoc: a material loop needs 4 to 32 times that rate (C99).**
  - Under grants the loop stays material down to four times the data-allowed
    rate. At baseline it stays material down to 16 times at once and 32
    times with the delay.
  - At the data-allowed rate, learning adds more productive than failed
    grant pivots.
  - The entry-order gradient there equals the frozen arm's, so Study 2's
    falling gradient comes from its structure, not its learning.
- **The study reproduced Study 2 and experiment 4 exactly (C100).**
  - At the tracked rate it reproduced all 270 runs and all 12 comparisons.
  - It also reproduced all 378 compared estimator values.
  - It read no empirical row.

## What is narrowed

No ledger claim is narrowed. Two results of the case study are:

- **R5, the imitative loop,** holds at Study 2's hand-set learning rate. At
  the data-allowed rate the loop is below material (C95).
- **R6, failure-only learning against hype failures,** holds at the
  hand-set rate with immediate visibility. With the delay it is about two
  fifths as large, and at the data-allowed rate it is −0.012 to −0.020
  (C98).

Nothing is rejected. All four expectations stated before the run held.

## What is open

| Rank | Question | Kind |
|---|---|---|
| 1 | Do the Regime B attributions hold with the steepness also matched in the world? (O34, carried) | Synthetic |
| 2 | Is there a coauthor-outcome design that leaves Study 2's frozen arm flat and has power at the empirical shape? (O38, carried) | Synthetic, then empirical |
| 2 | Does the LLM direction still add nothing inside tuned learners? (O37, carried) | Synthetic |
| 2 | Is it the location or the steepness that carries the cost of D5's prior? (O32, carried) | Synthetic |
| 2 | Does the onset move with a larger table, or with information that shifts within a run? (O29, carried) | Synthetic |
| 3 | Does R5's narrowing hold under a calibration that does not put agent steps and topic quarters on one scale? (O40) | Synthetic |
| 3 | Does a table suffice across veracity relations or domains? (O30, carried) | Synthetic |
| 3 | Do the comparisons hold when an arm is charged for the reach of its false shares? (O31, carried) | Synthetic |

O39 is answered by C95 to C97.

**Data state.** No data row was read by this study. The confirmatory and G4
outcomes stay sealed.

## Reporting rules

These carry v13's rules and add one.

- **Learning strength with Study 2.** State Study 2's learning results with
  the learning rate they need. At the rate whose outcome dependence the
  coauthor data do not exclude under experiment 4's estimator, the imitative
  loop is below material (C95). A material loop needs 4 to 32 times that
  rate (C99).

## Paper-1

**The central claim is unchanged.** NABM is a reusable framework that
separates policy learning from social interaction, so fixed policies,
adaptive policies, and external priors can be compared and traced within one
simulation lifecycle.

**What this consolidation changes:**
- Study 2's imitative loop needs learning far stronger than the coauthor
  data allow, under one estimator and one log-odds scale.
- The lifecycle carried a data bound back into the generator through the
  same estimator that produced it.

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N44 | Paper | Add C95–C100 and narrow R5 and R6 in the manuscript kept outside the repository | After this consolidation |
| N45 | Repository | Merge the learning-strength branch into `main` through a pull request | The user's confirmation |
| N46 | Comparison design | Choose among O29–O32, O34, O37, O38, and O40, each as a separate precommit | The user's choice |

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical row, and changes no generic NABM API. It selects no
model and establishes no neural necessity, scientific effect, or learning in
real researchers.

The study's own limits carry over:
- **Synthetic.** One recorded number from the discovery partition sets the
  bound.
- **One scale.** The calibration puts Study 2's agent steps and OpenAlex
  topic quarters on one log-odds scale, and takes the data bound at face
  value.
- **Not a calibration of Study 2.** Only the learning rate moves; the other
  hand-set constants stay as tracked.
