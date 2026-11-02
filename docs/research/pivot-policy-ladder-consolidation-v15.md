# Pivot policy-layer ladder consolidation v15

Date: 2026-10-30

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v15.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v15.yaml)

Extends: [consolidation v14](pivot-policy-ladder-consolidation-v14.md), and
through it v1 to v13, which remain the records for everything before.

## Purpose

The user directed recommended experiment 3 on 2026-10-30 and chose to run
it in two stages: synthetic first, then OpenAlex discovery episodes. This
consolidation records stage 1.

In the information-spread ladder, a table written before the run loses to
the in-run learner once run-specific information is common enough (O26).
Can that crossing be predicted before the comparison runs, from two
measurable quantities?
- **Between-run heterogeneity:** how different runs are.
- **Within-run evidence:** how much evidence a run gives its learner.

Every claim below cites a hash-bound result and a field that must hold, and
a test checks all of them. This document executes nothing.

| Study | Freeze / record | Outcome | Runs (development / reproduction) |
|---|---|---|---|
| Crossover prediction v1 | `871ace3` / `9144929` | `E3_THE_RULE_BEATS_THE_BASELINES_BUT_MISSES_BY_MORE_THAN_THE_BOUND` | 3,475 / 3,509 s, byte identical |

## The design in one paragraph

**The rule.** It compares the expected squared error of a k-donor table
with the in-run learner's, against the generator's truth.
- **Inputs.** Its components come from learner runs with one fixed sign, on
  worlds the comparison does not use.
- **Simple reading.** With no learner bias, it says the learner wins when
  p(1 − p) · Dd / V > (k − 1)/(k + 1):
  - Dd is the between-run heterogeneity;
  - V is the within-run variance;
  - p is the flip rate.
- **No fitted parameter.**

**The comparison.** O26's Regime A exact-linear design at:
- visibilities 0.2, 0.5, and 1.0;
- nine flip rates from 0 to 0.5;
- tables of 1, 2, 4, and 8 donors.

**The scoring.** Nine series (k = 2, 4, 8) were scored by the error in the
crossing, against a constant baseline and a table-size-only baseline.

## What is established

- **The rule beats both baselines but misses the bound (C101).**
  - Its mean crossing error is 0.0549, against 0.1052 and 0.0987.
  - It is within 0.05 in 6 of 9 series; each baseline is within 0.05 in 2.
  - The registered bound was 0.05.
- **The crossing moves later with table size and earlier with evidence
  (C102).**
  - Within-run variance fell in proportion to the visibility (0.032, 0.013,
    0.0067), while heterogeneity changed little.
  - The learner's first material advantage came at flip rates from 0.05 to
    0.25.
  - At visibility 0.5 it came at 0.05, 0.05, 0.1, and 0.15 for 1, 2, 4, and
    8 donors. This answers the table-size half of O29.
- **Without heterogeneity, a larger table wins and one donor ties (C103).**
  This is the rule's own limit.
- **Post hoc: the rule misses only where evidence is scarce (C104).**
  - At visibility 1.0 it is within 0.010 of every crossing.
  - At 0.5 it is late by at most 0.030.
  - At 0.2 it is late by 0.062 to 0.236.
- **O26's four-donor comparison replicates on new worlds (C105).**
  - Here: −0.015, +0.017, and +0.067 at flip rates 0, 0.1, and 0.25.
  - O26: −0.017, +0.010, and +0.069.
  - The onset reads 0.1 here and 0.25 in O26, because only this interval
    at 0.1 excludes zero.
- **The study reproduced byte for byte on one host (C106).**

## What is rejected

- **X4 (R27).** The rule does not meet the registered accuracy bound.

The other four expectations held. No earlier claim is narrowed.

## What is open

| Rank | Question | Kind |
|---|---|---|
| 1 | Do the Regime B attributions hold with the steepness also matched in the world? (O34, carried) | Synthetic |
| 2 | Does a rule that weights errors near the share cutoff place the crossing within the bound where evidence is scarce? (O41) | Synthetic |
| 2 | Does the onset move with information that shifts within a run? (O29, its table-size half answered by C102) | Synthetic |
| 2 | Is there a coauthor-outcome design that leaves Study 2's frozen arm flat and has power at the empirical shape? (O38, carried) | Synthetic, then empirical |
| 2 | Does the LLM direction still add nothing inside tuned learners? (O37, carried) | Synthetic |
| 2 | Is it the location or the steepness that carries the cost of D5's prior? (O32, carried) | Synthetic |
| 3 | Applied to OpenAlex discovery episodes, does the rule order a pooled table against a within-episode learner as the forecasts do? (O42, stage 2) | Empirical, discovery partition |
| 3 | Does R5's narrowing hold under a calibration that does not put agent steps and topic quarters on one scale? (O40, carried) | Synthetic |
| 3 | Does a table suffice across veracity relations or domains? (O30, carried) | Synthetic |
| 3 | Do the comparisons hold when an arm is charged for the reach of its false shares? (O31, carried) | Synthetic |

**Stage 2 (O42).**
- The episodes give few labels, each visible five quarters after entry.
- That is the regime where stage 1 found the rule late. Its premise is
  therefore only partly met.
- O41 would test a decision-weighted rule first.

**Data state.** No data row was read by this study. The confirmatory and G4
outcomes stay sealed.

## Reporting rules

These carry v14's rules and add one.

- **Crossover with table size and evidence.** State where a table written
  before the run loses to the in-run learner together with the table's size
  and the within-run evidence. Across 1 to 8 donors and visibility 0.2 to
  1.0, the first material loss came at flip rates from 0.05 to 0.25 (C102).

## Paper-1

**The central claim is unchanged.** NABM is a reusable framework that
separates policy learning from social interaction, so fixed policies,
adaptive policies, and external priors can be compared and traced within one
simulation lifecycle.

**What this consolidation changes:**
- Where the closed loop matters is not one number. It moves with the
  table's size and the run's evidence.
- Between-run heterogeneity and within-run evidence, measured before the
  comparison, order the crossing.
- A rule built from those two quantities places the crossing well where
  evidence is ample, and too late where it is scarce.

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N47 | Paper | Add C101–C106 to the manuscript kept outside the repository | After this consolidation |
| N48 | Repository | Merge the crossover-prediction branch into `main` through a pull request | The user's confirmation |
| N49 | Comparison design | Choose between stage 2 on OpenAlex (O42), a decision-weighted rule (O41), and the other open items, each as a separate precommit | The user's choice |

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical row, and changes no generic NABM API. It selects no
model and establishes no neural necessity or scientific effect.

The study's own limits carry over:
- **One generator:** one veracity relation, Regime A, and the exact linear
  learner.
- **One kind of run-specific information:** a sign drawn once per run.
- **A proxy:** the rule's squared-error proxy is scored against share
  precision on unverified items.
