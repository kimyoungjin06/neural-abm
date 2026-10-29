# Pivot policy-layer ladder consolidation v11

Date: 2026-10-29

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v11.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v11.yaml)

Extends: [consolidation v10](pivot-policy-ladder-consolidation-v10.md), and
through it v1 to v9, which remain the records for everything before.

## Purpose

v10 reopened an anomaly from v2 as O35. In the linear world, the productive
label is a deterministic linear threshold, so the pooled linear learner is
correctly specified. Yet it forecasts worse than the MLP: under the hand rule
its probe Brier is 0.060, against 0.013. The user confirmed the study on
2026-10-29.

This document records what the diagnosis established. Every claim cites a
hash-bound result and a field that must hold, and a test checks all of them.
This document executes nothing.

| Study | Freeze / record | Outcome | Runs (development / reproduction) |
|---|---|---|---|
| Linear-learner diagnosis | `8c7ffd8` / `0d01c3c` | `O35_LEARNER_PARTLY_OPTIMIZATION` | 113 / 108 s, byte identical |

## The design in one paragraph

The hand rule filled the prior slot. The linear and MLP arms ran once each per
replicate and world. A probed-learner subclass that only lists itself filled
the learner slot, so each fold's final training buffer and last-refit seed
were recorded. Every buffer was then refitted offline and scored on the probe
cohort. The refits were:
- the registered linear and MLP fits;
- the linear fit with 10x epochs, or with 10x learning rate;
- the registered linear objective minimized to convergence; and
- the same at a 100x weaker penalty, reported descriptively.

Fitting each learner on the other arm's evidence separates learner from
evidence.

## What is established

All results are synthetic. Ledger ids are in parentheses.

- **In the linear world, the gap is the learner, not the evidence (C79).** On
  the same evidence, the registered linear learner is 0.046 and 0.053 worse
  than the MLP. Swapping the evidence makes no material difference.
- **Most of that gap is the Adam budget, and the rest is the penalty (C80).**
  - Converging the registered linear objective closes 71% of the gap. 10x the
    Adam steps gets almost all of the way there.
  - The converged fit is still 0.013 worse than the MLP, outside the
    registered 0.005 margin.
  - Descriptively, at a 100x weaker penalty the converged fit matches the MLP.
    The MLP reaches a steep slope through products of smaller parameters,
    which the same L2 penalty charges less.
  - The MLP's ranking advantage is only 0.007 in AUC.
- **In the interaction worlds, the MLP's edge is its policy class (C81).**
  Converging the linear learner changes its Brier by about 0.001 there. The
  MLP's AUC is 0.18 and 0.19 higher.
- **The diagnosis reproduced byte for byte, with exact refit parity (C82).**

## What is rejected

| Claim | Why it does not hold |
|---|---|
| The linear learner's gap is its optimization budget alone, the stated expectation E2 (R23) | Converging closes 71% of it. The rest comes from the registered weight decay |

No earlier claim is narrowed. C08 to C11 rest on the interaction worlds,
where the gap is the policy class. Their size under converged fits is open
(O36).

## What is open

| Rank | Question | Kind |
|---|---|---|
| 1 | Do the Regime B attributions hold with the steepness also matched in the world? (O34, carried) | Synthetic |
| 2 | Do the capacity conclusions hold with every learner converged, or with the penalty set per parameterization? (O36) | Synthetic |
| 2 | Is it the location or the steepness that carries the cost of D5's prior? (O32, carried) | Synthetic |
| 2 | Does the onset move with a larger table, or with information that shifts within a run? (O29, carried) | Synthetic |
| 3 | Does a table suffice across veracity relations or domains? (O30, carried) | Synthetic |
| 3 | Do the comparisons hold when an arm is charged for the reach of its false shares? (O31, carried) | Synthetic |

## Reporting rules

These carry v10's nine rules and add one.

- **Converged capacity comparison.** Name each learner's training budget with
  any comparison of policy classes, and report how far its registered
  objective is from convergence.

## Paper-1

**The central claim is unchanged.** NABM is a reusable framework that
separates policy learning from social interaction, so fixed policies,
adaptive policies, and external priors can be compared and traced within one
simulation lifecycle.

**What this consolidation changes:** the capacity picture is more precise.
- In the linear world, the MLP's edge over the linear learner is training,
  not representation.
- In the interaction worlds, it is the policy class.
- The learner slot let each fold's evidence be recorded and refitted with
  everything else held fixed.

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N35 | Paper | Rewrite the manuscript's LLM-prior and capacity passages with C74–C82, in the manuscript kept outside the repository (carried from N31) | After this consolidation |
| N36 | Repository | Merge the entry-matched and diagnosis branch into `main` through a pull request (carried from N33) | The user's confirmation |
| N37 | Comparison design | Choose among O29–O32, O34, and O36, each as a separate precommit (carried from N34) | The user's choice |

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical rows, and changes no generic NABM API. It selects no
model and establishes no neural necessity or scientific effect.

The diagnosis's own limits carry over:
- **The hand rule is the only prior.**
- **The refits are the registered ones and five linear variants.**
- **The weak penalty and the 10x variants are descriptive.**
