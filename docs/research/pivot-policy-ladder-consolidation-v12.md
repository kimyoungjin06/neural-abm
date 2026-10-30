# Pivot policy-layer ladder consolidation v12

Date: 2026-10-30

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v12.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v12.yaml)

Extends: [consolidation v11](pivot-policy-ladder-consolidation-v11.md), and
through it v1 to v10, which remain the records for everything before.

## Purpose

v11 opened O36. The capacity conclusions C07 to C14 were drawn with every
residual learner trained by 300 Adam steps. Do they hold when every residual
objective is converged, and when each fit's training is chosen on its own
evidence? The user chose the study on 2026-10-29.

This document records what the comparison established. Every claim cites a
hash-bound result and a field that must hold, and a test checks all of them.
This document executes nothing.

| Study | Freeze / record | Outcome | Runs (development / reproduction) |
|---|---|---|---|
| Converged capacity comparison v1 | `4062c8c` / none | Stopped at its convergence check; no outputs | — |
| Converged capacity comparison v2 | `1cfd1ab` / `ef712e7` | `O36_HOLD_UNDER_BOTH` | 4,610 / 4,627 s, byte identical |

## The design in one paragraph

The matched-marginal discriminator v3 and the nonlinear policy ABM v2 were
rerun on their own seeds, with their own decision code. Only the training
changed. Under `registered`, the recorded training reproduced both records
row for row. Under `converged`, L-BFGS minimized every residual objective to a
maximum absolute gradient of 1e-6. Under `tuned`, each fit chose its training
on its own evidence, by v3's inner folds. The menu crossed 300 Adam steps and
convergence with five weight decays, and the ABM tree chose among v3's 12 tree
settings.

**v1 stopped.** It required every converged fit to reach the tolerance, and
one inner fit of a tuned field MLP did not. v1 wrote no output and computed no
converged or tuned outcome. v2 changed only how such a fit is handled:
- an inner fit that misses makes its candidate unavailable for that selection;
- an acting fit that misses keeps its final iterate and counts against a 0.1%
  ceiling.

In v2, one inner fit of 63,777 missed, and every acting fit converged.

## What is established

All results are synthetic. Ledger ids are in parentheses.

- **The capacity conclusions hold under both retrained regimes (C83).** Every
  check of C07 to C14 holds as recorded, and R01 stays rejected.
- **The linear world's MLP edge was the linear learner's training (C84).**
  With field evidence and immediate visibility, the edge is 0.046 registered,
  0.008 converged, and −0.001 tuned. The policy layer's gap falls from 0.028
  and 0.038 to about zero.
- **The registered budget regularizes the MLP (C85).**
  - Converged at the registered penalty, the MLP is 0.004 worse in the linear
    world and 0.013 worse in the axis world.
  - Tuned, it chose 300 Adam steps in 62% of its linear-world selections.
  - The same budget handicaps the linear learner.
- **The interaction-world comparisons do not depend on training (C86).**
  - The crossover holds in 16 of 16 replicates under every regime.
  - The DiD margins grow, because they subtract a smaller linear-world gap.
    The MLP's field-learning DiD in the smooth world rises from 0.046 to
    0.094.
- **The MLP's behavior gains are interaction-specific too (C87).**
  - Registered, only the tree passed the entry and exit DiDs.
  - Tuned, the MLP also passes the entry DiD in both interaction worlds, and
    the exit DiD in the smooth one.
  - In the linear world the tuned linear learner's entrant precision and
    failed stock match the MLP's.
- **The comparison reproduced byte for byte (C88).** Every acting fit
  converged, and the tuned neighbor-learner rows equal the registered ones.

## What is rejected

Nothing. All four expectations stated before the run held.

No earlier claim is narrowed. v2's statement that only the tree's behavior
gains were interaction-specific was never a ledger claim; C87 records that it
came from the linear learner's training.

## What is open

| Rank | Question | Kind |
|---|---|---|
| 1 | Do the Regime B attributions hold with the steepness also matched in the world? (O34, carried) | Synthetic |
| 2 | Does the LLM direction still add nothing inside the pooled learners (C76) when their training is tuned? (O37) | Synthetic |
| 2 | Is it the location or the steepness that carries the cost of D5's prior? (O32, carried) | Synthetic |
| 2 | Does the onset move with a larger table, or with information that shifts within a run? (O29, carried) | Synthetic |
| 3 | Does a table suffice across veracity relations or domains? (O30, carried) | Synthetic |
| 3 | Do the comparisons hold when an arm is charged for the reach of its false shares? (O31, carried) | Synthetic |

## Reporting rules

These carry v11's rules, with one replaced.

- **Tuned capacity comparison** (replaces the converged capacity comparison).
  Compare policy classes with each class's training chosen on its own
  evidence from one shared menu. Otherwise, report beside each DiD the
  linear-world gap it subtracts and how far each learner is from its optimum.

## Paper-1

**The central claim is unchanged.** NABM is a reusable framework that
separates policy learning from social interaction, so fixed policies,
adaptive policies, and external priors can be compared and traced within one
simulation lifecycle.

**What this consolidation changes:** the capacity picture no longer depends
on one training budget.
- Every capacity conclusion holds with converged and with tuned training.
- With tuned training, the linear learner matches the MLP where the boundary
  is linear, and the MLP's entry gains are interaction-specific in both
  geometries.
- The learner slot let every family's training change with the worlds, the
  lifecycle, and the decision code held fixed.

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N38 | Paper | Rewrite the manuscript's capacity passages with C83–C88, in the manuscript kept outside the repository | After this consolidation |
| N39 | Repository | Merge the converged capacity branch into `main` through a pull request | The user's confirmation |
| N40 | Comparison design | Choose among O29–O32, O34, and O37, each as a separate precommit | The user's choice |

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical rows, and changes no generic NABM API. It selects no
model and establishes no neural necessity or scientific effect.

The comparison's own limits carry over:
- **The hand rule is the only prior.**
- **Three trainings, not a search.** The regimes are three ways to train
  these learners, from one menu.
- **Neighbor learners keep the registered setting under tuning.** With at most
  four events, they cannot be cross-validated.
