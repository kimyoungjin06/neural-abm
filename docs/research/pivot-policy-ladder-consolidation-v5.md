# Pivot policy-layer ladder consolidation v5

Date: 2026-10-13

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v5.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v5.yaml)

Extends: [consolidation v4](pivot-policy-ladder-consolidation-v4.md), and through
it v1 to v3, which remain the records for everything before.

## Purpose

v4 carried a one-sided indicator to the empirical stage. Since then the
empirical stage moved to a model ladder and was tested at the measured shape
of the data. This document records what that stage established, closes its
current design before confirmatory scoring, and fixes the reporting rules and
the Paper-1 direction set on 2026-10-13.

As before, every claim cites a hash-bound result and a field that must hold,
and a test checks all of them. Nothing new is executed, and no LLM is called.

## The empirical stage since v4

| Step | Question | Freeze / record | Outcome |
|---|---|---|---|
| Threshold transfer (N10) | Does the indicator's threshold transfer to a quarterly data clock? | `58c87f4` / `e10308a` | Yes, on twelve entry levels |
| S1 structural build | Can the episodes and entrants be built without outcomes? | `52dcf76` / `b603a4d` | Yes. 63 episodes, 19,132 retained entrants, byte identical |
| Ladder recovery v1 (N11a) | Does a fair model ladder recover learning against confounds? | `af43b16` / `d114272` | Yes, but anticipation also wins 8 of 8 |
| Ladder recovery v2 | With an anticipatory competitor B7? | `0cbe5f2` / `886be72` | Yes, at the S1-scale synthetic shape |
| Risk sets (N11b) | What shape does the real data have? | `8a375d8` / `72476b3` | Built without outcomes, byte identical |
| Ladder recovery v3 | Does recovery hold at that shape? | `288062a` / `bca896d` | No. Underpowered, no false recovery observed |
| Power-lever screen | Can a change of test, evidence, or scale restore it? | exploratory `d29752c` / `42d6b21` | No lever achieved power and false-recovery control together |

The model ladder replaced the indicator as the empirical test. In the v1
worlds, the indicator missed pure learning in 8 of 8 replicates. It turned
positive only when topic growth accompanied learning (R12).

## What is established

All results are synthetic, except the two structural builds. The ledger ids
are in parentheses.

- **Builds without outcomes.**
  - S1 rebuilt the episodes from the 2026-06-26 snapshot (C43).
  - N11b built the risk sets, entry events, and pre-entry features for the
    confirmatory window (C45).
  - Neither computed a persistence label, a follow-up count, or visible
    evidence.
- **Identification at an optimistic shape.** At the S1-scale synthetic shape
  (600 at risk, about 200 entrants per episode, exact dates), the ladder with
  B7 recovered learning in 8 of 8 replicates. It won 0 of 8 in the
  anticipation world (C44).
- **No false recovery observed at the N11b shape.**
  - The registered v3 analysis won in none of 8 replicates in each of the five
    no-learning worlds (C46).
  - With 8 replicates, the one-sided 95% upper bound on a world's
    false-recovery rate is about 31%. A zero rate is not established.

## What was rejected

| Claim | Why it does not hold |
|---|---|
| The outcome selection trend is an adequate empirical test of learning (R12) | It missed pure learning (0 of 8) and was positive only with growth (8 of 8) |
| The registered v3 analysis detects learning at the N11b shape (R13) | W1 1 of 8 and the confounded W2 3 of 8, against a required 7 of 8 |
| A joint-model test of the learned coefficient separates learning from anticipation (R14) | Exploratory: it flagged the anticipation world in 8 of 8 replicates, and 7 of 8 with a ramp |
| Field history or lifting the G4 guard restores power (R15) | Exploratory: the registered rule recovered W1 in 3 and 2 of 8. Field history added false wins in the static world (2 of 8) |

R14 is a counterexample for that test. The anticipatory term is a proxy
estimated from final observed labels. So the result does not show that an
exactly specified anticipation model would also fail.

## Closing the current N11 design

> In synthetic worlds that reflect the data structure measured in N11b, the
> registered analysis did not meet the power criterion needed to distinguish
> learning from anticipation. The explored alternatives did not achieve power
> and false-recovery control together, so the confirmatory analysis of the
> current design does not proceed. This does not imply the absence of
> learning.

N11c and N11d are not proposed. The closure does not say any of the following:

- **Not that learning is absent.** No confirmatory outcome was read.
- **Not that no method could separate learning from anticipation in this
  data.** Only the registered analysis and the screened alternatives failed.
- **Not that v3 reproduces the empirical structure completely.**
  - v3's median analyst evidence pool in W1 is 6, 40, and about 58 at
    quarters 5, 8, and 11.
  - N11b's median count of precise confirmatory entrants old enough for
    evidence is 3, 16, and 32. This is a post-hoc structural diagnostic, not
    visible evidence.
  - The two count different things. v3 also counts imprecise entrants'
    labels, which become visible from the second year after the entry year.
    Its entry timing is simulated, not copied from the data.
  - The remaining gap at quarter 5 was not decomposed (O21).

**Diagnosis.** Two candidate bottlenecks lead:

- scarce analyst evidence;
- learned and anticipatory terms that behave alike.

v3 changed several things at once: date imprecision, the gap between agent
and analyst evidence, sparse features, case-cohort sampling, and censoring.
No factor was varied on its own. Similar fitted coefficients (0.40 and 0.38
in W1) do not by themselves establish collinearity (O20).

### Data access

| | |
|---|---|
| Read | S1: the 2026-06-26 snapshot extract, episodes, and entrants |
| Read | N11b: confirmatory-window risk sets, entry events, pre-entry features, and the G4 guard's author-id column |
| Not read | N11 confirmatory outcome labels, follow-up outcomes, and visible evidence |

The confirmatory outcome labels stay unread. A future design could still use
them.

## Reporting rules

These rules apply wherever these results are cited.

- **0 of n.** Report a count of 0 of n replicates with its one-sided 95%
  binomial upper bound, 1 − 0.05^(1/n). At n = 8 this is about 31%. It means
  no event was observed, not a zero rate.
- **"Matches".** It means each mean metric lies within the registered 0.02
  margin of the comparator. It is not a test of statistical equivalence.
- **LLM scope.**
  - Both LLM-agent designs used the same local Qwen3-4B-Instruct-2507.
  - The comparison is conditional on the specified worlds and comparison
    conditions.
  - Its outcome reflects calibration and decision rules as well as the policy
    family. The forecaster drifted once evidence entered its prompt, and the
    persona almost never entered.
- **Neural necessity.** The learned arms include linear and tree learners, so
  these results do not establish that a neural policy is necessary.

## Paper-1 direction

The user set this direction on 2026-10-13.

- **Type.** Paper-1 is a method and framework paper.
- **Central claim.** NABM is a reusable framework that separates policy
  learning from social interaction. Fixed policies, adaptive policies, and
  external priors can therefore be compared and traced within one simulation
  lifecycle.
- **Performance comparisons** are conditional cases, not the central claim.
  For the LLM comparison, the safe wording is: "In the specified synthetic
  worlds and comparison conditions, the NABM hybrid met the registered
  performance criteria against two local LLM implementations."
- **The empirical stage** is reported with the closure statement above, not
  as evidence of learning.

## Status of earlier items

| Item | Was | Now |
|---|---|---|
| O17, persistence definition for the indicator's threshold | Open, priority 1 | Superseded: the empirical stage moved to the model ladder (R12) |
| O18, source choice | Open, priority 1 | Answered: the 2026-06-26 snapshot was completed and used (C43) |
| N02 | Your decision | Done |
| N10 | Separate precommit | Done as a threshold transfer, then superseded by the ladder |
| N04, the shared scoring contract (F03) | Optional | Done by Decision 0016 at `29bd59c` |

## What is open

| Rank | Question | Kind |
|---|---|---|
| 3 | Which v3 change cost the power, taken one at a time? (O20) | Synthetic |
| 3 | Why does v3's evidence timing differ from N11b's structure? (O21) | Synthetic |
| 3 | Can another unit or a measured information source separate learning from anticipation? Outside Paper-1 (O22) | Empirical |

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N12 | Paper | Update the claim matrix and outline to the method and framework claim, with these reporting rules | Directed by the user, 2026-10-13 |
| N13 | Comparison design | Decide from the claim matrix which added comparison the central claim needs. First candidate: an LLM baseline given the same information and recalibrated on public outcomes | Separate precommit after N12 |
| N14 | Framework | Design a second consumer first, then extract only capabilities whose shared semantics it confirms | After N12 |

Two notes on these steps:

- A larger local model would test sensitivity to model scale only. It does
  not remove the asymmetry of the comparison design.
- The bridge rule stands. The shared evidence record, the prior slot (F09),
  and the exploration floor (F10) stay domain-owned until a second model
  family needs the same semantics.

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical rows, changes no generic NABM API, selects no model,
and establishes no neural necessity or scientific effect.
