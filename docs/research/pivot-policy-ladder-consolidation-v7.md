# Pivot policy-layer ladder consolidation v7

Date: 2026-10-22

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v7.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v7.yaml)

Extends: [consolidation v6](pivot-policy-ladder-consolidation-v6.md), and
through it v1 to v5, which remain the records for everything before.

## Purpose

v6 left four next steps. All four are now done:

- **N15.** Decision 0017 was implemented and merged into `main`.
- **N16.** Paper Section 8 was drafted and merged.
- **N17.** The user chose to run O23, O26, O24, and O25 in order, each as
  its own precommit.
- **N18.** The information-spread branch was merged.

This document records what N15 and the four follow-ups established, and
which v6 readings they narrow. As before, every claim cites a hash-bound
result and a field that must hold, and a test checks all of them. This
document executes nothing.

| Study | Freeze / record | Outcome | Runs (development / reproduction) |
|---|---|---|---|
| N15 package parity | `cb0aa44` / `8b7a31b` | `N15_PARITY_HOLDS_THE_PACKAGE_REPRODUCES_BOTH_PIVOT_RECORDS` | Workstation, CPU |
| O23 transfer | `fded050` / `9a59e11` | `TRANSFERRED_TABLE_SUFFICES` in both regimes | 36 / 38 min |
| O26 flip rate | `86d3e80` / `38120b1` | Onset at p = 0.25 in every family | 115 / 114 min |
| O24 unclipped prior | `eb49057` / `da84c2d` | The prior informs; it adds little inside the learner | 85 / 85 min |
| O25 utility | `ebd2c8d` / `82088db` | `SOME_CONCLUSIONS_CHANGE_UNDER_UTILITY` | 156 / 155 min |

All four follow-ups ran on GPU-Server and reproduced byte for byte (C62).

## What is established

All results are synthetic. Ledger ids are in parentheses.

- **The package reproduces both recorded Pivot studies (C55).**
  - Through the modules of Decision 0017, the LLM-prior pilot v3 writes a
    file identical to its record.
  - The 432 field-public learner rows of the nonlinear policy ABM v2 are
    each identical to theirs, including under delays.
  - The ordering rule (visible step, then insertion) recovers Pivot's
    release order. This answers O27.
- **A table from other worlds suffices too (C56).**
  - The donors ran in other worlds from the same generator: another network,
    other thresholds, and other costs, but the same veracity relation.
  - The four-donor table from those worlds beats the learner in all 12
    study groups, by 0.012 to 0.054.
  - Same-world donors were worth nothing material over other-world ones
    (0 of 12).
- **The closed loop wins once a quarter of runs carry run-specific
  information (C57).**
  - Each run's credibility relation was reversed with probability p.
  - The learner beats the four-donor table from p = 0.25 in every family.
  - It beats the single-donor table already at p = 0.1, where 7.8% of runs
    were reversed.
- **Without the inherited clip, the LLM prior carries the relation's
  direction (C58).**
  - D5's first elicitation, read without the clip, distils with R² 0.81 in
    Regime A and 0.76 in Regime B.
  - Its credibility, novelty, and intensity weights have the world's signs.
  - It beats the hand rule in all 12 study groups. D5's flat Regime A prior
    was the readout, not the model.
- **On precision, an informative LLM prior adds little inside the learner
  (C59).** L1u − L0 is positive in 5 of 12 groups, by 0.012 to 0.020, and
  negative in none.
- **Under the agents' utility, the framework comparisons hold (C60).**
  - The best table still beats the learner in all 12 groups.
  - The closed loop at matched evidence is still `MIXED_OR_NONE`.
  - The learner still beats the hand rule.
  - NABM still beats the archetypes in all 12 groups.
- **Under utility, an LLM prior inside the learner lowers the agents'
  utility (C61).**
  - Five of 20 summaries change, and all involve an LLM prior. Each was
    `MIXED_OR_NONE` on precision.
  - L1 − L0 is negative in 6 of 6 Regime A groups and 4 of 6 Regime B
    groups.
  - L1u − L0 is negative in 4 of 6 Regime B groups, and positive in the
    other 2.
  - In Regime B, the closed loop on either prior raises utility over the
    prior alone (4 and 5 of 6).

## What is narrowed or rejected

| v6 claim | Scope now |
|---|---|
| C51, a table suffices in stationary worlds | Holds for donors in the same world and in other worlds of the same generator (C56), and only while run-specific information is rare (C57) |
| C53, an LLM prior adds nothing detectable inside the learner | Holds on precision only. Under utility it lowers the agents' utility (C61) |

| Claim | Why it does not hold |
|---|---|
| In Regime B, per-agent residuals keep a table from transferring across worlds (R18) | The transferred table beat the learner in all 6 Regime B groups. This was O23's stated expectation |
| A prior that adds nothing detectable on precision is neutral for the agents (R19) | Under utility, L1 − L0 is `MOSTLY_NEGATIVE` in both regimes |

**Why R18 failed, one explanation, not tested.** Costs enter only the
decision cutoff, which is computed from the evaluation world. So what must be
learned is a pooled belief, and a pooled belief transfers.

**Why R19 failed, one explanation, consistent with the records but not
tested.** A learner started from a selective prior keeps its precision but
shares less, and fewer shares at the same precision forgo shares worth
making.
- **Recorded description.** D5's unregistered share counts put L1 below L0
  in both regimes.
- **Post-hoc description, not in the ledger.** Read from the recorded D5
  and study v2 development cells on GPU-Server, L1's mean unverified share
  count is below L0's in each of the 12 study groups. It is 17–22% lower in
  Regime A and 9–10% lower in Regime B.

## What is open

| Rank | Question | Kind |
|---|---|---|
| 1 | Does the utility cost of an LLM prior inside the learner come from its calibration or from the model's answers? (O28) | Synthetic |
| 2 | Does the onset move with a table from more donors, or with information that shifts within a run? (O29) | Synthetic |
| 3 | Does a table still suffice across veracity relations or domains? (O30) | Synthetic |
| 3 | Do the comparisons hold when an arm is also charged for the reach of the false items it shares? (O31) | Synthetic |

## Reporting rules

These rules replace two of v6's and carry the other two.

- **Unverified measure** (carried). Information-spread comparisons from
  study v2 on are on share precision on unverified items. Study v1's groups
  are not read.
- **Transfer within one generator** (replaces "same-world donors"). "A
  table suffices" holds for donors in the same world and in other worlds of
  the same generator, which share the veracity relation. Transfer across
  relations or domains is not tested.
- **Rare run-specific information** (new). The table critique holds only
  while run-specific information is rare. Here a four-donor table falls
  behind between a tenth and a quarter of runs carrying it. That is one kind
  of information and one table size.
- **Precision and utility** (replaces "precision is not utility"). Report
  comparisons that involve an LLM prior under both, since they disagree in
  5 of 20 summaries. Utility scores only the sharer's own gain and cost, not
  the reach of false items.
- **LLM readout** (carried, extended). Name the readout with any LLM-prior
  result. P0 in Regime A saturated under the inherited clip; without the clip
  the same answers carry the relation's direction.

## Paper-1

**The central claim is unchanged.** NABM is a reusable framework that
separates policy learning from social interaction, so fixed policies,
adaptive policies, and external priors can be compared and traced within one
simulation lifecycle.

**What this consolidation changes:**

- **Reusability.** The two consumers now share package semantics on `main`,
  byte for byte. The surface is not in a released tag.
- **The precomputed-table critique.** It holds across worlds of one
  generator. It fails once about a quarter of runs carry run-specific
  information.
- **LLM priors.**
  - NABM beats the archetypes on precision and on utility.
  - An LLM prior adds nothing detectable inside the learner on precision,
    and lowers the agents' utility.
  - The case to report is the disagreement itself: a comparison scored only
    on precision would have missed it.

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N19 | Paper | Update Section 8, the claim matrix, and the outline with C55–C62 and these rules | After this consolidation |
| N20 | Repository | Merge the follow-up branch into `main` through a pull request | Confirmed by the user on 2026-10-22 |
| N21 | Comparison design | Choose among O28–O31, each as a separate precommit | The user's choice |
| N22 | Release | Tag a next alpha that ships the policy-ladder package surface | The user's confirmation |

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical rows, and changes no generic NABM API. It selects no
model and establishes no neural necessity or scientific effect.
