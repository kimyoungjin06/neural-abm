# Pivot policy-layer ladder consolidation v6

Date: 2026-10-19

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v6.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v6.yaml)

Extends: [consolidation v5](pivot-policy-ladder-consolidation-v5.md), and
through it v1 to v4, which remain the records for everything before.

## Purpose

v5 set the Paper-1 direction and left two steps:

- N13, a fairer LLM comparison;
- N14, a second consumer before any extraction.

The [information-spread design brief](info-spread-policy-ladder-design-v1.md)
took both on in one study. This document is that study's stage D6, run on
the user's go-ahead on 2026-10-19. It records:

- what the second domain established;
- the N14 reusability test;
- the reporting rules these results need.

As before, every claim cites a hash-bound result and a field that must hold,
and a test checks all of them. This document executes nothing.

## The second domain

**The world.** Information spread on a Watts–Strogatz network of 1,000
agents. Items carry four features, and half are fact-checked four steps after
they appear. Agents share when the probability that an item is true reaches
c/(1 + c).

**Regimes.**
- **A:** one pooled policy, with c = 1 for every agent.
- **B:** per-agent costs and partial pooling.

**Arms.** The arms ask one question: does an adaptive NABM policy differ
from a policy table written before the run?

| Arm | What it is |
|---|---|
| F0, F1 | The threshold rule and the hand rule |
| T0-1, T0-4 | Tables written before the run, from one donor run or the average of four, in the same world |
| L0 | The adaptive learner |
| P0, X0, L1 | A once-elicited LLM prior, AgentTorch-style LLM archetypes, and the learner started from P0 (local Qwen3-4B-Instruct-2507) |

| Stage | Freeze / record | Outcome |
|---|---|---|
| D1 feasibility | `dfc5372` / `d8e746e` | 14 exact nesting checks; the core reproduces Pivot's learners byte for byte |
| D2–D2c controls | `d6c5d44` / `bfd6269` | The revised design is adequate on controls |
| D4, study v1 | `daf5d80` / `947ca19` | `DESIGN_DEFECT_IN_NULL_WORLD`; the study groups were not read |
| Leak diagnosis | exploratory / `7a2e17b` | MLP learners recognize items already fact-checked |
| D2d, the unverified measure on controls | `09c0a18` / `bf49d03` | Adequate. The null control shows the leak on the old measure and not on the new |
| D4 v2, study v2 | `dcc6e74` / `27ce720` | `TABLE_SUFFICES_EVEN_IN_B` |
| D5, the LLM arms | `dcc6e74` / `9c164c3` | NABM beats the archetypes; the LLM prior adds nothing detectable |
| N14 parity | `c361a48` / `01f2f7a` | The Pivot ladder runs through the core byte for byte |

## What is established

All results are synthetic. Ledger ids are in parentheses.

- **A second domain runs the comparison through a shared core (C47).**
  - All 14 exact nesting checks pass.
  - With the Pivot prior, the core's learners reproduce Pivot's fitters byte
    for byte.
- **The Pivot ladder reproduces through the same core (C48).**
  - The Pivot LLM-prior pilot v3 was re-expressed through the core in its
    three learner arms.
  - It wrote a `runs.jsonl` identical to its record: 1,152 of 1,152 rows.
- **The closed loop wins when a run carries information donors lack
  (C49).** In the POS-sign control, the learner beats both tables in all four
  groups.
- **A registered null check caught a leak before any study group was read
  (C50).**
  - MLP learners recognized items whose fact-check was already public.
  - On items still unverified, the gain is gone.
- **In stationary worlds, a table written before the run suffices (C51).**
  - Every learner beats the hand rule.
  - The closed loop at matched evidence is material in 1 of 12 study groups
    (`TABLE_SUFFICES_EVEN_IN_B`).
  - The four-donor table beats the learner in all 12.
  - The donors ran in the same world.
- **NABM beats LLM archetypes (C52).** Given the same public information,
  the learner beats AgentTorch-style archetypes in all 12 study groups.
- **An LLM prior adds nothing detectable inside the learner (C53).** This
  holds even in Regime B, where the prior alone beats the hand rule.
- **Every registered run reproduced byte for byte (C54).** The two LLM
  elicitations matched, and the L0 parity check held in all 320 tasks.

## What was rejected

| Claim | Why it does not hold |
|---|---|
| Under public fact-checks, share precision over all items is a clean measure (R16) | MLP learners recognize fact-checked items, so they beat the hand rule in a null world |
| Regime B's per-agent costs with partial pooling make the closed loop matter in stationary worlds (R17) | 0 of 6 Regime B study groups. The donors' per-agent residuals carried over to the same agents. |

## What is open

| Rank | Question | Kind |
|---|---|---|
| 1 | Which order of outcome rows reproduces both consumers under delays? (O27) | Framework |
| 2 | Does a table still suffice when its donors ran in other worlds? (O23) | Synthetic |
| 2 | Where between stationary worlds and run-specific information does the closed loop begin to matter? (O26) | Synthetic |
| 3 | Does the LLM prior carry knowledge in Regime A with an unsaturated readout? (O24) | Synthetic |
| 3 | Do the comparisons hold on the agents' utility rather than precision? (O25) | Synthetic |

## N14: reusability

The bridge rule required a second consumer before any extraction. N14 met
it in both directions:

- the second domain uses the core (C47);
- the first domain reproduces through it (C48).

Decision 0017 proposes which shared semantics to extract:
[Decision 0017](../decisions/0017-policy-ladder-semantics-extraction.md).

| Extract | Keep in the domain |
|---|---|
| The prior rule, keyed randomness, the exploration floor, the prior-anchored residual linear and MLP learners, prior-boosted trees | Fixed-rule evaluation, an agent's own parameters, the exact Newton learner, partial pooling, table replay, cross-fitting, every world and measure |
| The outcome record, once O27 fixes its ordering rule | |

The re-expression found two reasons for the split.
- **A fixed rule is not numerically portable.** Pivot's torch rule differs
  from a numpy prior in the last bits.
- **Pivot holds the prior in two places.** It is the learners' anchor and
  each agent's own parameters. The core holds only the first.

Implementation awaits the user's approval.

## Reporting rules

These rules add to v5's.

- **Unverified measure.**
  - From study v2 on, information-spread comparisons are on share precision
    on unverified items.
  - Study v1's groups are not read.
- **Same-world donors.** "A table suffices" holds for donor runs in the same
  world. Transfer across worlds is not tested.
- **Precision is not utility.** Precision rewards selectivity. Report share
  counts beside it when a comparison may turn on selectivity.
- **LLM readout.** P0 in Regime A saturated under the inherited clip and is
  nearly constant. It is not evidence about the model's knowledge.

## Paper-1

**The central claim is unchanged.** NABM is a reusable framework that
separates policy learning from social interaction, so fixed policies,
adaptive policies, and external priors can be compared and traced within one
simulation lifecycle.

**What this consolidation adds:**

- **Reusability, at repository level.** Two consumers share the core's
  semantics byte for byte. Package extraction is proposed, not implemented.
- **Four conditional cases:**
  - the precomputed-table critique holds in stationary, same-world designs,
    and fails when the run carries information donors lack;
  - NABM learners beat AgentTorch-style archetypes;
  - an LLM prior adds nothing detectable inside the learner;
  - a registered null check caught a leak before any study group was read.

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N15 | Framework | Implement Decision 0017, starting with the outcome-row-order parity on a delayed Pivot run | The user's approval |
| N16 | Paper | Write the policy-ladder section with these cases and rules | After this consolidation |
| N17 | Comparison design | Choose among O23–O26, each as a separate precommit | The user's choice |
| N18 | Repository | Merge the information-spread branch into `main` through a pull request | The user's confirmation |

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical rows, and changes no generic NABM API. It selects no
model and establishes no neural necessity or scientific effect.
