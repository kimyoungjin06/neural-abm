# Pivot policy-layer ladder consolidation v3

Date: 2026-09-30

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v3.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v3.yaml)

Extends:
[consolidation v2](pivot-policy-ladder-consolidation-v2.md), which in turn
extends [v1](pivot-policy-ladder-consolidation-v1.md). Both remain the records
for everything before.

## Purpose

Two studies followed v2:

- the model-size ladder, LLM prior pilot v4; and
- the LLM-agent reference, N06.

v2 left the LLM-agent end of NABM's positioning estimated from query counts.
N06 measured it. This document adds the new claims to the ledger, closes the
items they answer, and updates the positioning. As before, every claim cites a
hash-bound result and a field that must hold, and a test checks all of them.
Nothing new is executed, and no LLM is called.

## The ladder since v2

| Step | Question | Freeze / record | Outcome |
|---|---|---|---|
| LLM prior v4 | Can a lighter, faster LLM fill the prior slot? | `15dc4d1` / `4e923ca` | No. 4B is the smallest model that works; below it quality is erratic |
| LLM-agent reference (N06) | Does an LLM queried at every step do better than NABM? | `2a25ab5` / `2d469c1` | No. The NABM hybrid matches or beats it in 9 of 9 cells, at a fraction of the cost |

## What is established

All results are synthetic. The ledger ids are in parentheses.

**Model size.** Within one Qwen3 release, the 4B model is the smallest whose
calibrated prior, with the exploration floor, dominates the hand rule in a
majority of cells (7 of 9). Its selective prior selects better in all nine
(C26). The 0.6B model dominates in 4 of 9 and the 1.7B in 1 of 9, so quality is
not monotone below 4B (C27). The v1 first-token readout fails for the 1.7B
model, while a prefilled forced choice works for all four (C28). The exact v3
prior dominates again on fresh seeds, in 6 of 9 cells (C29).

**Cost of a lighter model.** Every model's one-time elicitation takes under 50
GPU seconds. Batched in 32s, the 0.6B model is about 4.7 times faster than the
4B; at batch 1 it is not faster (C30). Post hoc, a prior's design-cohort AUC
tracks its selection in the ABM, with a Spearman correlation of 0.857 (C31).

**The LLM-agent reference.** The NABM hybrid and the LLM agent used the same
Qwen3-4B-Instruct-2507 and the same exploration floor. The hybrid matched the
LLM agent, queried at every step, in all 9 of 9 cells. It was superior on
learning in nine and on entrant precision in eight (C32). The LLM agent
dominated the hand rule in no cell (C33).

Post hoc, the LLM agent's weakness was calibration, not ranking. It ranked probe
profiles about as well as the pooled nonlinear learners, but once outcomes were
public, its mean output rose well above the population rate, and it over-entered
(C34). It needed a median of about 110 GPU-seconds per run. An NABM arm run
needed about 5 single-core CPU-seconds, and the hybrid's whole LLM cost was one
elicitation of under 50 GPU seconds (C35).

## What was rejected

| Claim | Why it does not hold |
|---|---|
| A smaller model of the same family can replace the 4B prior (R08) | The 0.6B and 1.7B priors dominate in a minority of cells and select worse |

## Status of earlier items

| Item | Was | Now |
|---|---|---|
| O09, the LLM-agent reference | Open, priority 2 | Answered for this design (C32, C33, C35). The decision-making LLM agent stays open as O14 |
| N06 | Proposed | Done (`2a25ab5` / `2d469c1`) |
| The LLM-agent tier of the positioning | Estimated, not measured | Measured in this design |

## Updated positioning

| Tier | Cost | Behavior in the Pivot case |
|---|---|---|
| Rule-based ABM | 0.149 ms CPU per agent-step | Cannot produce the visibility response |
| NABM, pooled | 1.9 to 6.7 times the rule | Learns interaction structure out of sample |
| NABM with an LLM prior | Pooled cost plus one elicitation of under 50 GPU seconds | Beats the hand rule in 7 of 9 cells with the 4B prior |
| LLM agent, queried every step | About 110 GPU-seconds per run, every run | Matched or beaten by the NABM hybrid in 9 of 9 cells; beats the hand rule nowhere |

The ceiling on this claim is narrow. N06 tested one reasonable in-context
forecaster design, with one local 4B model, in synthetic worlds. It does not
cover every LLM agent, and it describes no real researchers. The finding that
carries over is the mechanism. Pooled learners are fitted to outcomes, so they
stay calibrated. The LLM's reaction to evidence in its prompt did not.

## What is open

| Rank | Question | Kind |
|---|---|---|
| 2 | What base rate should calibrate the LLM prior in real data? (O16) | Empirical |
| 3 | Would online recalibration close the LLM agent's gap, and at what cost? (O13) | Synthetic |
| 3 | How does an LLM agent that decides for itself, role-playing the researcher, compare? (O14) | Synthetic |
| 3 | How much of the LLM agent's ranking comes from the nearest-outcome counts in its prompt? (O15) | Synthetic |

The open questions inherited from v1 and v2 remain. O08, the post-failure
entry-share discriminant, is still the highest-priority synthetic item for the
Pivot empirical stage.

## Framework decisions

| Capability | Decision |
|---|---|
| LLM-agent slot (F14) | Domain-owned. A scoped agent-class slot, restored after every run and nested against the fixed rule |
| Prefilled forced-choice readout (F15) | Repository-only research instrument; the package gains no LLM dependency |

The studies add four practices (F16):

- check each model's readout format with a format-only dry run before freezing;
- query at batch size 1 when an answer must not depend on its batch;
- in synthetic worlds, screen priors by design-cohort AUC before running the
  ABM; and
- do not assume that a calibration made without evidence holds for prompts that
  contain evidence.

F03, out-of-sample policy scoring, is still the only extraction candidate.

## Next steps

| Id | Kind | Action | Serves | Authorization |
|---|---|---|---|---|
| N02 | Operational | Resume the OpenAlex transfer and close source intake | Pivot empirical stage | Your decision |
| N03 | Empirical method | H3 set-estimator successor on a new disjoint stress grid | Pivot empirical stage | Separate precommit |
| N04 | Framework | Shared out-of-sample scoring contract for F03 | Generic NABM API | Optional |
| N05 | Synthetic | Register the post-failure entry-share discriminant under an attention-surge clock | Pivot empirical stage | Separate precommit |
| N07 | Synthetic | A decision-making or online-recalibrated LLM agent, only if a research question needs one | NABM positioning | Separate precommit |

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical rows, changes no generic NABM API, selects no model,
and establishes no neural necessity or scientific effect.
