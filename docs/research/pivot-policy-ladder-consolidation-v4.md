# Pivot policy-layer ladder consolidation v4

Date: 2026-10-07

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v4.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v4.yaml)

Extends: [consolidation v3](pivot-policy-ladder-consolidation-v3.md), and through
it v2 and v1, which remain the records for everything before.

## Purpose

Four studies followed v3. One measured a second common LLM-agent design. Three
settled what the empirical stage should measure. This document adds their
claims to the ledger and closes the items they answer. As before, every claim
cites a hash-bound result and a field that must hold, and a test checks all of
them. Nothing new is executed, and no LLM is called.

## The ladder since v3

| Step | Question | Freeze / record | Outcome |
|---|---|---|---|
| Persona LLM agent (N07) | Does an LLM that role-plays each researcher and decides do better? | `150486a` / `02659d6` | No. The NABM hybrid matches it in 9 of 9 cells |
| Observable discriminant (N05) | Which data observable tracks the entry channel? | `5b73c19` / `2c18e9d` | Two persistence-trend observables separate every cell |
| Observable confirmation (N08) | Do they hold at fixed thresholds on new schedules? | `b270e51` / `77b6f96` | Mostly. No false actives, but some missed channels |
| One-sided test (N09) | Does "past the threshold means active" hold? | `c0a8b1a` / `47497da` | Yes for both. Only the outcome trend passes its no-learning control |

## What is established

All results are synthetic. The ledger ids are in parentheses.

**A second LLM-agent design.** The same Qwen3-4B-Instruct-2507 role-played each
researcher and decided whether to pivot. The NABM hybrid matched it in all nine
cells, and the persona selected better than the hand rule nowhere (C36). Used
uncalibrated, the persona said Yes to pivoting with no evidence for no profile.
It never chose to stay after a failed pivot, and it sealed itself in both
interaction worlds (C37). With N06's forecaster, both common LLM-agent designs
have now been matched or beaten by the NABM hybrid in 9 of 9 cells.

**What data can show about the entry channel.**

- The model-internal post-failure entry share is close to necessary but not
  sufficient. Under surge clocks it classified 18 of 24 cells, and every error
  called an inactive cell active (C38).
- Two observables built only from entry timing and persistence separated every
  N05 cell. When learning changes who enters, later entrants persist more
  (C39).
- At thresholds fixed from N05, neither classified every new N08 cell (C40).
- The one-sided reading is confirmed: whenever either observable passed its
  threshold, the channel was active. In N09 each flagged 11 cells with no false
  actives (C41), and across N05, N08, and N09 none of 144 predictions was a
  false active.
- Only the **outcome selection trend** also passed its no-learning control, and
  it is the indicator carried to the empirical stage (C42).

## What was rejected

| Claim | Why it does not hold |
|---|---|
| The post-failure entry share classifies the entry channel on its own (R09) | It produces false actives out of sample, and data cannot observe the time of the first public failure |
| A small cohort-persistence drop shows that learning acts on entry (R10) | Without learning, the fixed rule passes the drop threshold in 8 of 24 N09 cells |
| The observables miss only weak channels (R11) | N09 misses a cell with two passing family-geometry pairs |

## Status of earlier items

| Item | Was | Now |
|---|---|---|
| O08, the post-failure share as the T01 discriminant | Open, priority 1 | Answered: replaced by the outcome selection trend (C38, C42) |
| O14, decision-making LLM agents | Open, priority 3 | Answered for this design (C36, C37) |
| N05, N07 | Proposed | Done |

## The revised empirical target

The empirical target T01 is now one-sided. Within a pivot episode, it is the
rank correlation between when researchers enter and whether they persist:

- **A clearly positive correlation** indicates that learning changes who enters.
- **No correlation, or a negative one,** is not evidence against it. The
  indicator caught 85% of active cells in N09.
- **A later-cohort persistence decline,** the lean H3 pattern, is what these
  worlds produce when learning does not change who enters.

The model-scale threshold of 0.098 is defined for these worlds' persistence
horizon. It transfers to data only after the empirical persistence definition
and horizon are fixed (O17).

## What is open

| Rank | Question | Kind |
|---|---|---|
| 1 | How should persistence and its horizon be defined in OpenAlex so the threshold transfers? (O17) | Empirical |
| 1 | Which source: finish the 2026-06-26 snapshot or use the local 2026-03-30 snapshot under an amended design? (O18) | Empirical; your decision |
| 3 | What lowers the indicator's sensitivity, and can it be raised without false actives? (O19) | Synthetic |

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N02 | Operational | Decide the source (O18) | Your decision |
| N10 | Empirical design | Preregister the empirical outcome selection trend: episodes, years, persistence, and failure actions, before any row is read | Separate precommit after N02 |
| N03 | Empirical method | H3 set-estimator successor | Separate precommit |
| N04 | Framework | Shared out-of-sample scoring contract (F03) | Optional |

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical rows, changes no generic NABM API, selects no model,
and establishes no neural necessity or scientific effect.
