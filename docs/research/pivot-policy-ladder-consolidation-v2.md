# Pivot policy-layer ladder consolidation v2

Date: 2026-09-29

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v2.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v2.yaml)

Extends:
[consolidation v1](pivot-policy-ladder-consolidation-v1.md), which remains the
record for the ladder through nonlinear policy ABM v2.

## Purpose

After v1, the ladder added four studies: the relative-delay and cost study
(N01) and three local-LLM prior pilots. This document does three things:

- adds their claims to the ledger and updates the v1 items they answer;
- states what the whole ladder implies for NABM's intended position, between
  rule-based ABMs and LLM-agent simulations; and
- lists the next steps, with the goal each one serves.

As in v1, every claim cites a hash-bound result file and a field that must
hold, and a test checks all of them. The v1 ledger is bound by hash, and its
own test still audits the inherited claims. Nothing new is executed, and no
LLM is called.

## Where NABM sits

NABM is meant to be heavier than a rule-based ABM and much lighter than an
LLM-agent simulation. Agents carry small learned policies that take a hand
rule, or an LLM judgment elicited once, as their prior. They learn from pooled
outcome records rather than from a model call at every decision. The Pivot
ladder has now measured most of that range:

| Tier | Agent policy | Cost per agent-step | What the ladder found |
|---|---|---|---|
| Rule-based ABM | Fixed hand-written rule | 0.149 ms CPU (1×) | Cannot produce the visibility response (C05) |
| NABM, pooled | Learned policy with the rule as prior and a pooled outcome record | 1.9 to 6.7 times the rule | Learns interaction structure that holds out of sample (C10) |
| NABM, per agent | A separate learner per agent, on neighbor outcomes | 57 to 65 times the rule | Worse than the prior (C11), so not the design |
| NABM with an LLM prior | Calibrated, once-elicited LLM prior, an exploration floor, and pooled learning | Pooled cost, plus under 10 GPU minutes once | Beats the hand rule on selection while keeping pooled learning within the registered margin, in 7 of 9 cells (C22) |
| LLM-agent simulation | LLM queried at every agent-step | One query per agent-step | Estimated, not measured (O09) |

Costs are single-thread CPU time from the N01 benchmark, measured under load,
so they are indicative. For the LLM tier, the N01 design would need 14.2 million
policy queries. The first LLM pilot's design would need at least an estimated
38 to 125 GPU hours on the local model. The hybrid used 3,104 queries in total,
and its two successors made none.

What the ladder implies for the position:

- **NABM nests the rule ABM.** Every step of the ladder exactly reproduces its
  predecessor before a new arm is interpreted. The rule ABM is a special case,
  so any difference comes from the learned layer.
- **The value comes from learned policies, not from neural ones as such.** The
  MLP wins on smooth success boundaries and a tree on axis-aligned ones (C09).
- **Small learned agents are evidence-starved.** Per-agent learners are both
  the most expensive design and the one that loses to its prior. The working
  design is rule as prior, learned residual, and pooled evidence (C11, C16).
- **LLM knowledge enters best once, through the prior slot.** It works only
  after aggregate calibration (C19), and exploration has to be kept as a
  separate parameter (C21, C22).
- **Registered runs are byte-reproducible.** Each of the four studies since v1
  was run twice and matched byte for byte. The exception is the timing
  benchmark, which is noisy by nature. Paired designs with more than a thousand
  runs depend on that.

The claim ceiling is the same as for every study in the ladder. The worlds are
synthetic and stylized. The LLM-agent tier is estimated from query counts, and
nothing here describes real researchers.

## The ladder since v1

| Step | Question | Freeze / record | Outcome |
|---|---|---|---|
| Relative delay and cost (N01) | Does saturation depend on delay relative to the entry wave? What does each tier cost? | `22605d5` / `6d839de` | The boundary moves with the wave, but the registered threshold test failed |
| LLM prior v1 | Can a once-elicited local LLM prior stand in for the hand rule? | `75fbbf7` / `533ed03` | No. Uncalibrated, it self-seals |
| LLM prior v2 | Does calibration with aggregate information fix it? | `2ddcd93` / `fe4033e` | Yes, but selectivity starves pooled learning |
| LLM prior v3 | Does an exploration floor resolve that trade-off? | `87c90eb` / `d896782` | Mostly. The 20% hybrid dominates in 7 of 9 cells |

The LLM was Qwen3-4B-Instruct-2507, pinned by snapshot and run locally. No paid
API was called.

## What is established

All results are synthetic. The ledger ids are in parentheses.

**Timing.** Spreading entry over more steps keeps learned precision acting on
entry under longer failure delays. With entry hazard `q = 1`, only immediate
visibility is active. At `q = 0.5`, delays of 0 and 1 are active. At `q = 0.25`,
delays of 0, 1, 2, and 4 are all active (C15).

**Cost.** Pooled linear, tree, and MLP policies cost 1.9 to 6.7 times the
rule-based ABM per agent-step. The same families learning per agent from
neighbor outcomes cost 57 to 65 times, and they are also the learners that do
worse than the prior (C16).

**The LLM prior.** Used on its raw scale, the elicited prior is so pessimistic
that no agent enters. No evidence is produced to correct it, and the hand rule
wins every decision (C17). Its ranking is nonetheless about as good as the hand
rule's (C18). Shifting its intercept to a single stated base rate, without any
outcome labels, removes the self-sealing (C19). The calibrated prior then picks
more productive entrants than the hand rule in every world (C20). It admits
about a third of agents, whose entrants are productive 46% to 48% of the time,
against 33% under the hand rule.

**Selection and exploration.** A selective prior leaves pooled learners less
evidence. Without exploration, the calibrated prior dominates the hand rule in
only 3 of 9 cells (C21). A prior-independent exploration floor separates the
two roles. With a 20% exploratory share, the hybrid dominates in 7 of 9 cells,
and with 10% in 5 of 9. At 40%, the precision gain is gone (C22). Post hoc,
raising the share lowers entrant precision and raises evidence in all nine
cells, and improves pooled learning in seven (C23).

**Local learning.** Post hoc, the hand pressure rule is a worse starting point
for neighbor-scale learning than no prior at all. Most of the LLM variants'
local-learning gain over it comes from escaping it. Only the verbalized readout
beats the flat prior (C24).

**LLM cost.** Eliciting the prior took under 10 GPU minutes. Querying the LLM
at every agent-step of the same design would take an estimated 38 to 125 GPU
hours at least (C25).

## What was rejected

| Claim | Why it does not hold | Evidence |
|---|---|---|
| A single threshold on delay relative to the fixed rule's entry-wave width separates active from saturated cells (R07) | The fixed rule has a slow entry tail while learning arms rush in, so it is the wrong normalizer | N01 registered `NEITHER_SIMPLE_RULE` |

No v1 claim was overturned.

## Status of v1 items

| Item | Was | Now |
|---|---|---|
| O04, saturation versus entry-wave width | Open, priority 1 | Partly answered by C15 and R07. The normalizer question moves to O08 |
| N01 | Proposed | Done (`22605d5` / `6d839de`) |
| T01, relative failure-visibility delay | Delay divided by entry-wave width | The candidate discriminant is now the post-failure entry share. Its empirical analog is the share of topic entry after later-cohort persistence decline became observable |

For T01, the requirement that O04 be resolved is replaced by a new one. O08
must be registered and pass under an attention-surge clock.

## What is open

The new questions:

| Rank | Question | Kind |
|---|---|---|
| 1 | Registered in advance, does the post-failure entry share separate active from saturated cells under an attention-surge clock? (O08) | Synthetic |
| 2 | On a reduced design, how do the hybrid's behavior and cost compare with a local LLM queried at every agent-step? (O09) | Synthetic |
| 2 | Does the elicited prior carry knowledge of real pivot success? (O10) | Empirical |
| 3 | Does the exploration frontier hold under delayed failure visibility? (O11) | Synthetic |
| 3 | Why is the linear field learner in the linear world never non-inferior to the hand rule, even with the most evidence? (O12) | Synthetic |

Post hoc, the post-failure entry share separates all twelve N01 cells, but only
by a thin margin: at least 0.155 when active, at most 0.147 otherwise. O08 is
therefore a hypothesis to register, not a finding. O01 to O03 and O05 to O07
from v1 remain open.

## Framework decisions

The v1 rule still applies. A Pivot capability is generalized only when a second
model family needs the same semantics.

| Capability | Decision | Second consumer |
|---|---|---|
| Prior slot for external knowledge (F09) | Domain-owned; watch it, since it is central to the positioning | None |
| Exploration-floor adapter (F10) | Domain-owned | None |
| Local LLM elicitation and distillation (F11) | Repository-only research instrument. The package gains no LLM dependency, and pinned dependencies are added only for the run | None |
| Compute-cost benchmark (F12) | Engineering pattern, not extracted | None |

F03, out-of-sample policy scoring, is still the only extraction candidate.

The pilots add method guidance for hybrid priors (F13):

- calibrate an elicited prior with aggregate information only;
- report entry and evidence volume for every prior;
- keep selection and exploration as separate registered parameters;
- fix the LLM readout batch size, because batching shifted log-odds by up to
  0.75; and
- bind once-elicited outputs by hash and reuse them without new calls.

## Next steps

| Id | Kind | Action | Serves | Authorization |
|---|---|---|---|---|
| N02 | Operational | Resume the OpenAlex transfer and close source intake | Pivot empirical stage | Your decision |
| N03 | Empirical method | H3 set-estimator successor on a new disjoint stress grid | Pivot empirical stage | Separate precommit |
| N04 | Framework | Shared out-of-sample scoring contract for F03 | Generic NABM API | Optional |
| N05 | Synthetic | Register the post-failure entry-share discriminant under an attention-surge clock | Pivot empirical stage | Separate precommit |
| N06 | Synthetic | Reduced-design LLM-agent reference arm, with local Qwen queried at every agent-step | NABM positioning | Separate precommit |

N06 is the only step that measures the LLM-agent end of the positioning
directly. N02, N03, and N05 together prepare the first empirical comparison.

## Boundary

This document consolidates recorded results. It executes nothing, calls no
LLM, reads no empirical rows, changes no generic NABM API, selects no model,
and establishes no neural necessity or scientific effect.
