# Pivot local-LLM prior pilot v1

Date: 2026-09-29

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_llm_prior_pilot_v1.yaml`](../../experiments/configs/pivot_llm_prior_pilot_v1.yaml)

## Why this pilot exists

NABM aims to sit between rule-based ABMs and LLM-agent simulations: heavier
than hand-written rules because agents learn, and much lighter than querying an
LLM at every agent step. The ladder so far showed two things about that middle
ground:

- small learned agents are evidence-starved, since neighbor-scale evidence left
  every feature learner worse than its prior; and
- pooled learned policies cost only a few times a rule-based ABM per agent-step.

This pilot tests the natural hybrid. An LLM contributes knowledge once, as the
prior every agent starts from, and NABM does the learning and the scaling.

By user instruction, the LLM is local: `Qwen/Qwen3-4B-Instruct-2507` on the
workstation GPU, pinned by snapshot and file hashes. No paid API is called.

## Phase A: one-time elicitation

**Profiles.** 1,024 researcher profiles are drawn from the Pivot population
generator, using the hot-field-hype scenario and a fresh seed. Each prompt
describes one profile's nine policy features in plain language, each scored
from 0 to 1.

**Readouts.**

- *Primary:* ask whether the pivot will be productive, and read
  P(Yes) / (P(Yes) + P(No)) from the first answer token's logits. No sampling
  is involved.
- *Secondary:* ask for a probability from 0 to 100 and decode greedily.

**Determinism.** Each prompt runs alone, with no padding. A dry run on a
disjoint 64-profile cohort showed that bf16 batching could shift the log-odds
by up to 1.0. That dry run checked only format validity, determinism, and
timing; no probabilities were inspected. A repeat pass must reproduce the
primary logits exactly. A batch-32 pass on the first 32 profiles is reported as
a sensitivity diagnostic.

**Distillation.** The primary log-odds are fitted by a ridge regression on the
raw features. This yields nine weights and a bias in exactly the units of the
existing hand-set prior. The slot therefore accepts the LLM prior unchanged. The
pilot reports how faithful the fit is: R² on log-odds, and Spearman rank
correlation with both the primary and secondary readouts.

## Phase B: ABM evaluation

Four priors fill the same slot, the study-2 initial policy that every agent and
learner builds on:

| Prior | Source | Role |
|---|---|---|
| `hand` | The existing pressure rule | Nested predecessor |
| `llm` | Distilled Qwen elicitation | The hybrid under test |
| `oracle_linear` | Logistic fit to each world's structural labels | Best linear prior, reference |
| `flat` | All zeros | No knowledge, reference |

The arms are the fixed rule, the neighbor-evidence online logistic learner,
and the cross-fitted field linear, tree, and MLP learners from ABM v2. Each is
run in all three fit worlds under immediate visibility, with 16 paired
replicates of 256 agents and fresh seeds. Under the hand prior, the fixed and
online arms must equal the persistence predecessor. The slot must be restored
after every task.

## Registered decisions

Each decision is made per world and compares `llm` with `hand`, using the paired
rule: the mean must meet the threshold, and the improvement must be strictly
positive in at least 12 of 16 replicates.

| Decision | Metric | Threshold | Dispositions |
|---|---|---:|---|
| Prior quality | fixed-rule probe Brier | 0.01 | LLM better / hand better / no clear difference |
| Local rescue | online-logistic probe Brier | 0.01 | LLM improves local learning / hand better / no clear difference |
| Pooled washout | field-arm probe Brier, both directions | 0.01 | washes out / persists (arms listed) |
| Entry precision | fixed-rule productive share of entrants | 0.02 | LLM selects better / hand selects better / no clear difference |

The oracle and flat priors are reported as references, not decided.

## Cost ledger

The elicitation receipt records the GPU time of the one-time elicitation and
the time per prompt. Multiplying the per-prompt time by one query per agent-step
gives a lower bound for an LLM agent in the same ABM. It is a lower bound
because a real LLM agent would also need its evidence in context and would
generate text. The NABM side comes from the relative-delay cost ledger.

## Order and reproducibility

The contract is committed first. The elicitation then runs under that commit
and writes hash-bound artifacts. The ABM run verifies those hashes, and its
receipt binds them. Re-running the elicitation must reproduce the elicitation
records and the distilled prior exactly. Re-running the ABM must reproduce every
scorecard artifact byte for byte.

## Boundary

The fit worlds are synthetic, and so is the notion of a productive pivot. How
well an LLM prior scores against them shows how its intuitions line up with the
model designer's stylized worlds. It does not show that the LLM knows how real
pivots succeed. The pilot reads no empirical rows, makes no paid API calls,
selects no model, and changes no generic NABM API.
