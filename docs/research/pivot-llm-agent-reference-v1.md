# Pivot LLM-agent reference v1 (N06)

Date: 2026-09-30

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_llm_agent_reference_v1.yaml`](../../experiments/configs/pivot_llm_agent_reference_v1.yaml)

## Why this study exists

NABM claims to be much lighter than an LLM-agent simulation while doing
comparable work. Until now, the LLM-agent end of that claim was estimated from
query counts, not measured
([consolidation v2](pivot-policy-ladder-consolidation-v2.md), open question
O09). This study measures it directly. The same local LLM is used in two ways:

- **NABM hybrid.** The LLM is consulted once, distilled into the prior slot,
  and calibrated. Pooled field learners then learn from public outcomes.
- **LLM agent.** The LLM is queried at every agent-step as the agent's policy,
  with the public outcomes in its prompt.

Both use v3's 20% exploration floor, so only the policy differs. The model is
Qwen3-4B-Instruct-2507, the strongest prior in
[v4](../../experiments/evidence/results/pivot_llm_prior_pilot_v4_20260930.summary.md).
The hybrid uses v4's calibrated prior for this model, bound by hash.

## The LLM agent

The agent sits in the fixed-rule lifecycle, which is otherwise unchanged. Its
policy output is the LLM's calibrated probability that a pivot would be
productive. Readiness mixing, social exchange, entry, and exit are exactly as
in every other arm. A registered nesting check confirms this in every
linear-world replicate. When the agent's own hand policy is plugged into the
same machinery, it reproduces the fixed-rule arm exactly.

Each prompt has three parts: v1's profile, a line of public evidence, and v1's
question. The evidence line gives:

- the field-wide count of productive and failed public outcomes; and
- the same counts among the 8 public outcomes whose profiles are most similar,
  by Euclidean distance on the nine features.

An agent never sees its own outcome. The readout is v4's prefilled forced
choice: the reply starts with `Answer: **`, and the readout is the Yes-minus-No
logit.

Each step, the LLM is queried for every agent whose output is used. Those are
the agents that have not entered and the failed entrants whose exit depends on
the policy. Productive entrants keep their last output. Queries run one at a
time, so no answer depends on which other prompts share its batch. v4 showed
that batching shifts log-odds by up to 0.75. An identical prompt reuses its
logits, which is exact at batch size 1.

The agent is calibrated with aggregate information only, as the hybrid is. Its
log-odds get one fixed shift, chosen so that the mean probability over the v1
design cohort equals the stated 0.30 base rate when no evidence is public. The
LLM reacts to evidence on its own. At the end, every probe profile is scored
against the final public record, which gives a probe Brier comparable with
the pooled learners'.

This is one reasonable in-context design, not the best possible LLM agent.

## Conditions and decisions

| Condition | Policy | Arms |
|---|---|---|
| `hand` | Hand rule | Fixed rule and field linear, tree, and MLP learners |
| `hybrid_qwen3_4b_2507` | v4 calibrated prior, 20% floor | Same four arms |
| `llm_agent_qwen3_4b_2507` | The LLM at every step, 20% floor | One LLM-agent arm |

The design covers 3 fit worlds and 16 replicates of 256 agents over 18 steps,
with fresh seeds.

In each of the nine combinations of world and NABM field arm:

- **NABM against the LLM agent.** The hybrid matches the LLM agent when its
  mean entrant precision is no more than 0.02 lower and its mean probe Brier no
  more than 0.005 higher. Otherwise the LLM agent is better on precision, on
  learning, or on both. NABM superiority is also flagged, using the paired rule
  with 0.02 precision or 0.01 Brier in at least 12 of 16 replicates.
- **LLM agent against the hand rule**, with v3's dominance rule.

Each is summarized over the nine cells.

Cost is reported, not decided. The report gives GPU seconds and forward passes
per LLM-agent run, and CPU seconds per NABM arm run. The light-model cost floor
multiplies this run's forward passes by v4's batch-32 seconds per query for
Qwen3-0.6B. v4 showed the 0.6B prior to be too weak to stand in for LLM-agent
behavior, so it is not run as an agent. Timing is not byte-reproducible and is
written to a separate file.

## Boundary

The executable must be committed before the run, and a second run must
reproduce `runs.jsonl`, the scorecard, and the receipt byte for byte. A
format-only dry run on 48 disjoint profiles with synthetic evidence text
inspected only answer mass, determinism, and timing. No post-result change to
the prompt, calibration, share, or thresholds is authorized, and no model is
selected.

The worlds are synthetic. The study makes no paid API calls, reads no empirical
rows, adds no LLM dependency to the package, and changes no generic NABM API.
