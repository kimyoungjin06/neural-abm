# Pivot persona LLM agent v1 (N07)

Date: 2026-09-30

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_llm_persona_agent_v1.yaml`](../../experiments/configs/pivot_llm_persona_agent_v1.yaml)

## Why this study exists

[N06](../../experiments/evidence/results/pivot_llm_agent_reference_v1_20260930.summary.md)
compared NABM with an LLM agent that forecast success inside the ABM's decision
rule. The NABM hybrid matched or beat it in 9 of 9 cells. Typical LLM-agent
simulations work differently: the LLM role-plays each agent and decides. This
study compares the NABM hybrid with that design (open question O14 in
[consolidation v3](pivot-policy-ladder-consolidation-v3.md)).

## The persona agent

The same Qwen3-4B-Instruct-2507 role-plays each researcher. Its prompt is
written in the second person and has four parts:

- the researcher's nine attributes;
- the public evidence, as in N06: field-wide productive and failed counts, and
  the counts among the 8 most similar profiles;
- how many of the researcher's closest collaborators have already pivoted. This
  stands in for the social influence that NABM agents get through readiness
  mixing; and
- the question: "Do you pivot into this field now? Answer with Yes or No."

The persona decides for itself, greedily: it pivots when the "Yes" logit
exceeds the "No" logit. This is the temperature-zero standard for reproducible
LLM agents; sampled decisions are not run. The persona is not calibrated,
because in this design the LLM's own decision is the behavior.

A failed entrant past the minimum active period is asked, in the same voice,
whether it stays in the field. It exits when it says no. The exploration floor
is the same 20% as in the hybrid, so a researcher who says no can still enter
through the prior-independent draw.

The persona sits in the fixed-rule lifecycle through a restored adapter slot,
whose entry and exit decisions are hooks. A registered nesting check confirms,
in every linear-world replicate, that with no persona and no exploration the
machinery reproduces the fixed-rule arm exactly. Queries run at batch size 1,
with an exact prompt cache.

## Decisions

The conditions are `hand`, `hybrid_qwen3_4b_2507` (the v4 prior with the 20%
floor), and `persona_agent_qwen3_4b_2507`. The design covers 3 fit worlds and 16
replicates of 256 agents, with fresh seeds.

A persona states willingness, not a forecast. Its ranking of researchers is
therefore scored by probe AUC, which is scale-free, instead of by Brier. In each
of the nine combinations of world and NABM field arm:

- **NABM against the persona.** The hybrid matches the persona when its mean
  entrant precision is no more than 0.02 lower and its mean probe AUC no more
  than 0.02 lower. NABM superiority is flagged with the paired rule: 0.02 in at
  least 12 of 16 replicates, for either metric.
- **Persona precision against the hand rule**, with the paired rule at 0.02.

Self-sealing is flagged in a world when entry by the persona's own decisions,
excluding exploratory entrants, averages below 5%. Entry, exits, the counts of
"Yes" to pivoting and "No" to staying, and the share of "Yes" answers with no
evidence are all reported. Cost goes in a separate timing file.

## Boundary

The executable must be committed before the run, and a second run must
reproduce `runs.jsonl`, the scorecard, and the receipt byte for byte. A
format-only dry run on 48 disjoint profiles inspected only answer mass and
determinism, for both questions. No post-result change to the prompts, decision
rule, share, or thresholds is authorized, and no model is selected.

This is one reasonable persona design with greedy decisions, not every persona
agent. The worlds are synthetic. The study makes no paid API calls, reads no
empirical rows, adds no LLM dependency to the package, and changes no generic
NABM API.
