# Pivot LLM-agent reference v1 result (N06)

Date: 2026-09-30

Status: `SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED_WITH_A_LOCAL_LLM_AT_EVERY_AGENT_STEP`

## Result

**Consulting the LLM once did at least as well as asking it at every step.**
Both designs used the same local Qwen3-4B-Instruct-2507 and the same 20%
exploration floor. The NABM hybrid distils the LLM into the prior slot once,
then learns from pooled outcomes. It matched the LLM agent in all nine
combinations of world and field arm. It was also superior on learning in all
nine and on entrant precision in eight.

**The LLM agent did not beat the hand rule anywhere.** Its entrants were more
productive in two cells, but its probe forecasts were worse than the
hand-rule learners' in every cell.

**Its weakness was calibration, not ranking.** Once outcomes were public, the
LLM agent ranked probe profiles about as well as the pooled nonlinear learners
(AUC 0.89 to 0.97). Its probabilities, however, rose well above the population's
productive rate, so it admitted too many agents and forecast poorly.

The run used `2a25ab5`: 432 runs, covering 2 NABM conditions with 4 arms each
and 1 LLM-agent arm, in 3 fit worlds with 16 replicates of 256 agents. The
LLM-agent arm made 118,772 forward passes. A second execution reproduced every
reproducible artifact byte for byte. Only the separate timing file differed,
as registered. Every nesting check passed exactly, including one confirming that
the LLM-agent machinery, given the agent's own hand policy, reproduces the
fixed-rule arm in every linear-world replicate. Repeated LLM forward passes were
identical. No paid API was called.

## Registered decisions

"Matches" means the hybrid's mean entrant precision is at most 0.02 below the
LLM agent's and its mean probe Brier at most 0.005 above. "NABM superior" uses
the paired rule: 0.02 precision or 0.01 Brier, in at least 12 of 16
replicates. The last column applies v3's dominance rule to the LLM agent
against the same hand-rule arm.

| World | NABM arm | Precision, NABM − LLM | Brier, LLM − NABM | NABM vs LLM agent | NABM superior | LLM agent vs hand |
|---|---|---:|---:|---|---|---|
| Linear | linear | +0.063 | +0.073 | matches | precision, learning | neither |
| Linear | tree | +0.045 | +0.073 | matches | precision, learning | selection only |
| Linear | MLP | +0.108 | +0.127 | matches | precision, learning | neither |
| Smooth | linear | +0.025 | +0.052 | matches | precision, learning | neither |
| Smooth | tree | +0.049 | +0.096 | matches | precision, learning | neither |
| Smooth | MLP | +0.076 | +0.131 | matches | precision, learning | neither |
| Axis | linear | +0.006 | +0.050 | matches | learning | selection only |
| Axis | tree | +0.084 | +0.166 | matches | precision, learning | neither |
| Axis | MLP | +0.068 | +0.108 | matches | precision, learning | neither |

In the axis world against the linear learner, the hybrid's precision edge is
+0.006 and positive in only 10 of 16 replicates, so it is not superior there.
The LLM agent was selection-only in two cells, with precision gains of 0.022
and 0.027 over the hand rule. Its probe Brier was worse than the hand-rule
learners' by 0.050 to 0.169 in every cell.

## Condition profiles

"Evidence" is the number of outcome events per learner. For the LLM agent, it
is the number of public outcomes in its final prompt record.

| World | Condition | Arm | Entry | Entrant precision | Probe Brier | Probe AUC | Evidence |
|---|---|---|---:|---:|---:|---:|---:|
| Linear | hand | fixed rule | 0.828 | 0.326 | 0.200 | 0.660 | — |
| Linear | hand | mlp | 0.717 | 0.423 | 0.013 | 0.999 | 138 |
| Linear | NABM hybrid | fixed rule | 0.485 | 0.404 | 0.186 | 0.706 | — |
| Linear | NABM hybrid | linear | 0.616 | 0.486 | 0.066 | 0.984 | 117 |
| Linear | NABM hybrid | tree | 0.644 | 0.468 | 0.067 | 0.980 | 123 |
| Linear | NABM hybrid | mlp | 0.570 | 0.531 | 0.013 | 0.999 | 108 |
| Linear | LLM agent | llm agent | 0.719 | 0.422 | 0.140 | 0.973 | 183 |
| Smooth | hand | fixed rule | 0.828 | 0.326 | 0.203 | 0.651 | — |
| Smooth | hand | mlp | 0.715 | 0.398 | 0.084 | 0.957 | 137 |
| Smooth | NABM hybrid | fixed rule | 0.485 | 0.372 | 0.197 | 0.653 | — |
| Smooth | NABM hybrid | linear | 0.571 | 0.401 | 0.174 | 0.730 | 108 |
| Smooth | NABM hybrid | tree | 0.649 | 0.425 | 0.130 | 0.904 | 123 |
| Smooth | NABM hybrid | mlp | 0.616 | 0.452 | 0.095 | 0.935 | 117 |
| Smooth | LLM agent | llm agent | 0.800 | 0.376 | 0.226 | 0.910 | 204 |
| Axis | hand | fixed rule | 0.828 | 0.323 | 0.205 | 0.626 | — |
| Axis | hand | mlp | 0.718 | 0.389 | 0.120 | 0.921 | 138 |
| Axis | NABM hybrid | fixed rule | 0.485 | 0.359 | 0.199 | 0.632 | — |
| Axis | NABM hybrid | linear | 0.526 | 0.380 | 0.186 | 0.688 | 99 |
| Axis | NABM hybrid | tree | 0.640 | 0.458 | 0.069 | 0.956 | 122 |
| Axis | NABM hybrid | mlp | 0.610 | 0.442 | 0.127 | 0.902 | 116 |
| Axis | LLM agent | llm agent | 0.803 | 0.374 | 0.235 | 0.892 | 204 |

The LLM agent was calibrated like the hybrid. One log-odds shift of +12.79 set
its mean probability to 0.30 over the v1 design cohort with no evidence public.
Before calibration, its mean was 0.0035.

## Cost

The LLM agent needed a median of 110 GPU-seconds per run, measured the same in
both executions. That comes from about 2,500 forward passes at 0.044 seconds each.
Over the 48 runs it took 1.45 GPU hours. An NABM arm run needed a median of 4.8
single-core CPU-seconds in the development execution and 5.2 in the
reproduction, averaged over the fixed, linear, tree, and MLP arms. The hybrid's
LLM cost is a single 49-GPU-second elicitation from v4, reused by every run.

The two figures come from different hardware, so their ratio (about 23) is
indicative. At v4's batch-32 speeds, the same number of queries would take
75 GPU-seconds per run with the 4B model and 16 with Qwen3-0.6B. Even the
0.6B floor is about three times a NABM arm's CPU time per run, and every run
needs it again.

Of the 4,608 agent-steps in a run, a mean of 3,773 needed an LLM output. The
exact prompt cache cut the forward passes, probe included, by 38%.

## Post-hoc checks

These checks were not registered. They reuse the frozen rows. The population
rate was recomputed from the seeded population draws, with no ABM execution.

**Calibration drifts once evidence enters the prompt.** The mean policy output
across agents:

| World | LLM agent, no evidence | LLM agent, after evidence | Final | NABM MLP | Public success share | Population rate |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 0.32 | 0.41 to 0.50 | 0.44 | 0.30 to 0.31 | 0.43 | 0.305 |
| Smooth | 0.32 | 0.47 to 0.54 | 0.53 | 0.28 to 0.31 | 0.38 | 0.305 |
| Axis | 0.32 | 0.39 to 0.54 | 0.54 | 0.24 to 0.31 | 0.38 | 0.305 |

The calibration shift was set with no evidence, and it does not hold once
evidence is in the prompt. In the interaction worlds, the LLM agent's final mean
output also exceeds the final public success share of 0.38. The pooled learners are
fitted to outcomes under log loss, so they stay near the population rate.

**Ranking in context is strong.** The LLM agent's probe AUC was 0.973, 0.910,
and 0.892 in the linear, smooth, and axis worlds. The hybrid's tree and MLP
reached 0.980 and 0.999, 0.904 and 0.935, and 0.956 and 0.902. Its linear
learner reached only 0.730 and 0.688 in the interaction worlds. One caveat
applies. The prompt's counts over the 8 most similar outcomes are themselves a
nearest-neighbour estimate, and this study does not separate how much ranking
the LLM adds to them.

## What this means for NABM

- **The LLM-agent end of the positioning is now measured, in this design.**
  With the same model, the in-context LLM agent cost more per run and did not
  do better. The NABM hybrid got the LLM's knowledge for one elicitation and
  matched or beat the LLM agent everywhere.
- **Pooled learning is what keeps NABM calibrated.** The LLM's ranking in
  context was good, but its probabilities drifted. The hybrid's learners are
  fitted to outcomes, which keeps their forecasts on the decision scale.
- **A stronger LLM agent is conceivable.** For example, it could be
  recalibrated online or reason over individual cases. That is worth building
  only if a research question needs it. This design is one reasonable
  in-context agent, not the best possible one.

## Boundary

The worlds, the stated base rate, and the evidence summary are synthetic and
stylized. The results compare two ways of using the same LLM on the designer's
worlds. They do not describe real researchers. The study made no paid API
calls, read no empirical rows, selected no model, added no LLM dependency to
the package, and changed no generic NABM API.
