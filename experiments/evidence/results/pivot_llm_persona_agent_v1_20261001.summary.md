# Pivot persona LLM agent v1 result (N07)

Date: 2026-10-01

Status: `SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED_WITH_A_PERSONA_LLM_AT_EVERY_AGENT_STEP`

## Result

**The persona agent almost never chose to pivot.** The same local
Qwen3-4B-Instruct-2507 role-played each researcher and answered "Do you pivot
into this field now?" at every step. Across 176,578 questions it said Yes 267
times, or 0.15%. With no evidence public, it said No to every profile. Of the
1,492 failed entrants it asked, none chose to stay. Nearly all entry came from
the 20% exploration floor. The persona sealed itself in the two interaction
worlds, where decided entry was 0.4% and 0.5%, and came close in the linear
world, at 5.6%.

**The NABM hybrid matched it in all 9 cells.** The hybrid selected better
entrants in eight cells and ranked researchers better in seven. In the other
two cells the persona's AUC was higher by 0.013 or less, within the margin. The
persona's entrants were less productive than the hand rule's in every cell, so
it selected better than the hand rule nowhere.

This mirrors the first LLM prior pilot. An LLM used on its raw scale is so
pessimistic that it almost closes off the experience that would correct it.
NABM corrects the level once, with one aggregate number, and then learns from
outcomes.

The run used `150486a`: 432 runs, covering 2 NABM conditions with 4 arms each
and 1 persona arm, in 3 fit worlds with 16 replicates of 256 agents. A second
execution reproduced every reproducible artifact byte for byte. Only the
timing file differed, as registered. Every nesting check passed exactly,
including one confirming that the decision machinery, without a persona,
reproduces the fixed-rule arm in every linear-world replicate. No paid API was
called.

## Registered decisions

"Matches" means the hybrid's mean entrant precision and probe AUC are each at
most 0.02 below the persona's. "NABM superior" uses the paired rule: 0.02 in at
least 12 of 16 replicates. The persona states willingness rather than a
forecast, so ranking is scored by AUC instead of Brier. The last column is the
persona's precision gain over the hand rule, with the number of positive
replicates out of 16.

| World | NABM arm | Precision, NABM − persona | AUC, NABM − persona | NABM superior | Persona precision vs hand |
|---|---|---:|---:|---|---:|
| Linear | linear | +0.102 | +0.024 | precision, ranking | −0.009 (8) |
| Linear | tree | +0.070 | +0.022 | ranking | −0.007 (8) |
| Linear | MLP | +0.146 | +0.041 | precision, ranking | −0.034 (7) |
| Smooth | linear | +0.079 | −0.008 | precision | −0.032 (5) |
| Smooth | tree | +0.093 | +0.175 | precision, ranking | −0.049 (5) |
| Smooth | MLP | +0.128 | +0.217 | precision, ranking | −0.057 (4) |
| Axis | linear | +0.077 | −0.013 | precision | −0.049 (4) |
| Axis | tree | +0.154 | +0.262 | precision, ranking | −0.091 (2) |
| Axis | MLP | +0.136 | +0.212 | precision, ranking | −0.073 (2) |

## Decisions and self-sealing

| World | Yes to pivoting | Runs with any Yes | Stayed after failure | Self-sealing |
|---|---:|---:|---:|---|
| Linear | 231 of 58,331 (0.40%) | 10 of 16 | 0 of 513 | no (0.056) |
| Smooth | 15 of 59,140 (0.03%) | 6 of 16 | 0 of 481 | **yes** (0.004) |
| Axis | 21 of 59,107 (0.04%) | 4 of 16 | 0 of 498 | **yes** (0.005) |

"Self-sealing" means the persona's own decisions admitted fewer than 5% of
agents. The values in parentheses are the mean shares entering by decision.

## Condition profiles

| World | Condition | Arm | Entry | Decided entry | Entrant precision | Probe AUC | Failed exits |
|---|---|---|---:|---:|---:|---:|---:|
| Linear | hand | fixed rule | 0.805 | — | 0.322 | 0.656 | 0.486 |
| Linear | hand | mlp | 0.710 | — | 0.422 | 0.999 | 0.395 |
| Linear | NABM hybrid | tree | 0.651 | — | 0.458 | 0.981 | 0.296 |
| Linear | NABM hybrid | mlp | 0.562 | — | 0.535 | 0.999 | 0.234 |
| Linear | persona agent | persona agent | 0.250 | 0.056 | 0.388 | 0.958 | 0.125 |
| Smooth | hand | fixed rule | 0.805 | — | 0.323 | 0.638 | 0.499 |
| Smooth | hand | mlp | 0.684 | — | 0.388 | 0.934 | 0.375 |
| Smooth | NABM hybrid | tree | 0.637 | — | 0.424 | 0.903 | 0.316 |
| Smooth | NABM hybrid | mlp | 0.596 | — | 0.459 | 0.945 | 0.277 |
| Smooth | persona agent | persona agent | 0.202 | 0.004 | 0.331 | 0.728 | 0.117 |
| Axis | hand | fixed rule | 0.805 | — | 0.325 | 0.617 | 0.493 |
| Axis | hand | mlp | 0.698 | — | 0.386 | 0.906 | 0.375 |
| Axis | NABM hybrid | tree | 0.618 | — | 0.467 | 0.956 | 0.302 |
| Axis | NABM hybrid | mlp | 0.594 | — | 0.449 | 0.906 | 0.272 |
| Axis | persona agent | persona agent | 0.203 | 0.005 | 0.313 | 0.694 | 0.122 |

In the linear world, the persona's willingness still ranked probe researchers
well (AUC 0.958), but in the interaction worlds it did so only at 0.728 and
0.694.

## Cost

The persona needed a median of 205 GPU-seconds per run
(233 in the reproduction), from about
3,500 forward passes. That is more than N06's forecaster needed, because
almost no agent ever entered and stopped being asked. An NABM arm run needed a
median of 5.7 single-core CPU-seconds
(4.2 in the reproduction). The hardware differs, so
the ratio is indicative.

## What this means for NABM

- **Both common LLM-agent designs are now measured.** The forecaster (N06) and
  the persona (this study) were each matched or beaten by the NABM hybrid in 9
  of 9 cells, using the same model.
- **The failure modes differ, and the fix is the same.** The forecaster drifted
  upward once evidence entered its prompt. The persona stayed at the LLM's own
  extreme caution. In both cases the LLM's raw scale became behavior. NABM uses
  the LLM's ranking, sets the level once from an aggregate number, and keeps
  it calibrated by learning from outcomes.
- **This is one persona design.** Decisions were greedy and uncalibrated. A
  sampled or recalibrated persona would behave differently.

## Boundary

The worlds, prompts, and evidence summary are synthetic and stylized. The
results compare ways of using the same LLM on the designer's worlds, not real
researchers. The study made no paid API calls, read no empirical rows, selected
no model, added no LLM dependency to the package, and changed no generic NABM
API.
