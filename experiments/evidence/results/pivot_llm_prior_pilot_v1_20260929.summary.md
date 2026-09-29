# Pivot local-LLM prior pilot v1 result

Date: 2026-09-29

Status: `SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED_WITH_REPRODUCED_LOCAL_LLM_ELICITATION`

## Result

**The hand-set prior won every registered decision in every world.** The reason
is informative. The distilled Qwen prior ranks researchers about as well as the
hand rule, and slightly better in the linear world. Its probability level,
however, is far too low. In the prior slot, that level becomes each agent's
readiness target, so no agent ever pivots. With no entrants there is no outcome
evidence, and no learner, local or pooled, can correct the prior.

The failure is one of calibration, not knowledge. It is self-sealing because
Pivot observes outcomes only for agents who enter.

The pilot used `75fbbf7`: 1,024 profiles elicited once from a pinned local
`Qwen/Qwen3-4B-Instruct-2507`, then 960 ABM runs over 4 priors, 3 fit worlds, 5
arms, and 16 replicates. A second elicitation reproduced the records and the
distilled prior exactly. A second ABM run reproduced all four artifacts byte for
byte. Under the hand prior, the fixed and online arms exactly nested the
persistence predecessor.

## Elicitation

| Check | Value |
|---|---|
| Median probability mass on Yes or No | 0.999 |
| Top token is Yes or No | 100% |
| Repeat pass identical | yes |
| Log-odds shift from batching 32 prompts (diagnostic) | up to 0.75 |
| Verbalized readout parse rate | 100% |
| Mean primary P(Yes) | 0.027 |
| Profiles with primary P(Yes) above 0.5 | 2.1% |
| Mean verbalized probability | 0.475 (range 0.20 to 0.72) |
| Rank correlation, primary vs verbalized | 0.87 |
| Distillation R² on log-odds | 0.45 |
| Rank correlation, primary vs distilled | 0.86 |

The model answers "No" to almost every profile with high confidence. When asked
for a number instead, it gives moderate probabilities that order the profiles
almost the same way. First-token Yes/No probabilities from an instruct model
are known to be overconfident. Here they are also strongly pessimistic.

The distilled prior (bias −7.2) keeps some intuitive signs. Network support
(+2.8) and openness (+4.2) help, while skill distance (−4.4) and reputation
risk (−4.3) hurt. Resource security (−1.8) and field opportunity (−1.0) come
out negative, against intuition.

## Registered decisions

| World | Prior quality | Local rescue | Pooled washout | Entry precision |
|---|---|---|---|---|
| Linear | hand better (16/16) | hand better (13/16) | persists, all field arms | hand better (16/16) |
| Smooth | hand better (16/16) | hand better (14/16) | persists, all field arms | hand better (16/16) |
| Axis | hand better (16/16) | hand better (14/16) | persists, all field arms | hand better (16/16) |

Under the LLM prior, entry is 0.000 in every cell. Every arm, learning or not,
stays at the prior's fixed policy, with a probe Brier of about 0.308. The
"persists under pooling" disposition holds in the worst sense: the pooled
learners received no evidence at all.

## Probe outcomes by prior

Probe Brier for the fixed rule and for the best pooled learner in each world,
with the fixed rule's entry rate:

| World | Prior | Fixed probe Brier | Fixed entry | Best pooled probe Brier |
|---|---|---:|---:|---:|
| Linear | hand | 0.202 | 0.817 | 0.011 (MLP) |
| Linear | llm | 0.308 | 0.000 | 0.308 |
| Linear | oracle linear | 0.026 | 0.323 | 0.012 (MLP) |
| Linear | flat | 0.250 | 1.000 | 0.011 (MLP) |
| Smooth | hand | 0.207 | 0.817 | 0.088 (MLP) |
| Smooth | llm | 0.308 | 0.000 | 0.308 |
| Smooth | oracle linear | 0.178 | 0.390 | 0.119 (MLP) |
| Smooth | flat | 0.250 | 1.000 | 0.072 (MLP) |
| Axis | hand | 0.205 | 0.817 | 0.052 (tree) |
| Axis | llm | 0.308 | 0.000 | 0.308 |
| Axis | oracle linear | 0.184 | 0.386 | 0.099 (tree) |
| Axis | flat | 0.250 | 1.000 | 0.044 (tree) |

**Priors shape their own evidence.** The flat prior sends every agent in and
produces the most outcome evidence. Its pooled learners then match or beat
those under every other prior. The oracle-linear prior is the best starting
policy, but it is selective. It admits 32% to 39% of agents, so its pooled
learners see less evidence and end up worse in the interaction worlds. The
prior decides not only where an agent starts but also how much the population
learns.

## Post-hoc checks

These checks were not registered. No new LLM call or ABM run was made.

**No evidence reached the LLM-prior learners.** The maximum entry rate under
the LLM prior was 0, and every field learner received 0 outcome events.

**Ranking and calibration on the design cohort.** Structural labels for each
world were computed on the 1,024 elicitation profiles. The base rate is 0.306
in every world.

| World | Readout | AUC | Mean prediction | Brier |
|---|---|---:|---:|---:|
| Linear | primary | 0.659 | 0.027 | 0.285 |
| Linear | verbalized | 0.579 | 0.475 | 0.239 |
| Linear | distilled | 0.707 | 0.001 | 0.304 |
| Linear | hand | 0.647 | 0.358 | 0.202 |
| Smooth | primary | 0.599 | 0.027 | 0.286 |
| Smooth | verbalized | 0.569 | 0.475 | 0.242 |
| Smooth | distilled | 0.632 | 0.001 | 0.304 |
| Smooth | hand | 0.622 | 0.358 | 0.206 |
| Axis | primary | 0.610 | 0.027 | 0.280 |
| Axis | verbalized | 0.582 | 0.475 | 0.240 |
| Axis | distilled | 0.644 | 0.001 | 0.304 |
| Axis | hand | 0.643 | 0.358 | 0.203 |

Distillation improves ranking over the raw readout by smoothing noisy
judgments. It also worsens calibration. Fitting on log-odds of near-zero
probabilities pushes the mean prediction from 0.027 to 0.001. The LLM's ranking
skill is comparable to the hand rule. What it lacks is the right level.

## Cost ledger

| Item | Measurement |
|---|---|
| Elicitation, one time | model load 28 s, 1,024 primary prompts 104 s, 1,024 generations 271 s, plus a 104 s repeat pass; under 10 GPU minutes |
| One LLM query | 0.102 s alone, 0.031 s in a batch of 32 (RTX 4080) |
| Phase B agent-steps | 4,423,680 |
| Phase B as an LLM agent, one query per agent-step | at least 38 GPU hours batched, 125 GPU hours unbatched |
| Phase B as NABM | 3,206 CPU seconds; 4 minutes wall-clock on 16 workers |
| LLM queries the hybrid used | 3,104 |

The LLM-agent figures are lower bounds. A real LLM agent would also need its
evidence in context and would generate text. The hybrid design is cheap: the
LLM is consulted about 3,000 times rather than about 4.4 million.

## What this means for NABM

- The hybrid is cheap and reproducible. A local model's knowledge enters once,
  hash-bound, and the ABM stays byte-reproducible.
- The hybrid needs a calibration step. The elicited ranking was usable, but its
  level was not. Anything that feeds an adaptive agent's decision threshold
  must be on the decision's scale.
- Under selective observation, a pessimistic prior is self-sealing. This is a
  general ABM lesson, not an LLM one. Knowledge that discourages action also
  prevents the experience that would correct it.
- Exploration matters for pooled learning. An uninformative prior that admits
  everyone generated more evidence than a well-informed selective one.

## Next step

A registered successor should calibrate the elicited prior before it enters the
slot, using aggregate information only. For example, it could keep the
distilled weights and set the intercept so that the mean prior matches a stated
population base rate. It should compare that against the verbalized readout and
the hand rule, and it should report entry and evidence volume alongside
accuracy. No post-result tuning of v1 is authorized.

## Boundary

The fit worlds and the notion of a productive pivot are synthetic. The LLM's
scores show how its intuitions line up with the designer's stylized worlds, not
knowledge of real pivots. The pilot made no paid API calls, read no empirical
rows, selected no model, and changed no generic NABM API.
