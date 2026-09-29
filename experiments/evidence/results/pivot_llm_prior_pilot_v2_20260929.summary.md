# Pivot local-LLM prior pilot v2 result

Date: 2026-09-29

Status: `SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED`

## Result

**Calibration makes the once-elicited LLM prior usable.** Only the
uncalibrated v1 prior is self-sealing, and it is in every world. Once the
same Qwen weights are put on the decision scale using aggregate information
only, agents enter, evidence flows, and the LLM's ranking shows up in
behavior.

**What the calibrated prior buys, and what it costs.** Calibrated to a stated
0.30 base rate, the primary variant admits about a third of agents. Its
entrants are productive 46% to 48% of the time, against 33% under the hand
rule. This holds in every replicate of every world. As a probability forecast,
though, it is no better than the hand rule. Its selectivity also leaves pooled
learners about 30% less evidence, so they end up worse in the interaction
worlds.

**The prior sets the exploration regime.** The verbalized prior admits
everyone. It produces the most evidence and the best pooled nonlinear learning.
It is also the only LLM variant that improves neighbor-scale learning beyond a
no-knowledge prior. Picking the best entrants now and generating evidence for
later learning pull in opposite directions.

The run used `2ddcd93`: 1,680 ABM runs over 7 priors, 3 fit worlds, 5 arms, and
16 replicates. It made no new LLM calls, reusing the v1 elicitation by hash. The
calibrated variants kept the v1 weights exactly and hit their target means. The
uncalibrated variant matched v1 exactly. Under the hand prior, the fixed and
online arms nested the persistence predecessor. A second run reproduced all
five artifacts byte for byte.

## Calibration

| Prior | Mean over the design profiles | Change from v1 |
|---|---:|---|
| v1 uncalibrated | 0.001 | none |
| Base rate (primary) | 0.300 | intercept +6.90 |
| Hand level (control) | 0.358 | intercept +7.29 |
| Verbalized | 0.475 | re-distilled; R² 0.59 on log-odds |
| Hand rule | 0.358 | reference |

## Registered decisions

Each LLM variant is compared with the hand rule. The mean is shown with the
number of positive replicates out of 16.

**Base-rate variant (primary)**

| World | Prior quality | Local rescue | Entry precision | Pooled |
|---|---|---|---|---|
| Linear | no clear (+0.002, 9) | better (+0.023, 16) | better (+0.143, 16) | washes out |
| Smooth | hand better (−0.013) | no clear (+0.016, 11) | better (+0.133, 16) | hand better: MLP, tree |
| Axis | hand better (−0.015) | better (+0.023, 15) | better (+0.127, 16) | hand better: linear, tree |

**Hand-level variant (control)**

| World | Prior quality | Local rescue | Entry precision | Pooled |
|---|---|---|---|---|
| Linear | no clear (−0.004) | better (+0.022, 16) | better (+0.105, 16) | washes out |
| Smooth | hand better (−0.021) | better (+0.020, 12) | better (+0.081, 16) | hand better: linear |
| Axis | hand better (−0.025) | better (+0.022, 15) | better (+0.068, 16) | hand better: linear, tree |

**Verbalized variant**

| World | Prior quality | Local rescue | Entry precision | Pooled |
|---|---|---|---|---|
| Linear | hand better (−0.028) | better (+0.033, 16) | hand better (−0.022) | washes out |
| Smooth | hand better (−0.024) | better (+0.043, 16) | no clear (−0.017) | better: MLP, tree; hand better: linear |
| Axis | hand better (−0.024) | better (+0.046, 16) | no clear (−0.014) | better: MLP, tree |

**Uncalibrated variant:** hand better on every decision in every world, and
self-sealing in all three worlds, reproducing v1.

## Prior profiles

Fixed-rule entry and entrant precision, and the evidence and probe Brier of the
pooled MLP learner:

| World | Prior | Entry | Entrant precision | Evidence per learner | Pooled MLP Brier |
|---|---|---:|---:|---:|---:|
| Linear | hand | 0.823 | 0.336 | 138 | 0.012 |
| Linear | base rate | 0.331 | 0.479 | 95 | 0.014 |
| Linear | hand level | 0.465 | 0.441 | 109 | 0.013 |
| Linear | verbalized | 1.000 | 0.314 | 192 | 0.012 |
| Linear | oracle linear | 0.337 | 0.921 | 85 | 0.017 |
| Linear | flat | 1.000 | 0.314 | 192 | 0.012 |
| Smooth | hand | 0.823 | 0.331 | 137 | 0.086 |
| Smooth | base rate | 0.331 | 0.464 | 93 | 0.108 |
| Smooth | hand level | 0.465 | 0.412 | 108 | 0.099 |
| Smooth | verbalized | 1.000 | 0.314 | 192 | 0.067 |
| Smooth | oracle linear | 0.382 | 0.513 | 80 | 0.115 |
| Smooth | flat | 1.000 | 0.314 | 192 | 0.068 |
| Axis | hand | 0.823 | 0.328 | 138 | 0.121 |
| Axis | base rate | 0.331 | 0.455 | 96 | 0.144 |
| Axis | hand level | 0.465 | 0.396 | 110 | 0.133 |
| Axis | verbalized | 1.000 | 0.314 | 192 | 0.098 |
| Axis | oracle linear | 0.377 | 0.482 | 88 | 0.144 |
| Axis | flat | 1.000 | 0.314 | 192 | 0.098 |

At a similar entry rate, the calibrated LLM prior selects entrants in the
interaction worlds nearly as well as the oracle linear prior, which was fitted
to those worlds' labels (0.46 versus 0.51 smooth, 0.46 versus 0.48 axis). In
the linear world the oracle is far better, at 0.92.

## Post-hoc checks

These checks were not registered. They reuse the frozen rows and run nothing new.

**Most of the local rescue comes from escaping the hand prior, not from the
LLM.** Online-logistic probe Brier improvement over the flat, no-knowledge
prior:

| World | Hand | Base rate | Hand level | Verbalized |
|---|---:|---:|---:|---:|
| Linear | −0.021 (0) | +0.002 (7) | +0.001 (8) | +0.013 (16) |
| Smooth | −0.030 (0) | −0.014 (4) | −0.010 (5) | +0.013 (16) |
| Axis | −0.032 (0) | −0.009 (3) | −0.010 (5) | +0.013 (16) |

The hand pressure rule is a worse starting point for neighbor-scale learning
than no prior at all. The calibrated LLM variants escape that handicap but add
nothing beyond flat. Only the verbalized prior carries knowledge that helps
local learning, by 0.013, in every replicate of every world.

**As static forecasts, all LLM variants carry information.** Every variant beats
the flat prior's fixed-rule probe Brier by 0.017 to 0.049, mostly in 16 of 16
replicates. The base-rate variant matches the hand rule in the linear world.

**Evidence drives pooled nonlinear learning.** Across priors, the rank
correlation between evidence volume and pooled probe Brier is −0.9 for the MLP
in every world. For the tree it is −1.0 in the smooth world and −0.9 in the
axis world. More evidence means more accurate pooled learning.

**The hand-level control did not isolate ranking.** Matching the hand rule's
mean did not equalize entry: 0.465 against 0.823. The LLM prior is more
dispersed, so fewer agents cross the pivot threshold at the same mean. Its
entrant-precision gain therefore mixes ranking with selectivity.

## What this means for NABM

- **The hybrid works once it is calibrated.** An LLM consulted once, then
  calibrated with one aggregate number, drives a reproducible ABM whose agents
  select better entrants than the hand-written rule.
- **The prior's level sets the exploration regime.** A selective,
  knowledge-driven prior exploits: it picks better entrants but starves pooled
  learning of evidence. A permissive prior explores: it produces the evidence
  pooled learners need.
- **LLM knowledge pays most at the point of action.** It pays for choosing who
  acts now, and modestly for local learning through the verbalized readout. It
  does not pay for pooled learning, which needs exploration more than prior
  knowledge.
- **The next design is suggested by the trade-off.** Separate selection from
  exploration, for example with an LLM prior plus an exploration floor, so
  knowledge picks entrants while a minimum share of agents keeps generating
  evidence.

## Boundary

The worlds and the stated base rate are synthetic and stylized. The results
show how the LLM's intuitions line up with the designer's worlds, not knowledge
of real pivots. The pilot made no LLM or paid API calls, read no empirical rows,
selected no model, and changed no generic NABM API.
