# Pivot local-LLM prior pilot v3 result

Date: 2026-09-29

Status: `SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED`

## Result

**An exploration floor resolves most of the v2 trade-off.** In v2, the
calibrated Qwen prior picked better entrants than the hand rule but left pooled
learners with less evidence. v3 splits the two roles. The LLM prior decides who
is ready, and a prior-independent floor lets a share of the remaining agents
enter anyway.

**The 20% hybrid dominates the hand rule in 7 of 9 cells.** Its entrants are
more productive than under the hand rule for every world and field arm, by
0.026 to 0.086, in at least 13 of 16 replicates. In seven cells, its pooled
learning is also within the registered 0.005 margin of the hand rule. The
exceptions are the linear learner in the linear and axis worlds, where it is
0.0076 and 0.0053 worse.

**Too little or too much exploration gives up one side.** Without exploration,
the LLM prior still selects better entrants everywhere, but it falls short of
the hand rule's learning in 6 of 9 cells. The 10% hybrid dominates in 5 of 9.
At 40%, learning is non-inferior in 8 of 9 cells, but the precision gain is gone
(−0.003 to +0.019, never passing). The flat prior, which has no knowledge,
dominates nowhere: its entrants are 0.048 to 0.120 less productive than under
the hand rule, in every replicate.

The run used `87c90eb`: 1,152 ABM runs over 6 conditions, 3 fit worlds, 4 arms,
and 16 replicates. It made no LLM calls, reusing the v2 base-rate prior by hash.
Every nesting check passed exactly. Under the hand condition, the fixed arm
matched the persistence predecessor. The prior and adapter slots were restored
after every task, and the LLM prior matched v2 exactly. A second run reproduced
all four artifacts byte for byte.

## Registered decisions

Each condition was compared with the hand rule in each combination of world and
field arm. A cell is dominated when the entrant-precision gain passes and pooled
learning is non-inferior.

| World | Field arm | `llm_selective` | `hybrid_10` | `hybrid_20` | `hybrid_40` | `flat` |
|---|---|---|---|---|---|---|
| Linear | linear | selection only | selection only | selection only | neither | neither |
| Linear | tree | **dominates** | **dominates** | **dominates** | learning only | learning only |
| Linear | MLP | **dominates** | **dominates** | **dominates** | learning only | learning only |
| Smooth | linear | **dominates** | **dominates** | **dominates** | learning only | neither |
| Smooth | tree | selection only | selection only | **dominates** | learning only | learning only |
| Smooth | MLP | selection only | **dominates** | **dominates** | learning only | learning only |
| Axis | linear | selection only | selection only | selection only | learning only | learning only |
| Axis | tree | selection only | selection only | **dominates** | learning only | learning only |
| Axis | MLP | selection only | **dominates** | **dominates** | learning only | learning only |
| **Cells dominated** | | **3 of 9** | **5 of 9** | **7 of 9** | **0 of 9** | **0 of 9** |

The 20% hybrid in detail. Precision gain and learning against the selective
prior are shown as the mean with the number of positive replicates out of 16.
Learning against the hand rule is the mean of hand minus hybrid probe Brier, so
negative means the hybrid learned worse.

| World | Field arm | Precision gain | Learning vs hand | Learning vs selective | Disposition |
|---|---|---:|---:|---:|---|
| Linear | linear | +0.080 (16) | −0.0076 | +0.003 (15) | selection only |
| Linear | tree | +0.064 (16) | +0.0066 | +0.004 (13) | **dominates** |
| Linear | MLP | +0.086 (16) | −0.0014 | +0.001 (11) | **dominates** |
| Smooth | linear | +0.035 (15) | −0.0031 | 0.000 (8) | **dominates** |
| Smooth | tree | +0.039 (16) | −0.0024 | +0.009 (12) | **dominates** |
| Smooth | MLP | +0.044 (15) | +0.0007 | +0.024 (14) | **dominates** |
| Axis | linear | +0.026 (13) | −0.0053 | +0.003 (11) | selection only |
| Axis | tree | +0.050 (16) | −0.0036 | +0.023 (14) | **dominates** |
| Axis | MLP | +0.040 (15) | −0.0029 | +0.022 (14) | **dominates** |

Against the selective prior, the 20% hybrid restores learning in three cells,
all nonlinear learners in the interaction worlds. These are the MLP in both
interaction worlds and the tree in the axis world. In the smooth world, the tree
gains 0.009, just short of the 0.01 threshold.

## Condition profiles

Entry, exploratory entry, entrant precision and evidence are for the pooled MLP
arm. Probe Brier is shown for the pooled MLP and tree learners.

| World | Condition | Entry | Exploratory entry | Entrant precision | Evidence per learner | MLP Brier | Tree Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| Linear | hand | 0.711 | — | 0.418 | 137 | 0.012 | 0.068 |
| Linear | LLM selective | 0.492 | — | 0.607 | 94 | 0.015 | 0.066 |
| Linear | hybrid 10% | 0.543 | 0.054 | 0.549 | 104 | 0.013 | 0.062 |
| Linear | hybrid 20% | 0.591 | 0.106 | 0.504 | 112 | 0.014 | 0.061 |
| Linear | hybrid 40% | 0.691 | 0.209 | 0.431 | 130 | 0.012 | 0.060 |
| Linear | flat | 1.000 | — | 0.299 | 192 | 0.011 | 0.063 |
| Smooth | hand | 0.703 | — | 0.397 | 135 | 0.093 | 0.128 |
| Smooth | LLM selective | 0.481 | — | 0.512 | 92 | 0.116 | 0.139 |
| Smooth | hybrid 10% | 0.573 | 0.057 | 0.470 | 109 | 0.097 | 0.135 |
| Smooth | hybrid 20% | 0.633 | 0.111 | 0.441 | 120 | 0.092 | 0.130 |
| Smooth | hybrid 40% | 0.724 | 0.208 | 0.397 | 137 | 0.084 | 0.126 |
| Smooth | flat | 1.000 | — | 0.299 | 192 | 0.079 | 0.115 |
| Axis | hand | 0.697 | — | 0.382 | 134 | 0.117 | 0.070 |
| Axis | LLM selective | 0.465 | — | 0.482 | 89 | 0.142 | 0.097 |
| Axis | hybrid 10% | 0.570 | 0.059 | 0.451 | 108 | 0.121 | 0.079 |
| Axis | hybrid 20% | 0.615 | 0.112 | 0.422 | 116 | 0.120 | 0.074 |
| Axis | hybrid 40% | 0.726 | 0.205 | 0.386 | 137 | 0.106 | 0.063 |
| Axis | flat | 1.000 | — | 0.299 | 192 | 0.096 | 0.056 |

Realized exploratory entry is about half the targeted share. The target applies
only to agents the prior has not admitted, which is roughly half the population.

## Post-hoc checks

These checks were not registered. They reuse the frozen rows and run nothing new.

**Exploration traces a frontier.** Moving from the selective prior through the
10%, 20% and 40% hybrids, entrant precision falls and evidence per learner rises
in all nine cells. Pooled probe Brier falls in seven of them. The exceptions are
the MLP in the linear world and the linear learner in the smooth world.

**The linear learner's shortfall in the linear world is not only about
evidence.** No condition's linear learner is non-inferior to the hand rule
there. That includes the flat prior, which gives it the most evidence and is
0.0167 worse. The closest is the 40% hybrid, at 0.0062 worse. In the axis world,
the 40% hybrid and the flat prior do pass, at 0.0027 and 0.0042 worse.

## What this means for NABM

- **Selection and exploration are separable roles.** An LLM consulted once and
  calibrated with one aggregate number decides who acts. A cheap,
  prior-independent exploration floor keeps the evidence flowing that pooled
  learners need. Together they beat the hand-written rule on both counts in most
  cells, at no added LLM cost.
- **The exploratory share is a design parameter, not a tuning knob.** It sets a
  point on a precision–evidence frontier. The 20% share dominated in the most
  cells, but no share is selected. It should be chosen for the question a study
  asks, before results are seen.
- **The linear field learner is where the hybrids fall short.** Neither the
  10% nor the 20% hybrid made it non-inferior to the hand rule in the linear or
  axis worlds. Where the floor restores learning against the selective prior,
  it does so for the nonlinear learners in the interaction worlds.

## Boundary

The worlds and the stated base rate are synthetic and stylized. The results
show how the LLM's intuitions line up with the designer's worlds, not knowledge
of real pivots. The pilot made no LLM or paid API calls, read no empirical rows,
selected no share or model, and changed no generic NABM API.
