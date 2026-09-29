# Pivot relative delay and cost v1 result

Date: 2026-09-29

Status: `SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED_WITH_SEPARATE_COST_LEDGER`

## Result

**The saturation boundary moves with the entry wave.** When entry is spread
over more steps, learned policy precision keeps changing who enters under
longer failure delays. The entry channel was active at these delays:

| Hazard `q` | Active delays |
|---:|---|
| 1.0 | 0 |
| 0.5 | 0, 1 |
| 0.25 | 0, 1, 2, 4 |

**The registered test was `NEITHER_SIMPLE_RULE`.** No single threshold on delay
relative to the fixed rule's entry-wave width separated active from inactive
cells. The fixed rule is the wrong normalizer: it has a slow entry tail, while
learning arms rush in once successes are public. Post hoc, two other
normalizations separate all twelve cells, but only by thin margins. They are
hypotheses for a registered successor, not findings.

**Pooled learning is cheap. Per-agent learning is not.** Field-wide learned
policies cost about 2 to 7 times the rule-based ABM per agent-step.
Per-agent local learners cost about 60 times. They are also the ones the
previous study found worse than the prior.

The run used `22605d5`: 2,304 runs over 3 hazards, 4 failure delays, 3 fit
worlds, and 4 arms, with 16 replicates of 256 agents over 24 steps. At
`q = 1`, the fixed arm exactly nested the persistence predecessor. Under
immediate visibility, the MLP field arm also exactly nested the v2 arm. A
second execution reproduced all four scorecard artifacts byte for byte.

## Registered cells

The entry-channel test is the v2 rule. The tree or MLP must beat the linear
policy on productive share of entrants by at least 0.02 more than it does in
the linear world, in at least 12 of 16 replicates. Relative delay is
(delay + 1) divided by the fixed rule's mean entry-wave width for that `q`.

| q | Delay | Relative delay | Status | Tree, axis | Tree, smooth | MLP, smooth |
|---:|---:|---:|---|---:|---:|---:|
| 1.0 | 0 | 0.163 | active | 0.053 (16) | 0.030 (16) | 0.009 (11) |
| 1.0 | 1 | 0.327 | saturated | 0.000 (0) | 0.000 (0) | 0.000 (0) |
| 1.0 | 2 | 0.490 | saturated | 0.000 (0) | 0.000 (0) | 0.000 (0) |
| 1.0 | 4 | 0.816 | saturated | 0.000 (0) | 0.000 (0) | 0.000 (0) |
| 0.5 | 0 | 0.126 | active | 0.085 (16) | 0.031 (16) | 0.000 (8) |
| 0.5 | 1 | 0.252 | active | 0.025 (16) | 0.012 (14) | 0.009 (15) |
| 0.5 | 2 | 0.378 | absent | 0.010 (16) | 0.005 (13) | 0.003 (11) |
| 0.5 | 4 | 0.630 | saturated | 0.002 (12) | 0.001 (7) | -0.000 (6) |
| 0.25 | 0 | 0.083 | active | 0.111 (16) | 0.035 (13) | -0.021 (4) |
| 0.25 | 1 | 0.166 | active | 0.087 (16) | 0.050 (16) | 0.032 (15) |
| 0.25 | 2 | 0.249 | active | 0.048 (16) | 0.026 (16) | 0.013 (14) |
| 0.25 | 4 | 0.415 | active | 0.022 (16) | 0.012 (14) | 0.008 (14) |

Cells show the DiD mean with positive replicates out of 16 in parentheses. The
tree passes in every active cell. The MLP passes only once, on the smooth world
at `q = 0.25` with a one-step delay. This matches v2: only the tree's entry
gains are interaction-specific.

The test fails because q = 0.25 with a four-step delay is active at a relative
delay of 0.415. That is above both q = 1 with a one-step delay (0.327, saturated)
and q = 0.5 with a two-step delay (0.378, absent).

## Post-hoc checks

These checks were not registered. They reuse the frozen rows and run nothing
new. Both normalizations were chosen after the results were opened.

| q | Delay | Feature-arm wave width | Delay relative to it | Post-failure entry share | Status |
|---:|---:|---:|---:|---:|---|
| 1.0 | 0 | 1.85 | 0.539 | 0.155 | active |
| 1.0 | 1 | 2.00 | 1.000 | 0.000 | saturated |
| 1.0 | 2 | 2.00 | 1.500 | 0.000 | saturated |
| 1.0 | 4 | 2.00 | 2.500 | 0.000 | saturated |
| 0.5 | 0 | 4.48 | 0.223 | 0.434 | active |
| 0.5 | 1 | 3.47 | 0.577 | 0.282 | active |
| 0.5 | 2 | 4.00 | 0.750 | 0.147 | absent |
| 0.5 | 4 | 4.06 | 1.233 | 0.039 | saturated |
| 0.25 | 0 | 10.25 | 0.098 | 0.659 | active |
| 0.25 | 1 | 6.89 | 0.290 | 0.502 | active |
| 0.25 | 2 | 6.62 | 0.453 | 0.373 | active |
| 0.25 | 4 | 7.02 | 0.712 | 0.197 | active |

The feature-arm wave width is the mean for the linear, tree, and MLP field
arms. The post-failure entry share is the fraction of their final entry that
happened at or after the step when the first failure became public.

**Normalizing by the learning arms' own wave separates the cells.** Every
active cell is at or below 0.712, and every inactive cell is at or above 0.750.

**So does the post-failure entry share.** Every active cell is at or above
0.155, and every inactive cell is at or below 0.147. The mechanism reading is
direct. Learned precision can change who enters only when about 15% or more
of entrants decide after failure evidence is public.

Both margins are thin, and both quantities depend on the arms' own behavior,
so neither is a confirmed threshold. The post-failure entry share is the more
useful candidate. Empirically, it maps to the share of topic entry that occurs
after persistence decline became observable. That makes it a natural
discriminant to register for T01.

## Timing observables

| q | Runs peaking at the first open step | Runs with a decline onset | Median model H3 gap |
|---:|---:|---:|---:|
| 1.0 | 96.5% | 86.7% | -1 |
| 0.5 | 57.8% | 88.2% | -1 |
| 0.25 | 43.5% | 88.4% | 0 |

Entry friction unpins the entry peak in about half the runs. In the model, the
entry peak comes one step before, or at, the onset of cohort persistence
decline. These values are descriptive. A full H3 comparison still needs an
attention-surge clock that produces interior peaks by design.

## Cost ledger

One development-size task (256 agents, 24 steps, immediate visibility, axis
world, `q = 1`) was timed per arm. Timings are single-thread process CPU time,
the median of three repeats. Machine load was 33 to 47 during the benchmark, so
timings are noisy and indicative.

| Arm | Evidence | CPU s per run | ms per agent-step | × rule-based |
|---|---|---:|---:|---:|
| Fixed rule | none | 0.91 | 0.149 | 1.0 |
| Online logistic | neighbor | 1.05 | 0.171 | 1.1 |
| Scalar | neighbor | 1.20 | 0.195 | 1.3 |
| Linear | neighbor | 51.72 | 8.417 | 56.6 |
| Tree | neighbor | 59.04 | 9.609 | 64.6 |
| MLP | neighbor | 55.76 | 9.076 | 61.0 |
| Scalar | field | 0.72 | 0.117 | 0.8 |
| Linear | field | 1.72 | 0.279 | 1.9 |
| Tree | field | 6.11 | 0.995 | 6.7 |
| MLP | field | 3.34 | 0.543 | 3.7 |

The fixed rule's 0.15 ms per agent-step is the NABM lifecycle itself: local
update, typed social mixing, and domain commit. Pooled learners add a few
refits per step for the whole field. Per-agent learners refit one model per
agent whenever its neighbors' outcomes arrive.

**LLM reference.** An LLM-agent version of this ABM would query a policy once
per agent-step. That is 6,144 queries per run, of which about 2,200 to 2,600
are decisions by agents who have not yet entered. For this study's 2,304 runs,
it comes to 14.2 million queries, 4.76 million of them pre-entry. No price or
latency is assumed. Those depend on the provider and model.

## What this means for NABM

- The middle ground is real in cost: pooled learned policies cost a small
  multiple of a rule-based ABM. At one policy query per agent-step, an LLM
  version of the same paired design needs millions of calls.
- Cost and usefulness point the same way. The expensive design, a separate
  small learner for every agent, is also the one that fails to learn.
- For the Pivot case, whether learning can change entry depends on how much
  entry happens after failure becomes visible. That is the quantity the
  empirical stage should measure.

## Boundary

This scorecard uses deterministic synthetic ABM runs, plus a separate
timing benchmark that is not byte-reproducible. It reads zero empirical rows,
performs no empirical fit, changes no generic NABM API, selects no model,
establishes no neural necessity or scientific effect, and authorizes no package
promotion.
