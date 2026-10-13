# Information-spread policy ladder: design brief v1 (N13, N14)

Date: 2026-10-13

Status: `DRAFT_DESIGN_NOT_FROZEN_NO_EXECUTION`

This brief proposes the second domain for the Paper-1 central claim. Nothing in
it is registered. Thresholds, sizes, and cells are proposals to be fixed in a
later precommit, after the feasibility and power stages below.

## Why this study

The central claim set on 2026-10-13
([consolidation v5](pivot-policy-ladder-consolidation-v5.md),
[claim matrix](../../paper/claim-matrix.md)) is that NABM separates policy
learning from social interaction. Fixed policies, adaptive policies, and
external priors can therefore be compared and traced within one simulation
lifecycle. Three gaps remain.

1. **Reusability (N14).** The fixed / adaptive / prior comparison exists in
   one domain, Pivot, in repository-only scripts. The bridge rule requires a
   second consumer before any of its semantics enter the package.
2. **A stronger LLM comparator (N13).** The
   [LLM-ABM survey](../../paper/related-work/llm-abm-frameworks-v1.md)
   identifies AgentTorch-style archetypes as the closest published alternative
   to a once-elicited prior. Archetypes query an LLM per archetype at each
   step, without outcome learning.
3. **The precomputed-table critique.** Every seeded run is a deterministic
   ABM. A frozen learned policy is a rule table, and a once-elicited prior is a
   table computed before the run. NABM differs from a precomputed table only
   when the policy depends on the run's own outcome history (closed loop).
   - In Pivot, the useful learner was one pooled policy, which is
     table-like.
   - N11 could not separate closed-loop learning from open-loop anticipation
     at the measured data shape.
   - The original NABM design aimed at models complex enough that policies
     must be computed during the run. No study has yet tested where that
     regime begins.

This study addresses all three in one design. Its question is when an
adaptive NABM policy differs from the best precomputed table, and what that
difference costs.

## The world: delayed-veracity information spread

The domain extends Toy 5 (contagion and adoption), which already provides:

- the network;
- exposure counting;
- the absorbing threshold rule;
- the verified Granovetter reduction (`docs/classical-reductions.md`).

Toy 5's neural policy currently learns from a utility proxy derived from the
threshold. It has no outcome channel. The outcome channel below is new.

**Agents and network.**
- N agents on a Watts–Strogatz graph (k = 6, p = 0.1, the Toy 5 default).
- A scale-free variant serves as a robustness cell.

**Items.**
- Each step, M new items enter. Each item is shared first by a few randomly
  chosen seed agents.
- Item j has observable features z_j: four scores in [0, 1], for example
  source credibility, novelty, emotional intensity, and topical fit.
- Its latent veracity is drawn as y_j ~ Bernoulli(f(z_j)).
- f follows the Pivot geometry families: linear, smooth interaction, and
  axis-aligned interaction.
- A null world sets f constant, so features carry no information and learning
  cannot help.

**Decision.**
- When a neighbor shares item j, agent i is exposed and decides once whether
  to share it. The decision is binary and absorbing per item, as Toy 5
  adoption is.
- The observation has three parts:
  - z_j;
  - exposure features: the fraction and count of neighbors sharing j;
  - in Regime B only, the track record of the neighbors who shared it.

**Outcome.**
- Veracity is revealed d steps after the item enters, for a share π of items.
  This is the fact-check delay and visibility.
- A sharer earns +1 for a true item and −c_i for a false one.
- A revealed outcome enters the outcome record with the item's features, the
  exposure context, and the sharer.

**Randomness.**
- Every stochastic draw is keyed to (seed, replicate, step, item, agent):
  item features, veracity, seed agents, the exploration floor, and sampled
  actions.
- Arms therefore share common random numbers, and an early random event is
  identical across arms in a replicate.

## Two regimes

| | Regime A: table expected to suffice | Regime B: computation during the run expected to matter |
|---|---|---|
| Policy | One pooled policy shared by all agents | One small learner per agent, warm-started from the prior |
| Evidence | The pooled public record | Each agent's own revealed shares, plus neighbor track records |
| Payoff | Common c | Heterogeneous c_i, so policies should diverge |
| Coupling | None beyond exposure | Co-adaptation: an exposure's value depends on neighbors' current policies |
| Network | Fixed | Fixed in v1. Unfollow after false shares (B+) is an open decision |

Regime B is where the original NABM design sits. It is also where per-agent
learners failed in Pivot for lack of evidence (C11). Before freezing, a
feasibility check must therefore show that agents in Regime B accumulate
enough revealed outcomes (stage D1).

## Arms: every policy is a table

Each arm is classified by who writes its table, what the table is indexed by,
and when it is written.

| Arm | Who writes the table | Indexed by | When |
|---|---|---|---|
| F0 threshold rule | Modeler | Exposure | Before the run |
| F1 hand rule with item features | Modeler | Item features + exposure | Before the run |
| P0 LLM prior, elicited once | LLM, distilled and calibrated to the base rate | Item features + exposure | Before the run |
| T0-final, precomputed table | Final policy of adaptive runs on donor seeds | Item features + exposure | Before the run |
| T0-traj, precomputed trajectory | Time-indexed policy of adaptive runs on donor seeds, averaged over 4 donors | Item features + exposure + time | Before the run |
| X0 LLM archetypes | LLM, queried per archetype and item class at each step | Archetype + item class + public aggregates | During the run |
| L0 NABM adaptive | Learning from this run's outcomes, starting from F1 | Item features + exposure + this run's history | During the run |
| L1 hybrid | P0 as the prior, then learning from this run's outcomes | Item features + exposure + this run's history | Before and during the run |

**Notes on the arms.**

- **T0 is the direct test of the critique.**
  - Its donors are adaptive runs on seeds disjoint from the evaluation seeds,
    in the same world.
  - Averaging four donors gives the table its best available form, rather
    than a single noisy realization.
  - A diagnostic, T0-self, replays a run's own trajectory under the same seed.
    It must reproduce that run exactly. This shows that after the fact every
    run is a table. The open question is whether a table written before the
    run can match it.
- **X0 follows the AgentTorch design.**
  - Agents are grouped into K archetypes by observable agent attributes.
  - Items are binned into item classes.
  - At each step, the LLM is queried once per (archetype, item class) pair
    that has a pending decision.
  - The prompt carries public aggregates: revealed true and false counts by
    item class. Agents sample from the archetype's probability.
  - X0 gets the same calibration as P0, a single log-odds shift to the base
    rate, and the same exploration floor.
  - Identical prompts reuse their logits, as in N06. K and the bins are fixed
    before any result.
- **Learning arms.**
  - Linear, tree, and MLP learners in Regime A.
  - MLP in Regime B, batched per agent with `neural_abm.accelerator` on the
    GPU server.
  - Each learning arm also runs with a prior-independent exploration floor ε,
    with keyed draws as in the Pivot pilots. LLM arms get the same floor for
    parity.

## Nesting checks (exact, registered before results)

1. F0 reproduces the classical threshold dynamics. This includes the
   Granovetter knife edge already verified on the all-to-all network.
2. L0 with learning off reproduces F1 byte for byte, and with F0 as its prior
   reproduces F0.
3. T0 with a constant table equal to F1 reproduces F1.
4. T0-self reproduces L0 under the same seed.
5. X0 with the LLM replaced by a stub that returns F1's probabilities
   reproduces F1. This checks the archetype machinery, as N06 did for the LLM
   agent.
6. An exploration adapter with ε = 0 reproduces its predecessor.

## Measures

**Behavior.**
- Share precision: the share of shares that are true.
- False reach: mean cascade size of false items.
- True reach.
- Sharer welfare.

**Learning.** Brier and AUC of each arm's policy on a probe item set that is
never simulated, scored with `neural_abm.policy_scoring`.

**Closed-loop effect.**
- The paired difference L0 − T0-traj on each behavior measure, per replicate.
- Path dependence:
  - Regress final false reach on an early exogenous shock across replicates.
    The shock is the share of false items among the first revealed outcomes,
    which is fixed by keyed draws and identical across arms.
  - Compare the slope between L0 and T0-traj. A closed-loop policy responds to
    the run's own early history; a precomputed table cannot.
- Seed-to-seed dispersion of the behavior measures.

**Cost.**
- CPU and GPU seconds per agent-step for every arm.
- LLM queries, tokens, and GPU seconds for P0 and X0.

## Proposed decision rules (fixed at freeze, not now)

**Closed loop is material in a cell when both hold:**
- L0 differs from T0-traj by at least 0.02 in share precision in at least 12
  of 16 paired replicates;
- the path-dependence slope difference has a bootstrap 95% interval
  excluding zero.

**Registered dispositions.**

| Disposition | Meaning |
|---|---|
| `AS_EXPECTED` | Material in most Regime B cells and in few Regime A cells |
| `TABLE_SUFFICES_EVEN_IN_B` | The critique holds in this domain |
| `CLOSED_LOOP_MATTERS_EVEN_IN_A` | Closed loop matters even with a pooled policy |
| `UNDERPOWERED` | Fewer than the required power in the D2 check |

**Comparators.** Rules and reporting follow v5:
- "matches" means each mean metric lies within the registered 0.02 margin;
- "superior" means at least 0.02 better in at least 12 of 16 paired
  replicates;
- 0 of n is reported with its one-sided 95% upper bound.

**Null worlds.** In the null worlds, no learning arm may beat F1 on probe
Brier. A violation is a defect in the design, not a finding.

## Reusability test (N14)

The study is built so that the Pivot ladder's semantics are tested by a
second consumer.

**Candidate semantics.**
- A policy slot that holds a fixed rule, a learned policy, or a
  prior-initialized policy.
- An outcome record with pooled and per-agent views, a visibility delay, and a
  visibility share.
- An exploration floor with keyed draws.
- A nesting-check harness.
- Out-of-sample probe scoring through `policy_scoring`.
- Precomputed-table replay, which is new.

**Method.**
1. Implement these semantics once as a repository-only core.
2. Use the core in the information-spread prototype.
3. Re-express one recorded Pivot configuration through the same core.

**Parity.** The re-expressed Pivot run must reproduce its recorded result:
byte for byte on the workstation where it was recorded, or decision for
decision on the GPU server, as the user chooses.

**Extraction.** Only semantics used by both consumers without
domain-specific branches are proposed for the package, through a new decision
record. Domain-owned parts stay outside, such as the item world, the
archetype prompt, and LLM elicitation.

## Compute plan

- **CPU arms:** non-LLM Regime A and small Regime B cells run on the GPU
  server's CPUs, with at most 40 workers.
- **GPU arms:** Regime B MLPs at scale use batched per-agent updates on GPUs 2
  and 3, with deterministic algorithms enforced.
- **LLM arms:** P0 elicitation and X0 run on GPUs 2 and 3 with the pinned HF
  weights of `Qwen/Qwen3-4B-Instruct-2507`, copied from the workstation.
- **Reproducibility:** development and reproduction runs use the same host,
  and every record names its host, GPU, driver, and library versions.

## Stages

Each stage starts only on the user's go-ahead.

| Stage | Content | Interprets results? |
|---|---|---|
| D0 | This brief | No |
| D1 | Prototype, nesting checks, smoke runs. Feasibility only: evidence per agent in Regime B, runtime, query budget for X0 | No |
| D2 | Synthetic power check for the closed-loop test at the chosen sizes, as N11a should have been done first | Power only |
| D3 | Freeze: worlds, cells, arms, rules, seeds | No |
| D4 | Non-LLM arms: development and reproduction runs, then record | Yes |
| D5 | P0 elicitation and X0 runs on the GPU server, then record | Yes |
| D6 | Consolidation, and the N14 extraction decision | Yes |

## Threats and how the design meets them

| Threat | Response |
|---|---|
| The worlds are designed by us, so learning recovers what we built | Null and simple stationary worlds where learning cannot help or should not be needed; report where NABM loses |
| Regime B starves for evidence, as per-agent Pivot learners did | D1 measures revealed outcomes per agent before freezing; D2 checks power |
| T0 is a strawman | Disjoint donor seeds, averaged over four donors, in the same world |
| X0 is a strawman | Same information, calibration, and exploration floor as the hybrid. K and bins fixed in advance |
| GPU nondeterminism | Deterministic algorithms; same-host reruns |
| Claims about real misinformation | None. The worlds are synthetic, and the study makes no claim about real spread |

## Open decisions for the user

1. Include endogenous unfollow (B+) in v1, or defer it with Toy 3 rewiring.
2. Sizes: N for Regimes A and B (proposal: 1,000), plus a GPU scale cell in
   Regime B (proposal: 10⁴ to 10⁵).
3. X0 design: the number of archetypes K and item classes. Whether a larger
   local Qwen runs as a sensitivity arm.
4. N14 parity: byte for byte on the workstation, or decision for decision on
   the GPU server.

## Boundary

This brief executes nothing, reads no empirical data, calls no LLM, changes
no package API, and establishes no result.
