# Section 8 Draft: Policy Ladder — Fixed, Adaptive, and Prior-Informed Policies

Status: draft prose candidate.

**Sources.**
- Claim-matrix rows P1 to P11.
- The hash-audited consolidation ledgers v1 to v9:
  `docs/research/pivot-policy-ladder-consolidation-v9.md` extends the others.
  Ledger ids are in parentheses.
- Decision 0017 for the package surface.

**Code.**
- The repository-only Pivot and information-spread ladder scripts in
  `scripts/`.
- The package modules `src/neural_abm/policy_ladder.py` and
  `src/neural_abm/policy_learners.py`.

**Figures.** Both are made from recorded results only.
- Figure 1, `paper/figures/info_spread_policy_ladder_comparisons.png`, by
  `scripts/plot_info_spread_policy_ladder_v1.py`.
- Figure 2, `paper/figures/info_spread_policy_ladder_follow_ups.png`, by
  `scripts/plot_info_spread_policy_ladder_follow_ups_v1.py`.

Every result in this section comes from synthetic worlds designed by the
authors. Each is a conditional case of the central claim, not a performance
claim about policy classes, language models, or real populations.

## 8.1 Purpose

The central claim of this paper is that NABM separates policy learning from
social interaction. So fixed policies, adaptive policies, and external priors
can be compared and traced within one simulation lifecycle. This section
shows the comparison running in two domains.

It also takes up a critique that any such framework invites. Every seeded run
is a deterministic ABM, and a frozen learned policy is a rule table. A policy
elicited once from a language model is a table written before the run. So
when, if ever, does an adaptive NABM policy differ from a table that could
have been written in advance?

## 8.2 The Ladder Design (P1)

A ladder holds these arms in one ABM design:

- **A fixed rule.**
- **Learned policies anchored at a prior.** Each learner predicts the prior
  exactly until it sees data, and then fits a residual to observed outcomes.
- **An external prior.**

The ladder adds one tier at a time. Before any learned arm is read, the new
tier with learning switched off must reproduce its rule-based predecessor
exactly.

**Fixed infrastructure.** Three things hold across both domains:

- **Keyed randomness.** Draws depend only on a tuple of integers, so arms
  that share a key share their draws, and paired arms differ only in their
  policies.
- **A prior-independent exploration floor.** A declined action is taken with
  probability ε on its own keyed stream.
- **An outcome record.** Each outcome becomes visible from a stated step,
  with pooled and per-agent views.

**Registration.** Every study is frozen before it runs and recorded after:
its design, cells, seeds, decision rule, and the hashes of its sources. A
development run and a reproduction run must match byte for byte.

In the first domain, the researcher-pivot ladder, every nesting check passed
exactly (P1). That includes the check that the LLM-agent decision machinery
without its persona reproduces the fixed-rule arm.

## 8.3 Policy Class as a Modeling Assumption (P2)

In the pivot worlds, the best policy class depends on the question.

- Feature-conditional learning is needed when the structure reverses (C07).
- Nonlinear policies beat linear ones on interactions (C08).
- The MLP wins smooth boundaries and the tree axis-aligned ones (C09).
- Every learner beats the prior with field-wide evidence and loses to it with
  neighbor-local evidence (C11).

The learned arms include linear and tree learners, so these results do not
establish that a neural policy is necessary (R01). Policy class is a modeling
assumption to be chosen per question, as Section 7.5 argues for policy
capacity.

## 8.4 External Priors From a Local Language Model (P3)

The external prior is elicited once:

1. A local model (Qwen3-4B-Instruct-2507) answers a fixed design cohort.
2. Its answers are distilled into a linear rule.
3. The rule's intercept is shifted to a stated base rate.

Used raw, the prior self-seals: agents that never act never produce the
outcomes that would correct them (C17). Shifted to a stated base rate, it
stops sealing (C19). Paired with a 20% exploration floor, the hybrid
dominated the hand rule in 7 of 9 cells (C22): it raised entrant precision by
at least the registered 0.02, and its pooled learning stayed within the registered
non-inferiority margin (probe Brier no more than 0.005 worse). On fresh seeds
it did so in 6 of 9 (C29). Within one model release, 4B was the smallest size that worked
(C26, C27).

## 8.5 Cost (P4)

On one machine and one design:

- **Pooled learners** cost 1.9 to 6.7 times the rule ABM per agent-step, and
  per-agent learners 57 to 65 times (C16).
- **The one-time elicitation** took under 10 GPU minutes (C25).
- **The every-step LLM agent** needed about 110 GPU-seconds per run, against
  about 5 CPU-seconds for an NABM learner run (C35).

In the second domain (§8.7), an AgentTorch-style archetype baseline needed:

- about 480 queries per run in Regime A and about 1,440 in Regime B, when
  queried during the run;
- 15,552 prompts once for its whole table, about 7.5 GPU minutes at batch 1.

These are costs of these designs, not benchmarks.

## 8.6 A Conditional LLM-Agent Comparison (P5)

In the specified synthetic worlds and comparison conditions, the NABM hybrid
met the registered performance criteria against two local LLM
implementations. These were an every-step forecaster agent (C32) and a
persona agent (C36), each in 9 of 9 cells.

- The forecaster over-entered once outcomes were public (C34).
- The persona said Yes in 0.15% of questions and self-sealed (C37).

**Asymmetries.** Both agents used the same 4B model, and neither was
recalibrated on public outcomes. The hybrid was calibrated to a base rate and
learned from outcomes. "Matched" means each mean metric lay within a 0.02
margin; it is not statistical equivalence.

## 8.7 A Second Domain: Information Spread (P7–P11)

The second domain answers two questions at once:

- Does the comparison carry over to another model family?
- When does an adaptive policy differ from a table written before the run?

**World.**
- **Network and items.** 1,000 agents share news items on a Watts–Strogatz
  graph (k = 6, p = 0.1). Each item has four features, and veracity is a
  function of them.
- **Fact-checks.** Half the items are fact-checked four steps after they
  appear, and fact-checks are public.
- **Decisions.** Agents share an item when its probability of being true
  reaches c/(1 + c). The exploration floor is ε = 0.02.
- **Regime A** has one pooled policy with c = 1.
- **Regime B** has per-agent costs drawn from {0.5, 1, 2}, a sharer-trust
  feature, and partial pooling: a pooled learner plus a per-agent linear
  residual.
- **Relations.** Three stationary veracity relations (linear, smooth,
  axis-aligned), plus a null world in which features carry no information.

**Arms.**

| Arm | What it is | Written |
|---|---|---|
| F0, F1 | The Granovetter threshold rule, and a hand rule on credibility and social proof (F1 is every learner's prior) | Before the run |
| T0-1 | One donor learner run's policy trajectory, replayed as a table | Before the run |
| T0-4 | The average of four donors' trajectories | Before the run |
| T0x-1, T0x-4 | The same tables from donors in other worlds of the same generator (§8.7.2) | Before the run |
| L0 | The adaptive learner (exact linear or MLP residuals on F1), refit every four steps | During the run |
| P0 | The once-elicited LLM prior, distilled and calibrated | Before the run |
| X0 | AgentTorch-style archetypes: one LLM answer per archetype, item class, and public fact-check aggregate | Before the run (§8.7.3) |
| L1 | The learner started from P0 instead of F1 | During the run |
| P0u, L1u | P0 read without the inherited clip, and the learner started from it (§8.7.3) | Before / during the run |
| L1_hcal, L0_pcal (and L1u_hcal, L0_ucal) | The learner started from the LLM prior's answers at F1's calibration, or from F1's answers at the LLM prior's calibration (§8.7.4) | During the run |
| L1_hcalw, L0_pcalw (and L1u_hcalw, L0_ucalw) | The same arms, with each recalibrated prior's location set in the world to share as many unverified items as its source (§8.7.4) | During the run |

Donors are runs of the same learner on disjoint seeds. For T0-1 and T0-4 they
ran in the same world. For T0x-1 and T0x-4 they ran in other worlds with
another network, other thresholds, and other costs, but the same veracity
relation.

**Design.**
- **Groups.** Two regimes × two learner families × four worlds give 16
  groups.
- **Size.** 20 cells per group, each with its own network, thresholds, and
  costs, and 16 paired replicates per cell.
- **Decision rule.** A comparison is *material* when the cell-resampled 95%
  interval of its group's mean cell difference excludes zero, and that mean
  is at least 0.01 in magnitude.

### 8.7.1 A registered null check caught a leak (P10)

The first registered study failed its own design check. In the null world,
where nothing is learnable, L0 beat F1 in one group (B/partial MLP), with a
gain in share precision of +0.027 [+0.025, +0.030]. As registered, its study
groups were not read.

An exploratory diagnosis placed the gain on items whose fact-check was
already public. The MLP recognized an item it had seen fact-checked from the
item's features (C50). Holding the trust feature constant left the gain in
place.

The primary measure became **share precision on unverified items**, whose
fact-check is not yet public at the decision. A control check then showed:

- the null control reproduces the leak on the old measure (+0.025, material);
- on the new measure the leak is gone;
- the closed-loop control effects survive (Figure 1, panel c).

The study was then frozen again with fresh seeds (R16).

### 8.7.2 When is a learned policy more than a table? (P7)

**In the stationary worlds** of the second study (Figure 1, panel a):

- **Learning matters.** Every learner beats the hand rule, by +0.040 to
  +0.129 in each of the 12 study groups.
- **The closed loop adds no detectable gain at matched evidence.** A single
  donor's trajectory, replayed as a table, is not detectably different from
  the adaptive learner in 11 of 12 groups. The exception is A/MLP/smooth,
  +0.011 [+0.001, +0.022]. The registered disposition is
  `TABLE_SUFFICES_EVEN_IN_B`.
- **The best table beats the adaptive learner in all 12 groups.** The
  four-donor average wins by 0.010 to 0.049 (C51).
- **The design's expectation fails.** It expected Regime B's per-agent costs
  and partial pooling to make the closed loop matter there. They did not
  (R17).

**When a run carries information that its donors lack,** the result
reverses. In a control where each run draws its own sign for the credibility
relation, L0 beats both tables in all four groups: the single donor by 0.126
to 0.152, and the four-donor average by 0.083 to 0.100 (C49).

**Donors from other worlds** (Figure 2, panel a).
- **The result.** A four-donor table whose donors ran in other worlds beats
  the learner in all 12 groups, by 0.012 to 0.054 (C56). Same-world donors
  added nothing material over other-world ones (0 of 12).
- **The transfer study's expectation fails.** It expected Regime B's per-agent
  residuals to keep the table from transferring. They did not (R18).
- **One explanation, not tested.** Costs enter only the decision cutoff,
  which comes from the evaluation world. So what is learned is a pooled
  belief, and a pooled belief transfers.

**Between the two cases** (Figure 2, panel b). Each run's credibility
relation was reversed with probability p, so a run carries information its
donors lack only when it is reversed.
- **At p = 0**, the four-donor table wins, as above.
- **At p = 0.1**, with 7.8% of runs reversed, the learner already beats the
  single-donor table in every family, by +0.034 to +0.036. Against the
  four-donor table the result is mixed.
- **From p = 0.25**, the learner beats the four-donor table in every family,
  by +0.050 to +0.069 (C57).

So the precomputed-table critique holds in stationary designs, whether the
donors ran in the same world or in other worlds of the same generator. It
fails once run-specific information is no longer rare. For a four-donor table
here, the boundary lies between a tenth and a quarter of runs carrying it.
- **What is not tested:** another kind of run-specific information, a larger
  table (O29), and another veracity relation or domain (O30).

### 8.7.3 LLM comparators (P8, P9)

**The archetype table.** The archetype baseline sees public fact-check
counts by item class. These depend on the item stream alone, never on any
agent's decision. So its table over archetype, item class, and aggregate
holds every answer it could need. Elicited once, it gives the same answers
as querying during the run with a prompt cache.

Given the same public information (Figure 1, panel b):

- **NABM against archetypes (P8).** The adaptive learner beats the
  archetypes in all 12 study groups, by 0.155 to 0.229 (C52). The archetypes
  also fall below the hand rule, sharing 1.4 to 1.7 times as many items at
  near base-rate precision.
- **An LLM prior inside the learner (P9).** On precision it adds nothing
  detectable. L1 − L0 is not material in any Regime B group and in four of
  six Regime A groups (C53).
  - In Regime B the prior alone beats the hand rule by +0.080 to +0.107, yet
    learning from the run's outcomes overrides the difference in starting
    point.
  - In Regime A the elicited prior saturated: 99% of its log-odds fell below
    the readout clip inherited from the pivot ladder. So P0/A is nearly
    constant, which is a property of the readout, not evidence about the
    model's knowledge.
- **The same answers without the clip (P9).**
  - **The prior.** Read without the clip, D5's first elicitation distils
    with R² 0.81 in Regime A and 0.76 in Regime B. Its credibility,
    novelty, and intensity weights have the world's signs. Alone, it beats
    the hand rule in all 12 groups, by +0.057 to +0.087 (C58).
  - **The prior inside the learner.** Started from it, the learner gains
    little on precision. L1u − L0 is positive in 5 of 12 groups, by +0.012
    to +0.020, and negative in none (C59).

### 8.7.4 Scoring by the agents' utility (P11)

Share precision rewards selectivity, and it is not what the agents optimize.

**The utility.** Each agent's decision rule maximizes its own utility:
- +1 for each true share;
- −c for each false share, where c is the sharer's cost.

**How the arms were rescored.** Every registered arm was scored by this
utility on unverified items, in the same cells and runs.
- **Regime A** was computed from the recorded counts.
- **Regime B** was rerun. Every rerun row reproduced its recorded counts
  exactly.
- **The minimum effect** was fixed before any utility was computed: 1% of
  F1's mean absolute utility in each regime.

**The framework comparisons hold (C60).**
- The best table still beats the learner in all 12 groups.
- The closed loop at matched evidence still shows no consistent gain.
- The learner still beats the hand rule.
- NABM still beats the archetypes in all 12 groups.

**The LLM-prior readings change (C61).** Five of 20 summaries change, and
each involves an LLM prior (Figure 2, panel c).
- **L1 − L0** lowers utility in 6 of 6 Regime A groups and 4 of 6 Regime B
  groups.
- **L1u − L0** lowers it in 4 of 6 Regime B groups, and raises it in the
  other 2.
- **In Regime B,** the closed loop on either prior raises utility over the
  prior alone, in 4 and 5 of 6 groups.

So "adds nothing detectable" holds on precision only. For the agents, the
LLM prior is not neutral (R19). The utility counts only the sharer's own
gain and cost, not the reach of false items (O31).

**Where the cost comes from.** Two follow-up studies split each LLM prior
into two parts:
- **its direction,** the model's answers;
- **its calibration,** its location and steepness over the design cohort.

The calibration step had set every LLM prior's mean probability over the
cohort to the base rate 0.4. F1 sits near 0.68 there.

Crossing the two parts against F1 gives two new learner arms per prior. One
starts from the LLM's answers at F1's calibration, the other from F1's
answers at the LLM's calibration. Each factor is then compared along both
routes from L0 to L1.

**The match must hold in the world.** The first study matched each
recalibrated prior to its source on the design cohort, and in the world it
overshot. At F1's calibration, an LLM direction shared 1.35 to 1.48 times as
much as F1 itself (C66). So its answers comparisons also carried a
difference in how much the starting prior shares. The second study kept each
recalibrated prior's weights and reset only its location, on calibration
worlds disjoint from the study cells, so that it shares as many unverified
items as its source. On the study cells, the world-calibrated priors then
shared 0.984 to 1.001 times as much as their sources (C72).

Below, each comparison is the arm with the LLM's part minus the arm with
F1's, under the world match. Each count is of study groups where the
comparison is material and negative. The calibration routes are given the
LLM's answers, then F1's; the answers routes are at F1's calibration, then
at the LLM's.

| Prior and regime | Calibration, along both routes | Answers, along both routes | Attribution | Under the cohort match |
|---|---|---|---|---|
| P0, Regime A | 6 and 6 of 6 negative | 5 of 6 negative; 0 of 6 | Calibration | Calibration |
| P0, Regime B | 4 and 4 of 6 negative | 6 and 5 of 6 negative | Both | Calibration |
| P0u, Regime B | 4 of 6 negative; mixed | 6 and 5 of 6 negative | Answers | Neither alone |

- **In Regime A,** the once-elicited prior's utility loss traces to how it
  was calibrated, not to the model's answers, under either match (C63, C68).
- **In Regime B, once the share rates match, the model's answers lower
  utility too.** For the once-elicited prior they do so beside its
  calibration (C69). For the unclipped prior they do so along both routes,
  and its calibration along only one (C70). The cohort match had read
  Regime B as calibration and as an
  interaction (C64). The second study had expected the once-elicited prior
  to stay at calibration in both regimes (R21).
- **How, descriptively.** In Regime B at matched calibration, a learner
  started from an LLM direction shares between 6% fewer and 0.3% more
  unverified items than one started from F1's direction, at a precision from
  0.013 lower to 0.001 higher. This was not judged.
- **No starting point improves on F1's.** Under either match, no
  LLM-informed start beats the learner started from F1 under utility, in
  any prior and regime (C65, C71).
- **Only the location was matched in the world.** The steepness stayed
  matched on the cohort (O34).

### 8.7.5 Reuse

The second domain ran through a domain-free core lifted from the pivot
ladder. Its arms nest exactly, with 14 checks (C47).

Re-expressed through the same core, a recorded pivot study reproduced its
recorded output byte for byte: 1,152 of 1,152 rows (C48).

The shared semantics are now package API, released in the `v0.1.0a6` tag
(Decision 0017):

- **torch-free:** the prior rule, keyed randomness, the exploration floor,
  the outcome record, and prior-boosted trees;
- **torch-backed:** the residual learners.

Through the package, the recorded pivot study reproduces byte for byte. So
do the delayed field-public arms of a second recorded pivot study. With
delays, the record's ordering rule (visible step, then insertion) recovers
the pivot ladder's release order on its own.

**What stays with each domain.**
- **Its fixed rule.** A rule computed elsewhere can differ in the last bits.
- **Its agents' own parameters, worlds, and measures.**

## 8.8 The Empirical Stage as a Limitation (P6)

The registration discipline also stopped an empirical test before any
confirmatory outcome was read. At the measured data shape, the registered
analysis recovered learning in only 1 of 8 replicates (W1) and 3 of 8
(confounded W2), against a required 7 of 8. No false recovery was observed:
0 of 8 per world, which bounds a world's false-recovery rate only below about
31%. The explored alternatives did not achieve power and false-recovery
control together.

The closure statement applies: in synthetic worlds that reflect the measured
data structure, the registered analysis did not meet the power criterion
needed to distinguish learning from anticipation. This does not imply the
absence of learning.

## 8.9 Reporting Rules

These rules apply wherever this section's results are cited.

- **0 of n** is reported with its one-sided 95% upper bound (about 31% at
  n = 8).
- **"Matches"** means each mean metric lies within the registered 0.02
  margin. It is not statistical equivalence.
- **LLM results** are conditional on one local 4B model, its prompts, and
  its readouts. They reflect calibration and decision rules as well as the
  policy family.
- **Neural necessity** is not established.
- **Information-spread comparisons** are on share precision on unverified
  items. The first study's groups are not read.
- **"A table suffices"** holds for donors in the same world or in other
  worlds of the same generator, and while run-specific information is rare.
  Transfer across veracity relations or domains is not tested.
- **Share precision** rewards selectivity and is not the agents' utility.
  Comparisons that involve an LLM prior are reported on both, because they
  disagree in 5 of 20 summaries.
- **An LLM prior's calibration** is named with any result about it: the
  rule that set its location, and its steepness. Its utility cost is
  attributed per regime.
- **A recalibrated prior** is matched to its source's share rate in the
  world before a cost is attributed to its parts, and the match is reported.
  A match on a design cohort need not hold in the world.

## 8.10 What This Section Claims

Four statements are supported, each with its boundary.

1. **Fixed rules, adaptive policies, tables written before the run, and
   external priors can be compared in one lifecycle, in two model families.**
   Each tier nests its predecessor exactly.
   - *Limitation:* synthetic worlds designed by the authors.
2. **The comparison machinery is reusable.**
   - Both domains run through the same semantics.
   - A recorded study of the first domain reproduces byte for byte through
     the package, as do the delayed arms of a second.
   - *Limitation:* two consumers and two recorded configurations, on one
     host. The surface first ships in the `v0.1.0a6` tag.
3. **The lifecycle makes the table critique testable.**
   - In stationary designs, a table written before the run suffices, whether
     its donors ran in the same world or in other worlds of the same
     generator.
   - Once about a quarter of runs carry information that donors lack, the
     adaptive policy wins.
   - These conclusions hold under the agents' utility as well as precision.
   - *Limitations:* one generator and veracity relation, one kind of
     run-specific information, and one table size.
4. **Conditional comparisons with local language models.**
   - The NABM learner beats AgentTorch-style archetypes given the same
     public information, on precision and on utility.
   - A once-elicited prior adds nothing detectable inside the learner on
     precision, and lowers the agents' utility. With share rates matched
     in the world, the loss traces to how the prior was calibrated in one
     regime; in the other, the model's answers carry part of it.
   - The hybrid met its registered criteria against two LLM agents.
   - *Limitations:*
     - one small model and fixed prompts;
     - a readout that matters: the inherited clip saturated one regime;
     - asymmetric recalibration in the agent comparison;
     - a utility that ignores harm to others;
     - a calibration split whose steepness is matched only on a design
       cohort.

The registration discipline is part of the evidence. It caught one leak and
closed one underpowered test before either could be read as a result. A
second measure, with its minimum effect fixed before it was computed, showed
which readings depend on precision.
