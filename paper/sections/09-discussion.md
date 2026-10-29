# Section 9 Draft: Discussion

Status: draft prose candidate.

**Sources.**
- The Primary Claim Path and policy-ladder rows P1 to P11 of
  [claim-matrix.md](../claim-matrix.md).
- The consolidation ledgers v1 to v9:
  `docs/research/pivot-policy-ladder-consolidation-v9.md` extends the others.
- Decisions 0001 (project positioning) and 0003 (social update pipeline).

## 9.1 What the Unit and the Toys Show

The unit contract carries binary policy, readiness, and diagnostic
lifecycles across three toy families and two source-free adapter domains, a
capacity-allocation holdout and a stochastic commons. Domain semantics stay
in the adapters throughout. This is an engineering architecture claim.

Where a toy result looks like a win, it is narrower than it looks.
- Threshold-aware readiness kept no-seed safety and recovered full cascades
  on a bounded Toy5 grid where output averaging stalled (5 of 5 against 0 of
  5). But exposure-anchor negative controls also spread in seeded cases, so
  the result separates safety from spread rather than showing that the
  threshold-aware direction is unique.
- Precommitment with peer evidence survived sparse seeds, open boundaries,
  and noisy ranking that broke reputation imitation (5 of 5 against 0 of 5).
  But that stress targets the baseline's own information channel, and clean
  reputation imitation remains faster in Toy4.

The toys therefore show reach and diagnosability, not dominance over
hand-coded rules.

## 9.2 What the Research Application Shows

The same contract carries a researcher-authored question end to end. Study 1
reports paired mean effects with intervals kept separate from the empirical
replicate intervals, and an exact true-null control. Study 2 shows that the
direction of learning is a first-class experimental axis. Population,
network, channel, commit adapter, and seeds are held fixed, and only the
learning rule differs. Imitative learning raises failed pivots at baseline
and under grants but not under hype. Failure-only learning lowers failures
under hype and also lowers productive pivots. The domain model is hand-set
and uncalibrated, so these are statements about the lifecycle, not about
science.

## 9.3 What the Policy Ladder Shows

Every tier of the ladder nests its predecessor exactly. Each comparison is
therefore a difference in one component of the design, and the question
becomes when each tier is worth its cost.

- **Pooled learning** costs 1.9 to 6.7 times the rule ABM per agent-step.
  Whether it helps depends on the world's geometry and on how much evidence
  the agents pool. Per-agent learners cost about 60 times as much, and with
  neighbor-scale evidence they do worse than the rule they start from.
- **An external prior** is cheap when consulted once. A local 4B model,
  elicited in under 10 GPU minutes, calibrated to a stated base rate, and
  paired with an exploration floor, selected better entrants than the hand
  rule while keeping pooled learning within a registered margin, in 7 of 9
  world and learner settings. Used raw, the same prior self-sealed: agents
  that never act produce no evidence to learn from.
- **An every-step LLM agent** of the same model needed about 110 GPU-seconds
  per run, against about 5 CPU-seconds for an NABM learner run.

Which tier is worth its cost depends on the world. The lifecycle makes that
dependence something to measure rather than assume.

## 9.4 When NABM Is More Than a Precomputed Table

A fixed policy is a table, and so is a policy frozen after a donor run or an
LLM judgment elicited once. What an adaptive policy adds is dependence on the
run's own outcome history. The second domain tested when that matters.

- **In stationary designs it does not.** A four-donor table written before
  the run beat the adaptive learner in all 12 study groups. So did a
  four-donor table whose donors ran in other worlds of the same generator.
- **It matters when a run carries information that no table written before
  the run can hold.** Here the closed loop beat the best table once about a
  quarter of runs carried such information.

This suggests that a neural ABM's adaptivity should be justified by
run-specific information, with a table arm as the control that tests it.
The evidence covers one generator, one veracity relation, one kind of
run-specific information, and one table size.

## 9.5 Tracing an External Prior

Share precision rewards selectivity, and it is not what the agents
optimize. Under the agents' own utility, an LLM prior that looked neutral
inside the learner lowered utility.

The lifecycle let the prior be taken apart. Its direction, the model's
answers, was crossed with its calibration against the hand rule, one part at
a time, with every other part of the run held fixed.
- **Matched on the design cohort,** the split traced the loss of the
  once-elicited prior to its calibration in both regimes. But the match
  overshot in the world: at the hand rule's calibration, an LLM direction
  shared 1.35 to 1.48 times as much as the hand rule itself.
- **Matched in the world,** the priors shared 0.984 to 1.001 times as much
  as their sources. The attribution then narrowed. The loss is the prior's
  calibration in one regime, and both its calibration and the model's
  answers in the other.

One lesson carries beyond these worlds as a reporting rule. The
attribution depended on what the calibration was matched to, so the match
should be reported with the result. The prior slot made the correction
possible: it let the location be reset in the world with the direction, the
steepness, and the rest of the run held fixed.

## 9.6 What the Registration Discipline Adds

Freezing each design before its results were read changed what this paper
can say.
- A registered null check caught a leak before any study group was read:
  learners recognized items whose fact-check was already public. The measure
  moved to unverified items.
- The empirical test of learning was closed before confirmatory outcomes
  were read, because it lacked power at the measured data shape.
- Expectations that failed were recorded as rejected, and earlier claims
  were narrowed when later studies required it. One example is the prior
  attribution above.

These are part of the evidence. They show which readings survive.

## 9.7 Limits of the Evidence

- The worlds are synthetic and were designed by the authors.
- The empirical test of learning closed without an answer. This does not
  imply the absence of learning.
- The LLM comparisons cover one model family and fixed prompts, and one of
  its readouts saturated under an inherited clip.
- The table comparisons use one generator and veracity relation.
- The utility scores only the sharer's own gain and cost, not the reach of
  false items.
- Neural necessity is not established.
- The policy-ladder designs use 256 to 1,000 agents per run.

## 9.8 Limits of the Transformer Analogy and of Parameter Averaging

An early framing treated the NABM node as a minimal unit of intelligence,
with peer mixing read by analogy to token mixing in Transformers or MLP-Mixers.
The analogy helps describe peer mixing as an aggregation over selected
neighbors, but it is technically limited. NABM is positioned instead as a
temporal heterogeneous graph simulation with neural agents, social mixing
functions, local learning, and explicit scheduling and logging. It is not a
Transformer-grade universal primitive.

Parameter-level social updates belong with federated averaging and should be
read through it. In the Toy 1 diagnostics:
- raw averaging of independently initialized agents fragmented under high
  raw parameter-similarity thresholds;
- aligning hidden units before averaging helped only when the peer graph was
  already connected;
- aligned peer selection changed the peer graph itself.

An observed effect of "social learning" can therefore come from peer
selection, alignment, the mixer, or their interaction. The unit keeps these
stages separately configurable and logged so that they can be told apart.

## 9.9 Toward Richer Environments

The next steps follow from the limits:
- decisions richer than binary;
- calibration against empirical data, through the separate pivot-feedback
  protocol rather than by enlarging Section 7;
- table comparisons across veracity relations and domains;
- a utility that charges an agent for the reach of its false shares;
- a prior whose steepness, and not only its location, is matched in the
  world;
- larger populations.

Each can be added as a new arm or a new domain adapter without changing the
lifecycle.
